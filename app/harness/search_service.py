from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.core.hard_filters import HardFilterParams, search_listings
from app.models.schemas import HardFilters, ListingsResponse
from app.participant.hard_fact_extraction import extract_hard_facts
from app.participant.ranking import rank_listings
from app.participant.soft_fact_extraction import extract_soft_facts
from app.participant.soft_filtering import filter_soft_facts


def _geocode_place(place: str) -> tuple[float, float] | None:
    try:
        import httpx
        resp = httpx.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": place, "format": "json", "limit": 1, "countrycodes": "ch"},
            headers={"User-Agent": "datathon2026-geo/1.0"},
            timeout=5.0,
        )
        hits = resp.json()
        if hits:
            return float(hits[0]["lat"]), float(hits[0]["lon"])
    except Exception:
        pass
    return None


def _resolve_near_place(hard_facts: HardFilters) -> HardFilters:
    if hard_facts.near_place and hard_facts.latitude is None:
        coords = _geocode_place(hard_facts.near_place)
        if coords:
            hard_facts.latitude = coords[0]
            hard_facts.longitude = coords[1]
            if hard_facts.radius_km is None:
                hard_facts.radius_km = 2.0
    return hard_facts


def filter_hard_facts(db_path: Path, hard_facts: HardFilters) -> list[dict[str, Any]]:
    return search_listings(db_path, to_hard_filter_params(hard_facts))


def query_from_text(
    *,
    db_path: Path,
    query: str,
    limit: int,
    offset: int,
) -> ListingsResponse:
    hard_facts = extract_hard_facts(query)
    hard_facts.limit = 1000
    hard_facts.offset = 0
    hard_facts = _resolve_near_place(hard_facts)
    soft_facts = extract_soft_facts(query)
    if hard_facts.neighborhood:
        soft_facts["neighborhoods"] = hard_facts.neighborhood
    candidates, _ = search_with_relaxation(db_path, to_hard_filter_params(hard_facts))
    candidates = filter_soft_facts(candidates, soft_facts)
    ranked = rank_listings(candidates, soft_facts)
    return ListingsResponse(
        listings=ranked[offset : offset + limit],
        meta={"total_candidates": len(candidates), "returned": min(limit, len(ranked))},
    )


def query_from_filters(
    *,
    db_path: Path,
    hard_facts: HardFilters | None,
) -> ListingsResponse:
    structured_hard_facts = hard_facts or HardFilters()
    soft_facts = extract_soft_facts("")
    candidates = filter_hard_facts(db_path, structured_hard_facts)
    candidates = filter_soft_facts(candidates, soft_facts)
    return ListingsResponse(
        listings=rank_listings(candidates, soft_facts),
        meta={},
    )


def _rooms_list(center: float, delta: float) -> list[float]:
    """Generate discrete Swiss room values around center ± delta (step 0.5)."""
    lo = center - delta
    hi = center + delta
    step = 0.5
    val = round(lo / step) * step
    result = []
    while val <= hi + 1e-9:
        result.append(round(val, 1))
        val = round(val + step, 1)
    return result


def _cities_in_canton(city_names: list[str], db_path: Path) -> str | None:
    """Return dominant canton for given cities (majority by listing count)."""
    try:
        con = sqlite3.connect(str(db_path))
        placeholders = ",".join("?" * len(city_names))
        rows = con.execute(
            f"SELECT canton, COUNT(*) n FROM listings "
            f"WHERE city IN ({placeholders}) AND canton IS NOT NULL "
            f"GROUP BY canton ORDER BY n DESC",
            city_names,
        ).fetchall()
        con.close()
        return rows[0][0].upper() if rows else None
    except Exception:
        return None


MIN_HARD = 10


def _relax_rooms(p: HardFilterParams) -> HardFilterParams | None:
    if p.min_rooms is None and p.max_rooms is None:
        return None
    lo = p.min_rooms if p.min_rooms is not None else 0.5
    hi = p.max_rooms if p.max_rooms is not None else lo
    return replace(p, rooms_values=_rooms_list(lo, (hi - lo) / 2 + 1.0),
                   rooms_allow_null=True, min_rooms=None, max_rooms=None)


def _relax_area(p: HardFilterParams) -> HardFilterParams | None:
    if p.min_area is None and p.max_area is None:
        return None
    kwargs: dict = {"area_allow_null": True}
    if p.min_area is not None:
        kwargs["min_area"] = int(p.min_area * 0.8)
    if p.max_area is not None:
        kwargs["max_area"] = int(p.max_area * 1.2)
    return replace(p, **kwargs)


def _relax_features(p: HardFilterParams) -> HardFilterParams | None:
    if not p.features:
        return None
    n = len(p.features)
    return replace(p, features_min_match=n - max(1, int(n * 0.3)))


def _relax_price(p: HardFilterParams) -> HardFilterParams | None:
    if p.max_price is None:
        return None
    return replace(p, max_price=int(p.max_price * 1.15))


def _relax_available_from(p: HardFilterParams) -> HardFilterParams | None:
    if p.available_from is None:
        return None
    try:
        relaxed = datetime.strptime(p.available_from, "%Y-%m-%d") + timedelta(weeks=4)
        return replace(p, available_from=relaxed.strftime("%Y-%m-%d"))
    except ValueError:
        return None


def _relax_radius(p: HardFilterParams) -> HardFilterParams | None:
    if p.radius_km is None:
        return None
    return replace(p, radius_km=p.radius_km * 1.25)


def _relax_city_to_canton(p: HardFilterParams, db_path: Path) -> HardFilterParams | None:
    if not p.city or p.canton:
        return None
    canton = _cities_in_canton(p.city, db_path)
    if not canton:
        return None
    return replace(p, city=None, canton=canton)


_RELAXATIONS = [
    ("rooms",           _relax_rooms),
    ("area",            _relax_area),
    ("features",        _relax_features),
    ("price",           _relax_price),
    ("available_from",  _relax_available_from),
    ("radius",          _relax_radius),
]


def search_with_relaxation(
    db_path: Path, params: HardFilterParams
) -> tuple[list[dict[Any, Any]], list[str] | None]:
    """Returns (results, relaxations_applied).

    relaxations_applied is None when strict match is sufficient (≥ MIN_HARD).
    """
    from itertools import combinations

    canton_fn = ("city_to_canton", lambda p, db=db_path: _relax_city_to_canton(p, db))
    all_fns = _RELAXATIONS + [canton_fn]

    results = search_listings(db_path, params)
    if len(results) >= MIN_HARD:
        return results, None

    best, best_names = results, []
    applicable = []
    for name, fn in all_fns:
        relaxed = fn(params)
        if relaxed is None:
            continue
        applicable.append((name, fn, relaxed))
        r = search_listings(db_path, relaxed)
        if len(r) > len(best):
            best, best_names = r, [name]
        if len(best) >= MIN_HARD:
            return best, best_names

    # Two-condition relaxation
    for (n1, fn1, r1), (n2, fn2, _) in combinations(applicable, 2):
        combined = fn2(r1)
        if combined is None:
            continue
        r = search_listings(db_path, combined)
        if len(r) > len(best):
            best, best_names = r, [n1, n2]
        if len(best) >= MIN_HARD:
            return best, best_names

    return best, best_names if best_names else None


_NON_RESIDENTIAL = ["Gewerbeobjekt", "Parkplatz", "Einzelgarage", "Tiefgarage", "Bastelraum"]


def to_hard_filter_params(hard_facts: HardFilters) -> HardFilterParams:
    # Exclude commercial/parking categories unless the query explicitly requests them
    exclude = None if hard_facts.object_category else _NON_RESIDENTIAL
    return HardFilterParams(
        city=hard_facts.city,
        postal_code=hard_facts.postal_code,
        canton=hard_facts.canton,
        min_price=hard_facts.min_price,
        max_price=hard_facts.max_price,
        min_rooms=hard_facts.min_rooms,
        max_rooms=hard_facts.max_rooms,
        min_area=hard_facts.min_area,
        max_area=hard_facts.max_area,
        available_from=hard_facts.available_from,
        latitude=hard_facts.latitude,
        longitude=hard_facts.longitude,
        radius_km=hard_facts.radius_km,
        features=hard_facts.features,
        offer_type=hard_facts.offer_type,
        object_category=hard_facts.object_category,
        exclude_object_category=exclude,
        limit=hard_facts.limit,
        offset=hard_facts.offset,
        sort_by=hard_facts.sort_by,
    )

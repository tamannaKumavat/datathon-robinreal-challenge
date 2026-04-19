from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import process as fuzz_process

from app.db import get_connection

# Loaded once on first use
_DB_CITIES: list[str] = []
_DB_CITIES_NORMALIZED: list[str] = []


def _normalize(text: str) -> str:
    """Strip diacritics for accent-insensitive comparison (Zürich → Zurich)."""
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii").lower()


def _load_db_cities(db_path: Path) -> None:
    global _DB_CITIES, _DB_CITIES_NORMALIZED
    if _DB_CITIES:
        return
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT city FROM listings WHERE city IS NOT NULL").fetchall()
    _DB_CITIES = [r[0] for r in rows if r[0]]
    _DB_CITIES_NORMALIZED = [_normalize(c) for c in _DB_CITIES]


def _resolve_cities(name: str, db_path: Path) -> list[str]:
    """Return all DB city variants matching the input.

    First collects all cities whose normalized form exactly matches the input
    (handles Zurich↔Zürich, zurich↔Zürich, Geneve↔Genève).
    Falls back to high-threshold fuzzy match only when no exact normalized match exists.
    """
    _load_db_cities(db_path)
    normalized_name = _normalize(name)

    # Exact normalized match — catches all unicode/case variants of the same city
    exact = [_DB_CITIES[i] for i, n in enumerate(_DB_CITIES_NORMALIZED) if n == normalized_name]
    if exact:
        return exact

    # Fuzzy fallback for genuine typos/abbreviations (high threshold to avoid false positives)
    threshold = 90 if len(name) <= 5 else 85
    matches = fuzz_process.extract(normalized_name, _DB_CITIES_NORMALIZED, limit=5)
    result = [_DB_CITIES[idx] for _, score, idx in matches if score >= threshold]
    return result if result else [name]


@dataclass(slots=True)
class HardFilterParams:
    city: list[str] | None = None
    postal_code: list[str] | None = None
    canton: str | None = None
    min_price: int | None = None
    max_price: int | None = None
    min_rooms: float | None = None
    max_rooms: float | None = None
    rooms_values: list[float] | None = None   # if set, overrides min/max_rooms with IN list
    rooms_allow_null: bool = False             # include listings where rooms IS NULL
    min_area: int | None = None
    max_area: int | None = None
    area_allow_null: bool = False              # include listings where area IS NULL
    available_from: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    radius_km: float | None = None
    features: list[str] | None = None
    features_min_match: int | None = None  # if set, require at least N features to match (SUM); else AND all
    offer_type: str | None = None
    object_category: list[str] | None = None
    exclude_object_category: list[str] | None = None
    limit: int = 20
    offset: int = 0
    sort_by: str | None = None


FEATURE_COLUMN_MAP = {
    "balcony": "feature_balcony",
    "elevator": "feature_elevator",
    "parking": "feature_parking",
    "garage": "feature_garage",
    "fireplace": "feature_fireplace",
    "child_friendly": "feature_child_friendly",
    "pets_allowed": "feature_pets_allowed",
    "temporary": "feature_temporary",
    "new_build": "feature_new_build",
    "wheelchair_accessible": "feature_wheelchair_accessible",
    "private_laundry": "feature_private_laundry",
    "minergie_certified": "feature_minergie_certified",
    "furnished": "feature_furnished",
    "garden": "feature_garden",
}


def _normalize_list(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    cleaned = [value.strip() for value in values if value and value.strip()]
    return cleaned or None


def search_listings(db_path: Path, filters: HardFilterParams) -> list[dict[str, Any]]:
    
    where_clauses: list[str] = []
    params: list[Any] = []

    city = _normalize_list(filters.city)
    canton = filters.canton.upper() if filters.canton else None

    if city:
        resolved = [c for name in city for c in _resolve_cities(name, db_path)]
        placeholders = ", ".join("?" for _ in resolved)
        where_clauses.append(f"city IN ({placeholders})")
        params.extend(resolved)
    elif canton:
        where_clauses.append("UPPER(canton) = ?")
        params.append(canton)

    postal_code = _normalize_list(filters.postal_code)
    if postal_code:
        placeholders = ", ".join("?" for _ in postal_code)
        where_clauses.append(f"postal_code IN ({placeholders})")
        params.extend(postal_code)

    if filters.min_price is not None:
        where_clauses.append("price >= ?")
        params.append(filters.min_price)

    if filters.max_price is not None:
        where_clauses.append("price <= ?")
        params.append(filters.max_price)

    if filters.rooms_values is not None:
        placeholders = ", ".join("?" * len(filters.rooms_values))
        null_clause = "rooms IS NULL OR " if filters.rooms_allow_null else ""
        where_clauses.append(f"({null_clause}rooms IN ({placeholders}))")
        params.extend(filters.rooms_values)
    else:
        null_clause = "rooms IS NULL OR " if filters.rooms_allow_null else ""
        if filters.min_rooms is not None:
            where_clauses.append(f"({null_clause}rooms >= ?)")
            params.append(filters.min_rooms)
        if filters.max_rooms is not None:
            where_clauses.append(f"({null_clause}rooms <= ?)")
            params.append(filters.max_rooms)

    null_clause = "area IS NULL OR " if filters.area_allow_null else ""
    if filters.min_area is not None:
        where_clauses.append(f"({null_clause}area >= ?)")
        params.append(filters.min_area)
    if filters.max_area is not None:
        where_clauses.append(f"({null_clause}area <= ?)")
        params.append(filters.max_area)

    if filters.available_from is not None:
        where_clauses.append("(available_from IS NULL OR available_from <= ?)")
        params.append(filters.available_from)

    if filters.offer_type:
        where_clauses.append("UPPER(offer_type) = ?")
        params.append(filters.offer_type.upper())

    object_category = _normalize_list(filters.object_category)
    if object_category:
        # Use LIKE so "Wohnung" also matches "Möblierte Wohnung", "Dachwohnung" etc.
        like_clauses = " OR ".join("object_category LIKE ?" for _ in object_category)
        where_clauses.append(f"({like_clauses})")
        params.extend(f"%{c}%" for c in object_category)

    exclude_category = _normalize_list(filters.exclude_object_category)
    if exclude_category:
        not_like_clauses = " AND ".join("(object_category IS NULL OR object_category NOT LIKE ?)" for _ in exclude_category)
        where_clauses.append(f"({not_like_clauses})")
        params.extend(f"%{c}%" for c in exclude_category)

    features = _normalize_list(filters.features)
    if features:
        cols = [FEATURE_COLUMN_MAP[f] for f in features if f in FEATURE_COLUMN_MAP]
        if cols:
            if filters.features_min_match is not None:
                # Soft: require at least N features to match (NULL counts as 0)
                sum_expr = " + ".join(f"COALESCE({c}, 0)" for c in cols)
                where_clauses.append(f"({sum_expr}) >= {filters.features_min_match}")
            else:
                # Hard: all features must match (NULL = unknown → include)
                for col in cols:
                    where_clauses.append(f"({col} = 1 OR {col} IS NULL)")

    query = """
        SELECT
            listing_id,
            title,
            description,
            street,
            city,
            postal_code,
            canton,
            price,
            rooms,
            area,
            available_from,
            latitude,
            longitude,
            distance_public_transport,
            distance_shop,
            distance_kindergarten,
            distance_school_1,
            distance_school_2,
            features_json,
            offer_type,
            object_category,
            object_type,
            original_url,
            images_json
        FROM listings
    """

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += " ORDER BY " + _sort_clause(filters.sort_by)

    with get_connection(db_path) as connection:
        rows = connection.execute(query, params).fetchall()

    parsed_rows = [_parse_row(dict(row)) for row in rows]

    if (
        filters.latitude is not None
        and filters.longitude is not None
        and filters.radius_km is not None
    ):
        nearby_rows: list[tuple[float, dict[str, Any]]] = []
        for row in parsed_rows:
            if row.get("latitude") is None or row.get("longitude") is None:
                continue
            distance = _distance_km(
                filters.latitude,
                filters.longitude,
                row["latitude"],
                row["longitude"],
            )
            if distance <= filters.radius_km:
                nearby_rows.append((distance, row))

        nearby_rows.sort(key=lambda item: (item[0], item[1]["listing_id"]))
        parsed_rows = [row for _, row in nearby_rows]

    return parsed_rows[filters.offset : filters.offset + filters.limit]


def _parse_row(row: dict[str, Any]) -> dict[str, Any]:
    features_json = row.pop("features_json", "[]")
    images_json = row.pop("images_json", None)
    try:
        row["features"] = json.loads(features_json) if features_json else []
    except json.JSONDecodeError:
        row["features"] = []
    row["image_urls"] = _extract_image_urls(images_json)
    row["hero_image_url"] = row["image_urls"][0] if row["image_urls"] else None
    return row


def _extract_image_urls(images_json: Any) -> list[str]:
    if not images_json:
        return []
    try:
        parsed = json.loads(images_json) if isinstance(images_json, str) else images_json
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, dict):
        return []

    image_urls: list[str] = []
    for item in parsed.get("images", []) or []:
        if isinstance(item, dict) and item.get("url"):
            image_urls.append(str(item["url"]))
        elif isinstance(item, str) and item:
            image_urls.append(item)
    for item in parsed.get("image_paths", []) or []:
        if isinstance(item, str) and item:
            image_urls.append(item)
    return image_urls


def _distance_km(
    center_lat: float,
    center_lon: float,
    row_lat: float,
    row_lon: float,
) -> float:
    earth_radius_km = 6371.0
    delta_lat = math.radians(row_lat - center_lat)
    delta_lon = math.radians(row_lon - center_lon)
    start_lat = math.radians(center_lat)
    end_lat = math.radians(row_lat)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(start_lat) * math.cos(end_lat) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius_km * c


def _sort_clause(sort_by: str | None) -> str:
    if sort_by == "price_asc":
        return "price ASC NULLS LAST, listing_id ASC"
    if sort_by == "price_desc":
        return "price DESC NULLS LAST, listing_id ASC"
    if sort_by == "rooms_asc":
        return "rooms ASC NULLS LAST, listing_id ASC"
    if sort_by == "rooms_desc":
        return "rooms DESC NULLS LAST, listing_id ASC"
    return "listing_id ASC"

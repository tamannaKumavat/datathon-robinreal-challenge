from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_HEADERS = {"User-Agent": "datathon2026-listings/1.0"}


@dataclass
class AmenityConfig:
    triggers: list[str]
    osm_tags: list[tuple[str, str]]  # (element_type, osm_tag_filter)


# Declarative config: add entries here to support new amenity types.
# OSM tag filters use Overpass QL syntax, e.g. '"amenity"="school"'
_AMENITY_CONFIG: dict[str, AmenityConfig] = {
    "lake": AmenityConfig(
        triggers=["near_lake"],
        osm_tags=[
            ("way",      '"natural"="water"["water"~"lake|reservoir"]'),
            ("relation", '"natural"="water"["water"~"lake|reservoir"]'),
        ],
    ),
    "school": AmenityConfig(
        triggers=["good_schools_nearby", "family_friendly"],
        osm_tags=[
            ("node", '"amenity"~"school|kindergarten"'),
            ("way",  '"amenity"~"school|kindergarten"'),
        ],
    ),
    "transport": AmenityConfig(
        triggers=["close_to_transport"],
        osm_tags=[
            ("node", '"public_transport"="stop_position"'),
            ("node", '"railway"~"station|halt|tram_stop"'),
        ],
    ),
    "park": AmenityConfig(
        triggers=["quiet", "family_friendly"],
        osm_tags=[
            ("way",  '"leisure"~"park|playground|nature_reserve"'),
            ("node", '"leisure"~"park|playground"'),
        ],
    ),
    "shop": AmenityConfig(
        triggers=["walkable"],
        osm_tags=[
            ("node", '"shop"~"supermarket|convenience|mall"'),
        ],
    ),
}

# City center distance is a proxy for urban density — only useful for these signals.
_CITY_CENTER_TRIGGERS = {"quiet", "not_urban"}


def filter_soft_facts(
    candidates: list[dict[str, Any]],
    soft_facts: dict[str, Any],
) -> list[dict[str, Any]]:
    if not candidates:
        return candidates

    _enrich_commute_distance(candidates, soft_facts)

    return candidates


# --- Helpers ---

def _candidate_coords(candidate: dict[str, Any]) -> tuple[float, float] | None:
    lat, lon = candidate.get("latitude"), candidate.get("longitude")
    return (lat, lon) if lat is not None and lon is not None else None


# --- Commute destination ---

def _enrich_commute_distance(candidates: list[dict[str, Any]], soft_facts: dict[str, Any]) -> None:
    destination = soft_facts.get("commute_destination")
    if not destination:
        return
    dest_coords = _geocode(destination)
    if not dest_coords:
        return
    for candidate in candidates:
        coords = _candidate_coords(candidate)
        candidate["computed_distance_to_destination_km"] = (
            round(_haversine(*coords, *dest_coords), 3) if coords else None
        )


# --- City center distance (urban density proxy) ---

def _enrich_city_center_distance(candidates: list[dict[str, Any]]) -> None:
    cities = {c["city"] for c in candidates if c.get("city") and _candidate_coords(c)}
    city_centers = {city: coords for city in cities if (coords := _geocode(city))}

    for candidate in candidates:
        coords = _candidate_coords(candidate)
        city = candidate.get("city")
        if not coords or not city:
            continue
        center = city_centers.get(city)
        if center:
            candidate["computed_distance_to_city_center_km"] = round(_haversine(*coords, *center), 3)


# --- Overpass amenity enrichment ---

def _enrich_overpass(candidates: list[dict[str, Any]], soft_facts: dict[str, Any]) -> None:
    needed = {
        name for name, cfg in _AMENITY_CONFIG.items()
        if any(soft_facts.get(t) for t in cfg.triggers)
    }
    if not needed:
        return

    candidate_coords = [
        (c, coords) for c in candidates if (coords := _candidate_coords(c))
    ]
    if not candidate_coords:
        return

    bbox = _bounding_box([coords for _, coords in candidate_coords])

    for amenity_name in needed:
        query = _build_overpass_query(_AMENITY_CONFIG[amenity_name].osm_tags, bbox)
        locations = _fetch_overpass(query)
        if not locations:
            continue
        for candidate, coords in candidate_coords:
            min_dist = min(_haversine(*coords, alat, alon) for alat, alon in locations)
            candidate[f"computed_distance_to_{amenity_name}_km"] = round(min_dist, 3)


def _build_overpass_query(osm_tags: list[tuple[str, str]], bbox: str) -> str:
    statements = "\n".join(f"  {el_type}[{tag}]({bbox});" for el_type, tag in osm_tags)
    return f"[out:json][timeout:20];\n(\n{statements}\n);\nout center;"


def _fetch_overpass(query: str) -> list[tuple[float, float]]:
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(_OVERPASS_URL, data={"data": query}, headers=_HEADERS)
        locations: list[tuple[float, float]] = []
        for el in resp.json().get("elements", []):
            if "center" in el:
                locations.append((el["center"]["lat"], el["center"]["lon"]))
            elif "lat" in el and "lon" in el:
                locations.append((el["lat"], el["lon"]))
        return locations
    except Exception:
        return []


# --- Nominatim geocoding ---

@lru_cache(maxsize=256)
def _geocode(place: str) -> tuple[float, float] | None:
    query = place if "switzerland" in place.lower() else f"{place} Switzerland"
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(
                _NOMINATIM_URL,
                params={"q": query, "format": "json", "limit": 1, "countrycodes": "ch"},
                headers=_HEADERS,
            )
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        pass
    return None


# --- Geometry ---

def _bounding_box(coords: list[tuple[float, float]], padding: float = 0.05) -> str:
    lats, lons = zip(*coords)
    return f"{min(lats) - padding},{min(lons) - padding},{max(lats) + padding},{max(lons) + padding}"


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    d_lat, d_lon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

"""NLU extraction: natural language query + user profile -> ExtractedQuery.

Uses Claude Haiku 4.5 with a strict JSON schema. The prompt is carefully
scoped so the model:
  - treats the user's query as the source of truth
  - only pulls from the profile when the query is silent on a dimension
  - separates hard from soft, and visual-VLM from semantic-embedding soft signals
  - reports which profile fields it used (so the UI can show "using your default budget")
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import anthropic

from .config import settings
from .schemas import ExtractedQuery, UserProfile

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a query-understanding module for a Swiss real-estate search engine.

Given a user's natural-language apartment query (English, German, French, or mixed) and optionally a User Profile, you must output a single JSON object matching the ExtractedQuery schema below. No prose, no markdown fences — JSON only.

# Schema
```
{
  "hard_filters": {
    "cities": [str],             // e.g. ["Zürich"]. OR-matched.
    "districts": [str],          // e.g. ["Kreis 4", "Oerlikon"]
    "cantons": [str],            // e.g. ["ZH"]
    "price_chf_max": int | null,
    "price_chf_min": int | null,
    "rooms_min": float | null,   // Swiss half-rooms OK: 2.5, 3.5...
    "rooms_max": float | null,
    "area_sqm_min": int | null,
    "area_sqm_max": int | null,
    "required_features": [str],  // ONLY if user is emphatic ("must have", "ist wichtig")
    "property_type": "apartment" | "studio" | "house" | null,
    "furnished": bool | null,
    "available_from": "YYYY-MM-DD" | null
  },
  "soft_structured": {
    "preferred_features": [str], // balcony, elevator, washing_machine, parking, garden, terrace, cellar, dishwasher, fireplace, pet_friendly
    "avoid_features": [str]
  },
  "soft_semantic": {
    // Weights 0.0-1.0. 0 = not mentioned, 1 = user strongly emphasized it.
    "brightness": float,       // "bright", "hell", "lots of light", "large windows"
    "modernity": float,        // "modern", "contemporary", "new"
    "condition": float,        // "well-maintained", "renovated", "gepflegt"
    "spaciousness": float,     // "spacious", "open", "großzügig", "good layout"
    "kitchen_appeal": float,   // "nice kitchen", "modern kitchen"
    "bathroom_appeal": float,  // "nice bathroom", "modern bathroom"
    "quietness": float,        // "quiet", "ruhig", "not on a busy street"
    "safety": float,           // "safe area", "sicher"
    "family_friendly": float,  // "good for kids", "family-friendly", "playgrounds"
    "near_lake_or_green": float, // "near the lake", "greenery", "park"
    "free_text": str           // Anything else that didn't fit: "cozy bohemian vibe", "not too anonymous"
  },
  "commute": {
    "destination": str,        // verbatim landmark or address
    "max_minutes": int,
    "mode": "public_transport" | "walking" | "cycling" | "driving"
  } | null,
  "relaxation_priority": [str], // ORDER matters: first = drop first.
  "clarifications_needed": [str] // Only if REALLY underspecified. Prefer empty.
}
```

# Rules
1. Hard vs soft: "must have", "unbedingt", "mindestens", "maximal", "bis" → hard. "gern", "ideally", "if possible", "schön wäre" → soft.
2. Feature vocabulary (canonical): balcony, terrace, garden, elevator, parking, garage, washing_machine, dishwasher, cellar, fireplace, furnished, pet_friendly, modern_kitchen, two_bathrooms. Map user words to these.
3. District vs city: "Kreis 4", "Oerlikon", "Altstetten" are DISTRICTS of Zürich. If user says "Oerlikon", set cities=["Zürich"] AND districts=["Oerlikon"].
4. Semantic weights: be conservative. Only non-zero when the query actually mentions it. Values:
   - 0.3: implied/background ("angenehm", "pleasant")
   - 0.6: explicit single mention ("bright")
   - 0.9: emphasized ("lots of light", "sehr hell", repeated)
5. relaxation_priority: list hard/soft keys in the order you'd drop them if results are sparse. Heuristic:
   - First: soft_semantic weights, avoid_features
   - Then: preferred_features (one by one)
   - Then: area_sqm_min, rooms range widening
   - Last: price_chf_max, cities (NEVER drop city without asking)
6. Profile usage: if the query is missing a city/budget/rooms AND the profile has defaults, use the profile values and include the field name in the implicit `profile_fields_used` (you don't need to output this; the orchestrator tracks it — just use the values).
7. Commute: "near ETH" → commute{destination:"ETH Zurich", max_minutes:20, mode:"public_transport"} unless user specifies otherwise. "max 25 min zum HB" → commute{destination:"Zurich HB", max_minutes:25, mode:"public_transport"}.
8. Budget-only floors: "günstiger", "affordable", "cheap" without a number: don't invent a number. Put weight in relaxation_priority with a low price preference signal via soft_semantic.free_text = "looking for an affordable option".
9. Output ONLY the JSON. No commentary.

# Examples

Query: "1.5-Zimmer Wohnung in Zürich nahe ETH, unter 2200 CHF"
Output:
{"hard_filters":{"cities":["Zürich"],"districts":[],"cantons":[],"price_chf_max":2200,"price_chf_min":null,"rooms_min":1.5,"rooms_max":1.5,"area_sqm_min":null,"area_sqm_max":null,"required_features":[],"property_type":"apartment","furnished":null,"available_from":null},"soft_structured":{"preferred_features":[],"avoid_features":[]},"soft_semantic":{"brightness":0,"modernity":0,"condition":0,"spaciousness":0,"kitchen_appeal":0,"bathroom_appeal":0,"quietness":0,"safety":0,"family_friendly":0,"near_lake_or_green":0,"free_text":""},"commute":{"destination":"ETH Zurich","max_minutes":20,"mode":"public_transport"},"relaxation_priority":["commute","rooms_max","price_chf_max"],"clarifications_needed":[]}

Query: "Something quiet and bright in Zurich"
Output:
{"hard_filters":{"cities":["Zürich"],"districts":[],"cantons":[],"price_chf_max":null,"price_chf_min":null,"rooms_min":null,"rooms_max":null,"area_sqm_min":null,"area_sqm_max":null,"required_features":[],"property_type":null,"furnished":null,"available_from":null},"soft_structured":{"preferred_features":[],"avoid_features":[]},"soft_semantic":{"brightness":0.7,"modernity":0,"condition":0,"spaciousness":0,"kitchen_appeal":0,"bathroom_appeal":0,"quietness":0.7,"safety":0,"family_friendly":0,"near_lake_or_green":0,"free_text":""},"commute":null,"relaxation_priority":["brightness","quietness"],"clarifications_needed":["What is your budget?","How many rooms do you need?"]}
"""


def _build_user_block(query: str, profile: Optional[UserProfile]) -> str:
    parts = [f"<query>\n{query.strip()}\n</query>"]
    if profile is not None:
        # Only include fields likely useful as defaults.
        profile_snippet = {
            "home_cities": profile.home_cities,
            "work_address": profile.work_address,
            "typical_budget_max_chf": profile.typical_budget_max_chf,
            "typical_rooms_min": profile.typical_rooms_min,
            "household_size": profile.household_size,
            "has_children": profile.has_children,
            "preferred_features": profile.preferred_features,
            "semantic_preferences": profile.semantic_preferences,
        }
        # Drop None/empty entries so we don't bias the model with noise.
        clean = {k: v for k, v in profile_snippet.items() if v not in (None, [], {}, "")}
        if clean:
            parts.append(
                "<user_profile>\n"
                + json.dumps(clean, ensure_ascii=False)
                + "\n</user_profile>\n"
                + "Use profile values ONLY for fields the query leaves unspecified."
            )
    return "\n\n".join(parts)


class QueryExtractor:
    def __init__(self, client: Optional[anthropic.Anthropic] = None):
        self.client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def extract(
        self,
        query: str,
        profile: Optional[UserProfile] = None,
    ) -> ExtractedQuery:
        user_block = _build_user_block(query, profile)

        resp = self.client.messages.create(
            model=settings.extractor_model,
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_block}],
        )

        # Haiku is reliable about JSON-only output when told explicitly, but
        # we still defensively strip code fences in case it wraps them.
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].lstrip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            log.exception("Extractor returned invalid JSON: %s", raw[:500])
            # Safe fallback: empty extraction, let pipeline handle from scratch
            return ExtractedQuery()

        try:
            extracted = ExtractedQuery.model_validate(parsed)
        except Exception:
            log.exception("Extractor JSON did not match schema: %s", parsed)
            return ExtractedQuery()

        # Record which profile fields we leaned on. We compare the extracted
        # hard_filters to the original query signal; if the query doesn't
        # mention a dimension but the output has it, we assume it came from profile.
        if profile is not None:
            extracted.profile_fields_used = _infer_profile_fields_used(query, extracted, profile)

        return extracted


def _infer_profile_fields_used(
    query: str, extracted: ExtractedQuery, profile: UserProfile
) -> list[str]:
    """Crude but useful: flag fields that match the profile and aren't in the query text."""
    used = []
    q = query.lower()

    if profile.typical_budget_max_chf and extracted.hard_filters.price_chf_max == profile.typical_budget_max_chf:
        if not any(tok in q for tok in ["chf", "franc", "€", "budget", "miete", "rent"]):
            used.append("typical_budget_max_chf")

    if profile.home_cities and extracted.hard_filters.cities:
        if set(extracted.hard_filters.cities) <= set(profile.home_cities):
            query_mentions_city = any(c.lower() in q for c in extracted.hard_filters.cities)
            if not query_mentions_city:
                used.append("home_cities")

    if profile.typical_rooms_min and extracted.hard_filters.rooms_min == profile.typical_rooms_min:
        if "zimmer" not in q and "room" not in q and "bedroom" not in q:
            used.append("typical_rooms_min")

    return used

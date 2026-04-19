"""Generate short natural-language explanations for top-K results.

Two modes:
  - cheap: deterministic template strings from the ranker's reasons + hard filters
  - rich:  a single Sonnet call that summarizes the whole result set as a
           friendly paragraph + one bullet per listing

Which mode we pick is controlled by the orchestrator. Cheap is default for
latency-sensitive calls; rich is used when the user is reading a final answer.
"""
from __future__ import annotations

import json
from typing import Any

import anthropic

from .config import settings
from .ranker import RankedListing
from .relaxation import RelaxationRound
from .schemas import ExtractedQuery


def cheap_explanations(
    ranked: list[RankedListing],
    extracted: ExtractedQuery,
) -> list[str]:
    out = []
    for r in ranked:
        parts = []
        lst = r.listing
        if lst.get("price_chf") and extracted.hard_filters.price_chf_max:
            parts.append(
                f"{lst['price_chf']} CHF (≤ {extracted.hard_filters.price_chf_max})"
            )
        if lst.get("rooms"):
            parts.append(f"{lst['rooms']} rooms")
        if lst.get("area_sqm"):
            parts.append(f"{lst['area_sqm']} m²")
        if r.boost_reasons:
            parts.append("boosted by: " + ", ".join(r.boost_reasons[:3]))
        out.append("; ".join(parts))
    return out


RICH_SYSTEM = """You write short, honest summaries for a real-estate search result.

Given the user's query, the extracted intent, any relaxations applied, and
the top listings (with their ranking reasons), produce JSON:

{
  "headline": str,      // <= 20 words, tells the user what we returned and why
  "caveats":  [str],    // 0-3 short notes about trade-offs / relaxations
  "per_listing": [      // same order as input
    {"listing_id": str, "why": str}  // <= 25 words each, concrete and specific
  ]
}

Rules:
  - Be concrete: cite price, rooms, area, features, VLM boosts. Do NOT invent facts.
  - If relaxations were applied, lead with that in the caveats.
  - Neutral tone. No sales language.
  - Same language as the user's query (German, English, etc.).
"""


def rich_explanations(
    original_query: str,
    extracted: ExtractedQuery,
    ranked: list[RankedListing],
    relaxations: list[RelaxationRound],
    client: anthropic.Anthropic | None = None,
) -> dict[str, Any]:
    client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)

    listings_for_prompt = []
    for r in ranked:
        lst = r.listing
        listings_for_prompt.append({
            "listing_id": lst.get("listing_id"),
            "city": lst.get("city"),
            "price_chf": lst.get("price_chf"),
            "rooms": lst.get("rooms"),
            "area_sqm": lst.get("area_sqm"),
            "features": lst.get("features"),
            "title": lst.get("title"),
            "ranking_reasons": r.boost_reasons,
            "final_score": round(r.final_score, 3),
        })

    payload = {
        "query": original_query,
        "extracted": extracted.model_dump(),
        "relaxations": [rr.__dict__ for rr in relaxations],
        "top_listings": listings_for_prompt,
    }

    resp = client.messages.create(
        model=settings.explainer_model,
        max_tokens=800,
        system=RICH_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()
    try:
        return json.loads(raw)
    except Exception:
        return {"headline": "", "caveats": [], "per_listing": []}

"""Re-rank the backend's candidate list using VLM scores and profile signals.

The backend already returns a `score`. We treat that as the **base relevance**
(text/BM25/embedding) and add our own **visual boost** on top using the
pre-computed variable_vlm_* fields.

Final score =
    base_score * w_base
  + visual_boost * w_visual
  + profile_boost * w_profile
  - avoid_penalty

Weights default to (0.6, 0.3, 0.1) — tweak via env vars.

Notes:
  - VLM scores on listings are 0-5. We normalize to 0-1.
  - Semantic weights from the extractor are 0-1. They scale which VLM dimensions matter.
  - If a listing is missing VLM data (variable_vlm_success == False), we fall back
    to base_score only (no penalty).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from .schemas import ExtractedQuery, UserProfile


W_BASE = float(os.getenv("W_BASE", "0.6"))
W_VISUAL = float(os.getenv("W_VISUAL", "0.3"))
W_PROFILE = float(os.getenv("W_PROFILE", "0.1"))


# Map semantic-weight field name -> VLM score field name on the listing.
# Listing fields come back with the `variable_vlm_` prefix from the DB,
# but the API response may normalize them. Handle both.
VLM_MAP = {
    "brightness": "brightness_score",
    "modernity": "modernity_score",
    "condition": "condition_score",
    "spaciousness": "spaciousness_score",
    "kitchen_appeal": "kitchen_appeal_score",
    "bathroom_appeal": "bathroom_appeal_score",
}


def _get_vlm(listing: dict[str, Any], short_key: str) -> Optional[float]:
    """Look up a VLM score, tolerant of prefix naming."""
    for key in (short_key, f"variable_vlm_{short_key}", f"vlm_{short_key}"):
        if key in listing and listing[key] is not None:
            try:
                v = float(listing[key])
                return max(0.0, min(1.0, v / 5.0))  # normalize 0-5 -> 0-1
            except (TypeError, ValueError):
                continue
    return None


@dataclass
class RankedListing:
    listing: dict[str, Any]
    base_score: float
    visual_boost: float
    profile_boost: float
    avoid_penalty: float
    final_score: float
    boost_reasons: list[str]  # e.g. ["bright (VLM 0.8)", "has balcony"]

    def to_payload(self) -> dict[str, Any]:
        out = dict(self.listing)
        out["_rank"] = {
            "base_score": round(self.base_score, 4),
            "visual_boost": round(self.visual_boost, 4),
            "profile_boost": round(self.profile_boost, 4),
            "avoid_penalty": round(self.avoid_penalty, 4),
            "final_score": round(self.final_score, 4),
            "reasons": self.boost_reasons,
        }
        return out


def _visual_boost(listing: dict[str, Any], extracted: ExtractedQuery) -> tuple[float, list[str]]:
    sem = extracted.soft_semantic
    reasons: list[str] = []

    # Compute weighted sum of (user_weight * normalized_vlm_score) over dims.
    total_weight = 0.0
    total_score = 0.0
    for sem_key, vlm_key in VLM_MAP.items():
        w = getattr(sem, sem_key)
        if w <= 0.0:
            continue
        vlm = _get_vlm(listing, vlm_key)
        if vlm is None:
            continue
        total_weight += w
        total_score += w * vlm
        if vlm >= 0.7:
            reasons.append(f"{sem_key} ({vlm:.2f})")
    if total_weight == 0.0:
        return 0.0, reasons
    return total_score / total_weight, reasons


def _profile_boost(
    listing: dict[str, Any], profile: Optional[UserProfile]
) -> tuple[float, list[str]]:
    """Boost listings that match durable user preferences."""
    if profile is None:
        return 0.0, []

    reasons: list[str] = []
    score = 0.0
    hits = 0

    # Feature overlap
    listing_features = set(listing.get("features") or [])
    pref_features = set(profile.preferred_features or [])
    overlap = listing_features & pref_features
    if overlap:
        score += min(1.0, len(overlap) * 0.3)
        reasons.append(f"matches preferred: {', '.join(sorted(overlap))}")
        hits += 1

    # Semantic prefs: profile keeps moving averages of user-liked dims
    for sem_key, vlm_key in VLM_MAP.items():
        w = (profile.semantic_preferences or {}).get(sem_key, 0.0)
        if w <= 0.0:
            continue
        vlm = _get_vlm(listing, vlm_key)
        if vlm is None:
            continue
        score += w * vlm
        hits += 1

    if hits == 0:
        return 0.0, []
    # Normalize so profile boost in [0,1]
    return min(1.0, score / max(1, hits)), reasons


def _avoid_penalty(listing: dict[str, Any], extracted: ExtractedQuery) -> float:
    avoid = set(extracted.soft_structured.avoid_features or [])
    if not avoid:
        return 0.0
    listing_features = set(listing.get("features") or [])
    hits = avoid & listing_features
    return min(1.0, 0.5 * len(hits))


_SEEN_PENALTY = 0.3  # reduce final_score by this much for already-shown listings


def rerank(
    api_results: list[dict[str, Any]],
    extracted: ExtractedQuery,
    profile: Optional[UserProfile] = None,
    top_k: int = 20,
    seen_ids: Optional[set] = None,
) -> list[RankedListing]:
    all_seen = set(seen_ids or set())
    if profile:
        all_seen.update(profile.seen_listing_ids or [])

    ranked: list[RankedListing] = []
    for item in api_results:
        base = float(item.get("score", 0.0))
        vboost, vreasons = _visual_boost(item, extracted)
        pboost, preasons = _profile_boost(item, profile)
        pen = _avoid_penalty(item, extracted)
        lid = str(item.get("listing_id", ""))
        if lid and lid in all_seen:
            pen += _SEEN_PENALTY
        final = W_BASE * base + W_VISUAL * vboost + W_PROFILE * pboost - pen
        ranked.append(
            RankedListing(
                listing=item,
                base_score=base,
                visual_boost=vboost,
                profile_boost=pboost,
                avoid_penalty=pen,
                final_score=final,
                boost_reasons=vreasons + preasons,
            )
        )
    ranked.sort(key=lambda r: r.final_score, reverse=True)
    return ranked[:top_k]

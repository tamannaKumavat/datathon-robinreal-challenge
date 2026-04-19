"""Smart relaxation.

The backend's /pipeline already auto-relaxes when results drop below ~10 and
returns `relaxations_applied`. Our job on top:

  1. Track what was relaxed and surface it to the user clearly.
  2. If results are STILL too few after the backend's relaxation, drop our own
     soft constraints in the order given by `extracted.relaxation_priority`
     and re-query.
  3. Never relax hard filters the user was emphatic about (cities, must-have features).

We cap the number of retry rounds at 3 to bound latency and API spend.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Optional

from .pipeline_client import PipelineClient
from .schemas import ExtractedQuery

log = logging.getLogger(__name__)

MIN_ACCEPTABLE_RESULTS = 5
MAX_ROUNDS = 3


@dataclass
class RelaxationRound:
    round_idx: int
    dropped: list[str]
    total_candidates: int


def _drop_constraint(extracted: ExtractedQuery, key: str) -> bool:
    """Mutate extracted to drop the named constraint. Return True if anything changed."""
    hf = extracted.hard_filters
    sem = extracted.soft_semantic
    soft = extracted.soft_structured

    if key in VLM_SEMANTIC_KEYS:
        if getattr(sem, key) > 0:
            setattr(sem, key, 0.0)
            return True
        return False

    if key == "preferred_features":
        if soft.preferred_features:
            soft.preferred_features = []
            return True
        return False

    if key == "area_sqm_min":
        if hf.area_sqm_min is not None:
            # Widen by 10 sqm per round rather than dropping entirely
            hf.area_sqm_min = max(0, hf.area_sqm_min - 10)
            return True
        return False

    if key == "rooms_min":
        if hf.rooms_min is not None:
            hf.rooms_min = max(1.0, hf.rooms_min - 0.5)
            return True
        return False

    if key == "rooms_max":
        if hf.rooms_max is not None:
            hf.rooms_max += 0.5
            return True
        return False

    if key == "price_chf_max":
        if hf.price_chf_max is not None:
            hf.price_chf_max = int(hf.price_chf_max * 1.10)  # +10%
            return True
        return False

    if key == "commute":
        if extracted.commute is not None:
            extracted.commute.max_minutes += 10
            return True
        return False

    if key == "districts":
        if hf.districts:
            hf.districts = []
            return True
        return False

    # Never drop cities or required_features automatically.
    return False


VLM_SEMANTIC_KEYS = {
    "brightness", "modernity", "condition", "spaciousness",
    "kitchen_appeal", "bathroom_appeal", "quietness", "safety",
    "family_friendly", "near_lake_or_green",
}


def relax_and_retry(
    client: PipelineClient,
    extracted: ExtractedQuery,
    original_query: str,
    initial_response: dict[str, Any],
    top_k: int,
) -> tuple[dict[str, Any], list[RelaxationRound]]:
    """Return (final_response, list of our own relaxation rounds).

    Only kicks in if the initial response has fewer than MIN_ACCEPTABLE_RESULTS.
    """
    rounds: list[RelaxationRound] = []
    response = initial_response

    if len(response.get("results", [])) >= MIN_ACCEPTABLE_RESULTS:
        return response, rounds

    working_extracted = copy.deepcopy(extracted)

    for round_idx in range(1, MAX_ROUNDS + 1):
        dropped_this_round: list[str] = []
        for key in list(working_extracted.relaxation_priority):
            if _drop_constraint(working_extracted, key):
                dropped_this_round.append(key)
                break  # drop ONE constraint per round, then re-query
        if not dropped_this_round:
            break  # nothing left we're willing to drop

        try:
            response = client.search(working_extracted, original_query, top_k=top_k)
        except Exception as e:
            log.warning("Relaxation retry round %d failed: %s", round_idx, e)
            break

        rounds.append(
            RelaxationRound(
                round_idx=round_idx,
                dropped=dropped_this_round,
                total_candidates=response.get("total_candidates", 0),
            )
        )

        if len(response.get("results", [])) >= MIN_ACCEPTABLE_RESULTS:
            break

    return response, rounds

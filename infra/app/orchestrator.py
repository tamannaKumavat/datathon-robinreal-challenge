"""End-to-end orchestrator.

User query + optional user_id
  -> load profile
  -> extract intent
  -> call retrieval API (with embedding endpoint when soft signals exist)
  -> relax if too few results
  -> re-rank with VLM + profile boosts
  -> explain top results
  -> (optionally) rewrite the profile
  -> return a complete SearchResponse
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Optional

from .config import settings
from .explain import cheap_explanations, rich_explanations
from .extractor import QueryExtractor
from .pipeline_client import PipelineClient
from .profile_manager import (
    ClaudeProfileRewriter,
    InMemoryProfileStore,
    ProfileStore,
)
from .ranker import RankedListing, rerank
from .relaxation import RelaxationRound, relax_and_retry
from .schemas import ExtractedQuery, UserProfile

log = logging.getLogger(__name__)


class SearchOrchestrator:
    def __init__(
        self,
        profile_store: Optional[ProfileStore] = None,
        pipeline_client: Optional[PipelineClient] = None,
    ):
        self.profile_store = profile_store or InMemoryProfileStore()
        self.pipeline = pipeline_client or PipelineClient()
        self.extractor = QueryExtractor()
        self.rewriter = ClaudeProfileRewriter()

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = None,
        rich: bool = False,
        update_profile: bool = False,
        favourited_ids: Optional[list[str]] = None,
        hidden_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        top_k = top_k or settings.default_top_k

        profile: Optional[UserProfile] = None
        if user_id:
            profile = self.profile_store.get(user_id) or UserProfile(user_id=user_id)

        # 1. Extract intent
        extracted: ExtractedQuery = self.extractor.extract(query, profile)

        # 2. Retrieve from backend (pool size larger than top_k for reranking headroom)
        api_resp = self.pipeline.search(
            extracted,
            original_query=query,
            top_k=settings.rerank_pool_size,
        )

        # 3. Relax if needed
        api_resp, our_relaxations = relax_and_retry(
            self.pipeline,
            extracted,
            query,
            api_resp,
            top_k=settings.rerank_pool_size,
        )

        # 4. Rerank (passing seen IDs for downranking)
        ranked: list[RankedListing] = rerank(
            api_resp.get("results", []),
            extracted,
            profile=profile,
            top_k=top_k,
        )

        # Track seen listings in profile
        if user_id and profile is not None:
            shown_ids = [r.listing.get("listing_id") for r in ranked if r.listing.get("listing_id")]
            seen = list(dict.fromkeys((profile.seen_listing_ids or []) + [str(i) for i in shown_ids]))
            profile.seen_listing_ids = seen[-200:]  # rolling window of 200
            self.profile_store.put(profile)

        # 5. Explanations
        explanations: dict[str, Any]
        if rich:
            explanations = rich_explanations(query, extracted, ranked, our_relaxations)
        else:
            explanations = {
                "headline": "",
                "caveats": _format_caveats(api_resp, our_relaxations),
                "per_listing": [
                    {"listing_id": r.listing.get("listing_id"), "why": why}
                    for r, why in zip(ranked, cheap_explanations(ranked, extracted))
                ],
            }

        # 6. Profile rewrite
        new_profile_snapshot = None
        if update_profile and user_id:
            base = profile or UserProfile(user_id=user_id)
            updated = self.rewriter.rewrite(
                base, query, extracted,
                favourited_ids=favourited_ids,
                hidden_ids=hidden_ids,
            )
            self.profile_store.put(updated)
            new_profile_snapshot = updated.model_dump()

        return {
            "query": query,
            "extracted": extracted.model_dump(),
            "total_candidates": api_resp.get("total_candidates"),
            "backend_relaxations_applied": api_resp.get("relaxations_applied"),
            "our_relaxations": [asdict(r) for r in our_relaxations],
            "results": [r.to_payload() for r in ranked],
            "explanations": explanations,
            "profile_fields_used": extracted.profile_fields_used,
            "profile_updated": new_profile_snapshot is not None,
            "profile_snapshot": new_profile_snapshot,
        }


def _format_caveats(
    api_resp: dict[str, Any],
    our_relaxations: list[RelaxationRound],
) -> list[str]:
    out = []
    backend = api_resp.get("relaxations_applied")
    if backend:
        out.append(f"Backend auto-relaxed: {', '.join(backend)}")
    for r in our_relaxations:
        out.append(
            f"Round {r.round_idx}: dropped {', '.join(r.dropped)} "
            f"→ {r.total_candidates} candidates"
        )
    return out

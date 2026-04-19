"""Thin client over the teammate's retrieval API.

Two endpoints:
  POST /pipeline       -> hard-filter + BM25 + simple soft
  POST /pipeline_embed -> hard-filter + embedding-based soft rerank

We use /pipeline_embed when soft_semantic signals are non-trivial, and
/pipeline otherwise. We also build an *enriched query string* from the
ExtractedQuery so the backend's own parser has the cleanest possible input.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from .config import settings
from .schemas import ExtractedQuery

log = logging.getLogger(__name__)


def build_enriched_query(extracted: ExtractedQuery, original: str) -> str:
    """Compose a query string the backend can reliably parse.

    Rationale: the teammate's API does its own NLU internally. By feeding it
    a canonical, high-signal sentence, we reduce double-extraction drift.
    We keep the original query as the trunk and *append* normalized filters
    so nothing the user said gets lost in translation.
    """
    parts = [original.strip()]
    hf = extracted.hard_filters

    if hf.cities:
        parts.append("in " + " or ".join(hf.cities))
    if hf.districts:
        parts.append("districts: " + ", ".join(hf.districts))
    if hf.price_chf_max is not None:
        parts.append(f"max {hf.price_chf_max} CHF")
    if hf.price_chf_min is not None:
        parts.append(f"min {hf.price_chf_min} CHF")
    if hf.rooms_min is not None and hf.rooms_max is not None and hf.rooms_min == hf.rooms_max:
        parts.append(f"{hf.rooms_min} rooms")
    else:
        if hf.rooms_min is not None:
            parts.append(f"at least {hf.rooms_min} rooms")
        if hf.rooms_max is not None:
            parts.append(f"at most {hf.rooms_max} rooms")
    if hf.area_sqm_min is not None:
        parts.append(f"from {hf.area_sqm_min} sqm")
    if hf.area_sqm_max is not None:
        parts.append(f"up to {hf.area_sqm_max} sqm")
    if hf.required_features:
        parts.append("must have: " + ", ".join(hf.required_features))
    if hf.furnished is True:
        parts.append("furnished")
    if hf.available_from:
        parts.append(f"available from {hf.available_from}")

    if extracted.soft_structured.preferred_features:
        parts.append("prefer: " + ", ".join(extracted.soft_structured.preferred_features))

    # Append non-trivial semantic signals as plain English — the backend's
    # embedding model will pick them up.
    sem = extracted.soft_semantic
    sem_tokens = []
    for name, threshold in [
        ("bright", sem.brightness),
        ("modern", sem.modernity),
        ("well-maintained", sem.condition),
        ("spacious", sem.spaciousness),
        ("modern kitchen", sem.kitchen_appeal),
        ("nice bathroom", sem.bathroom_appeal),
        ("quiet", sem.quietness),
        ("safe", sem.safety),
        ("family-friendly", sem.family_friendly),
        ("near lake or greenery", sem.near_lake_or_green),
    ]:
        if threshold >= 0.5:
            sem_tokens.append(name)
    if sem_tokens:
        parts.append(", ".join(sem_tokens))
    if sem.free_text:
        parts.append(sem.free_text)

    if extracted.commute:
        c = extracted.commute
        parts.append(f"max {c.max_minutes} min by {c.mode.replace('_', ' ')} to {c.destination}")

    return ". ".join(p for p in parts if p)


class PipelineClient:
    def __init__(self, base_url: str = None, timeout: float = 30.0):
        self.base_url = (base_url or settings.pipeline_url).rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _call(self, endpoint: str, query: str, top_k: int) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        resp = self._client.post(
            url,
            json={"query": query},
            params={"top_k": top_k},
        )
        resp.raise_for_status()
        return resp.json()

    def search(
        self,
        extracted: ExtractedQuery,
        original_query: str,
        top_k: int = None,
        prefer_embed: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Return the raw API response.

        prefer_embed: if None, auto-decide based on whether there are
        non-trivial semantic signals. If True/False, force.
        """
        top_k = top_k or settings.rerank_pool_size
        enriched = build_enriched_query(extracted, original_query)

        if prefer_embed is None:
            sem = extracted.soft_semantic
            has_strong_soft = (
                bool(sem.free_text)
                or max(
                    sem.brightness, sem.modernity, sem.condition, sem.spaciousness,
                    sem.kitchen_appeal, sem.bathroom_appeal, sem.quietness,
                    sem.safety, sem.family_friendly, sem.near_lake_or_green,
                ) >= 0.5
            )
            endpoint = "pipeline_embed" if has_strong_soft else "pipeline"
        else:
            endpoint = "pipeline_embed" if prefer_embed else "pipeline"

        log.info("Calling %s with query=%r top_k=%d", endpoint, enriched, top_k)
        return self._call(endpoint, enriched, top_k)

    def health(self) -> dict[str, Any]:
        resp = self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

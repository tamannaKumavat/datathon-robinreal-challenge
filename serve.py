"""
FastAPI hybrid retrieval server.

  uvicorn serve:app --host 0.0.0.0 --port 8000

Endpoints:
  GET /health
  GET /search?q=<query>&top=10&mode=hybrid_ws&alpha=0.7&city=&rooms_min=&rooms_max=&area_min=&price_max=
  GET /bm25?q=<query>&top=10&mode=hybrid&variant=bm25s
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel

import search as S
import bm25_retrieval as B

sys.path.insert(0, str(Path(__file__).parent))
from app.config import get_settings
from app.participant.hard_fact_extraction import extract_hard_facts
from app.participant.soft_fact_extraction import extract_soft_facts
from app.participant.soft_filtering import filter_soft_facts
from app.participant.ranking import rank_listings
from app.harness.search_service import to_hard_filter_params, search_with_relaxation, _resolve_near_place

warnings.filterwarnings("ignore")

app = FastAPI(title="Swiss RE Retrieval (BGE-M3 hybrid + BM25)")

# Load once at startup
_bge, _ort_tok, _ort_sess, _ids, _faiss, _inv, _texts = S.load_indexes()

print("Building BM25 indexes...")
_bm25 = B.MultiLingualRetriever(mode="hybrid", variant="bm25s")
_bm25.load_and_build()
print("BM25 ready.")

print("Loading SigLIP text tower (ONNX)...")
from app.participant.ranking import _load_siglip, _load_corpus, _load_vlm
_load_siglip()
_load_corpus()
_load_vlm()
print("SigLIP + corpus + VLM ready.")

_VLM_SCORE_COLS = [
    "brightness_score", "modernity_score", "condition_score",
    "spaciousness_score", "kitchen_appeal_score", "bathroom_appeal_score",
]

def _attach_vlm(result_dict: dict) -> dict:
    """Attach pre-computed VLM score fields so downstream rankers can use them."""
    vlm = _load_vlm()
    feats = vlm.get(str(result_dict.get("listing_id"))) or {}
    for col in _VLM_SCORE_COLS:
        result_dict[col] = feats.get(col)
    return result_dict


class Hit(BaseModel):
    rank: int
    id: str
    score: float
    city: Optional[str] = None
    rooms: Optional[float] = None
    area: Optional[float] = None
    price: Optional[int] = None
    text: str


class SearchResponse(BaseModel):
    query: str
    mode: str
    top: int
    filters: dict
    hits: list[Hit]


class BM25Hit(BaseModel):
    rank: int
    id: str
    score: float
    per_lang: dict
    text: str


class BM25Response(BaseModel):
    query: str
    mode: str
    variant: str
    top: int
    hits: list[BM25Hit]


_settings = get_settings()


@app.get("/health")
def health():
    return {"status": "ok", "docs": len(_ids), "bm25_docs": _bm25.n}


@app.post("/listings")
async def listings(body: dict):
    """MCP-compatible endpoint: POST /listings → {listings: [{listing, score, reason}]}"""
    query = body.get("query", "")
    limit = body.get("limit", 25)
    offset = body.get("offset", 0)
    hard_facts = extract_hard_facts(query)
    hard_facts.limit = 5000
    hard_facts.offset = 0
    hard_facts = _resolve_near_place(hard_facts)
    soft_facts = extract_soft_facts(query)
    if hard_facts.neighborhood:
        soft_facts["neighborhoods"] = hard_facts.neighborhood
    qv, qw = S.encode_query(_bge, _ort_tok, _ort_sess, query)
    soft_facts["_query_dense"] = qv
    soft_facts["_query_sparse"] = qw
    soft_facts["_query"] = query
    bm25_hits = _bm25.search(query, top_k=10)
    soft_facts["_bm25_top"] = {h["id"]: rank + 1 for rank, h in enumerate(bm25_hits)}
    candidates, _ = search_with_relaxation(_settings.db_path, to_hard_filter_params(hard_facts))
    candidates = filter_soft_facts(candidates, soft_facts)
    ranked = rank_listings(candidates, soft_facts)

    page = ranked[offset: offset + limit]
    return {
        "listings": [
            {
                "listing_id": r.listing_id,
                "score": r.score,
                "reason": r.reason,
                "listing": {
                    "title": r.listing.title,
                    "city": r.listing.city,
                    "canton": r.listing.canton,
                    "price_chf": r.listing.price_chf,
                    "rooms": r.listing.rooms,
                    "living_area_sqm": r.listing.living_area_sqm,
                    "features": r.listing.features,
                    "street": r.listing.street,
                    "available_from": r.listing.available_from,
                    "latitude": r.listing.latitude,
                    "longitude": r.listing.longitude,
                    "original_url": r.listing.original_listing_url,
                    "hero_image_url": r.listing.hero_image_url,
                },
            }
            for r in page
        ],
        "meta": {"total": len(ranked), "offset": offset, "limit": limit, "query": query},
    }


class PipelineResponse(BaseModel):
    query: str
    total_candidates: int
    total_results: int
    relaxations_applied: list[str] | None
    results: list[dict[str, Any]]


@app.post("/pipeline", response_model=PipelineResponse)
def pipeline(body: dict, top_k: int = Query(30, ge=1, le=500), min_results: int = Query(10, ge=0, le=500)):
    query = body.get("query", "")
    hard_facts = extract_hard_facts(query)
    hard_facts.limit = 5000
    hard_facts.offset = 0
    hard_facts = _resolve_near_place(hard_facts)
    soft_facts = extract_soft_facts(query)
    if hard_facts.neighborhood:
        soft_facts["neighborhoods"] = hard_facts.neighborhood
    qv, qw = S.encode_query(_bge, _ort_tok, _ort_sess, query)
    soft_facts["_query_dense"] = qv
    soft_facts["_query_sparse"] = qw
    soft_facts["_query"] = query
    bm25_hits = _bm25.search(query, top_k=10)
    soft_facts["_bm25_top"] = {h["id"]: rank + 1 for rank, h in enumerate(bm25_hits)}
    candidates, relaxations = search_with_relaxation(_settings.db_path, to_hard_filter_params(hard_facts))
    candidates = filter_soft_facts(candidates, soft_facts)
    ranked = rank_listings(candidates, soft_facts)

    if len(ranked) < min_results:
        from app.core.hard_filters import HardFilterParams
        fb_candidates, _ = search_with_relaxation(_settings.db_path, HardFilterParams(limit=200, offset=0))
        fb_candidates = filter_soft_facts(fb_candidates, soft_facts)
        fb_ranked = rank_listings(fb_candidates, soft_facts)
        existing_ids = {r.listing_id for r in ranked}
        extras = [r for r in fb_ranked if r.listing_id not in existing_ids]
        ranked = ranked + extras[:max(0, min_results - len(ranked))]

    return PipelineResponse(
        query=query,
        total_candidates=len(candidates),
        total_results=len(ranked),
        relaxations_applied=relaxations,
        results=[
            _attach_vlm({
                "listing_id": r.listing_id,
                "score": r.score,
                "city": r.listing.city,
                "canton": r.listing.canton,
                "price_chf": r.listing.price_chf,
                "rooms": r.listing.rooms,
                "area_sqm": r.listing.living_area_sqm,
                "features": r.listing.features,
                "street": r.listing.street,
                "title": r.listing.title,
                "hero_image_url": r.listing.hero_image_url,
                "original_url": r.listing.original_listing_url,
            })
            for r in ranked[:top_k]
        ],
    )


@app.post("/pipeline_embed", response_model=PipelineResponse)
def pipeline_embed(body: dict, top_k: int = Query(20, ge=1, le=500)):
    """Hard filter → BGE-M3+SigLIP+BM25 ranking → VLM fields attached."""
    query = body.get("query", "")
    hard_facts = extract_hard_facts(query)
    hard_facts.limit = 5000
    hard_facts.offset = 0
    hard_facts = _resolve_near_place(hard_facts)
    soft_facts = extract_soft_facts(query)
    if hard_facts.neighborhood:
        soft_facts["neighborhoods"] = hard_facts.neighborhood
    qv, qw = S.encode_query(_bge, _ort_tok, _ort_sess, query)
    soft_facts["_query_dense"] = qv
    soft_facts["_query_sparse"] = qw
    soft_facts["_query"] = query
    bm25_hits = _bm25.search(query, top_k=10)
    soft_facts["_bm25_top"] = {h["id"]: rank + 1 for rank, h in enumerate(bm25_hits)}
    candidates, relaxations = search_with_relaxation(_settings.db_path, to_hard_filter_params(hard_facts))
    candidates = filter_soft_facts(candidates, soft_facts)
    ranked = rank_listings(candidates, soft_facts)

    if not ranked:
        return PipelineResponse(query=query, total_candidates=len(candidates), total_results=0,
                                relaxations_applied=relaxations, results=[])

    return PipelineResponse(
        query=query,
        total_candidates=len(candidates),
        total_results=len(ranked),
        relaxations_applied=relaxations,
        results=[
            _attach_vlm({
                "listing_id": r.listing_id,
                "score": r.score,
                "city": r.listing.city,
                "canton": r.listing.canton,
                "price_chf": r.listing.price_chf,
                "rooms": r.listing.rooms,
                "area_sqm": r.listing.living_area_sqm,
                "features": r.listing.features,
                "street": r.listing.street,
                "title": r.listing.title,
                "hero_image_url": r.listing.hero_image_url,
                "original_url": r.listing.original_listing_url,
            })
            for r in ranked[:top_k]
        ],
    )


@app.get("/bm25", response_model=BM25Response)
def bm25_search(
    q: str = Query(..., description="Search query in any language"),
    top: int = Query(10, ge=1, le=50),
    mode: Literal["word", "ngram", "hybrid"] = Query("hybrid"),
    variant: Literal["okapi", "plus", "bm25s"] = Query("bm25s"),
):
    if _bm25.mode != mode or _bm25.variant != variant:
        r = B.MultiLingualRetriever(mode=mode, variant=variant)
        r.load_and_build()
    else:
        r = _bm25
    hits_raw = r.search(q, top_k=top)
    B._load_titles()
    items = []
    for rank, h in enumerate(hits_raw, 1):
        items.append(BM25Hit(
            rank=rank, id=h["id"], score=round(h["score"], 6),
            per_lang=h["per_lang"],
            text=(B._TEXTS_EN.get(h["id"], "") or "")[:400],
        ))
    return BM25Response(query=q, mode=mode, variant=variant, top=top, hits=items)


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Search query in any language"),
    top: int = Query(10, ge=1, le=50),
    mode: Literal["dense", "sparse", "hybrid_rrf", "hybrid_ws"] = Query("hybrid_ws"),
    alpha: float = Query(0.7, ge=0.0, le=1.0, description="weighted-sum dense weight"),
    rrf_k: int = Query(60, ge=1),
    fetch_k: int = Query(200, ge=10, le=1000),
    city: Optional[str] = None,
    rooms_min: Optional[float] = None,
    rooms_max: Optional[float] = None,
    area_min: Optional[float] = None,
    price_max: Optional[float] = None,
):
    hits = S.search(
        _bge, _ort_tok, _ort_sess, _ids, _faiss, _inv, _texts, q,
        mode=mode, alpha=alpha, k_rrf=rrf_k, top=top, fetch_k=fetch_k,
        city=city, rooms_min=rooms_min, rooms_max=rooms_max,
        area_min=area_min, price_max=price_max,
    )
    items = []
    for rank, (lid, score) in enumerate(hits, 1):
        m = S._META_CACHE.get(lid, {})
        items.append(Hit(
            rank=rank, id=lid, score=float(score),
            city=m.get("city"), rooms=m.get("rooms"), area=m.get("area"), price=m.get("price"),
            text=(_texts.get(lid, "") or "")[:400],
        ))
    return SearchResponse(
        query=q, mode=mode, top=top,
        filters={
            "city": city, "rooms_min": rooms_min, "rooms_max": rooms_max,
            "area_min": area_min, "price_max": price_max,
        },
        hits=items,
    )

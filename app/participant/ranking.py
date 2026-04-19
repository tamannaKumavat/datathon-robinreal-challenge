from __future__ import annotations

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.models.schemas import ListingData, RankedListingResult

_CORPUS: tuple | None = None
_SIGLIP_MODEL: tuple | None = None  # (model, processor)
_VLM: dict[str, dict] | None = None
_DATA_DIR = Path("/workshop/retrieval_aws/data")
_VLM_PATH = Path("/workshop/vlm_ranking/combined_results_full_6gpu.jsonl")
RRF_K = 60
_SIGLIP_MODEL_ID = "google/siglip2-so400m-patch14-384"

# Maps soft_facts keys → (vlm_feature_key, is_binary)
# Numeric scores are 1–5; binary features are bool.
_VLM_SIGNAL_MAP: dict[str, tuple[str, bool]] = {
    "bright":        ("brightness_score", False),
    "modern":        ("modernity_score", False),
    "modern_kitchen": ("kitchen_appeal_score", False),
    "furnished":     ("is_furnished", True),
    "nice_views":    ("has_balcony_or_terrace_visible", True),
}


def _load_corpus() -> tuple:
    global _CORPUS
    if _CORPUS is not None:
        return _CORPUS

    d = np.load(_DATA_DIR / "embeddings_bge_dense.npz")
    ids = [str(x) for x in d["ids"]]
    dense_vecs = d["vecs"].astype(np.float32)
    id_to_idx = {lid: i for i, lid in enumerate(ids)}

    sparse_by_id: dict[str, dict[str, float]] = {}
    with open(_DATA_DIR / "embeddings_sparse.jsonl") as f:
        for line in f:
            r = json.loads(line)
            sparse_by_id[r["id"]] = {k: float(v) for k, v in r["weights"].items()}

    siglip: tuple | None = None
    siglip_path = _DATA_DIR / "siglip_image_vecs.npz"
    if siglip_path.exists():
        s = np.load(siglip_path)
        siglip_ids = [str(x) for x in s["ids"]]
        siglip_vecs = s["vecs"].astype(np.float32)
        sid_to_idx = {lid: i for i, lid in enumerate(siglip_ids)}
        siglip = (siglip_vecs, sid_to_idx)

    _CORPUS = (dense_vecs, sparse_by_id, id_to_idx, siglip)
    return _CORPUS


def _load_siglip() -> tuple:
    global _SIGLIP_MODEL
    if _SIGLIP_MODEL is not None:
        return _SIGLIP_MODEL
    import onnxruntime as ort
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(_SIGLIP_MODEL_ID)
    onnx_path = str(_DATA_DIR / "siglip_text_tower.onnx")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    _SIGLIP_MODEL = (sess, proc)
    return _SIGLIP_MODEL


def _encode_query_siglip(query: str) -> np.ndarray | None:
    try:
        sess, proc = _load_siglip()
        inputs = proc(text=[query], return_tensors="np", padding="max_length", truncation=True, max_length=64)
        ids = inputs["input_ids"].astype(np.int64)
        result = sess.run(None, {"input_ids": ids})[0]
        v = result[0].astype(np.float32)
        norm = np.linalg.norm(v)
        return v / max(norm, 1e-9)
    except Exception:
        return None


def _load_vlm() -> dict[str, dict]:
    global _VLM
    if _VLM is not None:
        return _VLM
    _VLM = {}
    if _VLM_PATH.exists():
        with open(_VLM_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("success") and r.get("features"):
                        _VLM[str(r["id"])] = r["features"]
                except Exception:
                    pass
    return _VLM


def _vlm_score(feats: dict, signals: list[tuple[str, bool]]) -> float:
    """Compute normalized [0,1] VLM score from the relevant signal columns."""
    total = 0.0
    count = 0
    for col, is_binary in signals:
        val = feats.get(col)
        if val is None:
            continue
        if is_binary:
            total += 1.0 if val else 0.0
        else:
            # numeric 1–5
            try:
                total += (float(val) - 1.0) / 4.0
            except (TypeError, ValueError):
                continue
        count += 1
    return total / count if count else 0.0


def _rrf_fuse(rank_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for rlist in rank_lists:
        for rank, lid in enumerate(rlist, 1):
            scores[lid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def rank_listings(
    candidates: list[dict[str, Any]],
    soft_facts: dict[str, Any],
) -> list[RankedListingResult]:
    if not candidates:
        return []

    qv: np.ndarray | None = soft_facts.get("_query_dense")
    qw: dict[str, float] | None = soft_facts.get("_query_sparse")

    if qv is None or qw is None:
        # No query vectors — return as-is with stub scores
        return [
            RankedListingResult(
                listing_id=str(c["listing_id"]),
                score=1.0,
                reason="Hard filter match.",
                listing=_to_listing_data(c),
            )
            for c in candidates
        ]

    dense_vecs, sparse_by_id, id_to_idx, siglip = _load_corpus()
    pool = [str(c["listing_id"]) for c in candidates]

    # BGE dense scores
    valid_pool = [lid for lid in pool if lid in id_to_idx]
    cand_idxs = np.array([id_to_idx[lid] for lid in valid_pool])
    d_scores = dense_vecs[cand_idxs] @ qv[0]
    dense_rank = [valid_pool[i] for i in np.argsort(-d_scores)]

    # BGE sparse scores
    sp = np.zeros(len(valid_pool), dtype=np.float32)
    for i, lid in enumerate(valid_pool):
        w = sparse_by_id.get(lid, {})
        sp[i] = sum(qw.get(tok, 0.0) * dw for tok, dw in w.items())
    sparse_rank = [valid_pool[i] for i in np.argsort(-sp)]

    rank_lists: list[list[str]] = [dense_rank, sparse_rank]

    # SigLIP image scores: encode query with text tower → dot product with image vecs
    if siglip is not None:
        siglip_vecs, sid_to_idx = siglip
        query_text = soft_facts.get("_query", "")
        if query_text:
            qsig = _encode_query_siglip(query_text)
            if qsig is not None:
                valid_sig = [lid for lid in valid_pool if lid in sid_to_idx]
                if valid_sig:
                    sig_idxs = np.array([sid_to_idx[lid] for lid in valid_sig])
                    sig_scores = siglip_vecs[sig_idxs] @ qsig
                    siglip_rank = [valid_sig[i] for i in np.argsort(-sig_scores)]
                    rank_lists.append(siglip_rank)

    fused = _rrf_fuse(rank_lists)
    score_map = {lid: score for lid, score in fused}
    cand_map = {str(c["listing_id"]): c for c in candidates}

    results = []
    for lid, rrf_score in fused:
        c = cand_map.get(lid)
        if c is None:
            continue
        results.append(RankedListingResult(
            listing_id=lid,
            score=round(rrf_score, 6),
            reason="BGE-M3 dense+sparse+SigLIP RRF.",
            listing=_to_listing_data(c),
        ))

    # Append any candidates not in BGE corpus (no embedding)
    ranked_ids = set(score_map)
    for c in candidates:
        lid = str(c["listing_id"])
        if lid not in ranked_ids:
            results.append(RankedListingResult(
                listing_id=lid,
                score=0.0,
                reason="No embedding available.",
                listing=_to_listing_data(c),
            ))

    # BM25 diversity injection: top-20% BM25 hits not in top-100 BGE+SigLIP → replace tail
    bm25_top: dict[str, int] = soft_facts.get("_bm25_top", {})
    if bm25_top and results:
        bm25_sorted = sorted(bm25_top, key=bm25_top.__getitem__)
        top20 = bm25_sorted[:max(1, len(bm25_sorted) // 5)]
        top_front_ids = {r.listing_id for r in results[:100]}
        new_from_bm25 = [lid for lid in top20 if lid not in top_front_ids and lid in cand_map]
        if new_from_bm25:
            tail_score = results[-1].score * 0.9 if results else 0.0
            inject = [
                RankedListingResult(
                    listing_id=lid,
                    score=round(tail_score, 6),
                    reason="BM25 diversity.",
                    listing=_to_listing_data(cand_map[lid]),
                )
                for lid in new_from_bm25
            ]
            results = results[: len(results) - len(inject)] + inject

    return results


def _to_listing_data(candidate: dict[str, Any]) -> ListingData:
    return ListingData(
        id=str(candidate["listing_id"]),
        title=candidate["title"],
        description=candidate.get("description"),
        street=candidate.get("street"),
        city=candidate.get("city"),
        postal_code=candidate.get("postal_code"),
        canton=candidate.get("canton"),
        latitude=candidate.get("latitude"),
        longitude=candidate.get("longitude"),
        price_chf=candidate.get("price"),
        rooms=candidate.get("rooms"),
        living_area_sqm=_coerce_int(candidate.get("area")),
        available_from=candidate.get("available_from"),
        image_urls=_coerce_image_urls(candidate.get("image_urls")),
        hero_image_url=candidate.get("hero_image_url"),
        original_listing_url=candidate.get("original_url"),
        features=candidate.get("features") or [],
        offer_type=candidate.get("offer_type"),
        object_category=candidate.get("object_category"),
        object_type=candidate.get("object_type"),
    )


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _coerce_image_urls(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return None

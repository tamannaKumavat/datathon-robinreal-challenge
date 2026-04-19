"""FastAPI HTTP API + MCP HTTP transport.

Exposes:
  GET  /health              - liveness
  POST /search              - single-turn search
  POST /profile/rewrite     - manual profile update (for favourites/hides)
  GET  /profile/{user_id}   - inspect a profile
  ANY  /mcp                 - MCP streamable-http endpoint (handled by app.mcp_server)

Run locally:
  uvicorn app.api:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import settings
from .orchestrator import SearchOrchestrator
from .profile_manager import DynamoDBProfileStore, InMemoryProfileStore
from .schemas import UserProfile

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Robin Search Orchestrator", version="0.1.0")

# Swap to DynamoDBProfileStore() in prod by setting USE_DYNAMO=1
import os
if os.getenv("USE_DYNAMO") == "1":
    profile_store = DynamoDBProfileStore()
    log.info("Using DynamoDB profile store: %s", settings.profile_table_name)
else:
    profile_store = InMemoryProfileStore()
    log.info("Using in-memory profile store")

orchestrator = SearchOrchestrator(profile_store=profile_store)


# ---------- Request/response models ----------

class SearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    top_k: int = 20
    rich_explanations: bool = False
    update_profile: bool = False
    favourited_ids: Optional[list[str]] = None
    hidden_ids: Optional[list[str]] = None


class ProfileRewriteRequest(BaseModel):
    user_id: str
    # Either provide a query (we'll re-extract) or a pre-extracted intent
    query: str
    favourited_ids: Optional[list[str]] = None
    hidden_ids: Optional[list[str]] = None


# ---------- Endpoints ----------

@app.get("/health")
def health():
    try:
        backend = orchestrator.pipeline.health()
    except Exception as e:
        backend = {"error": str(e)}
    return {"status": "ok", "backend": backend}


@app.post("/search")
def search(req: SearchRequest):
    try:
        return orchestrator.search(
            query=req.query,
            user_id=req.user_id,
            top_k=req.top_k,
            rich=req.rich_explanations,
            update_profile=req.update_profile,
            favourited_ids=req.favourited_ids,
            hidden_ids=req.hidden_ids,
        )
    except Exception as e:
        log.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/profile/rewrite")
def profile_rewrite(req: ProfileRewriteRequest):
    current = profile_store.get(req.user_id) or UserProfile(user_id=req.user_id)
    extracted = orchestrator.extractor.extract(req.query, current)
    updated = orchestrator.rewriter.rewrite(
        current, req.query, extracted,
        favourited_ids=req.favourited_ids,
        hidden_ids=req.hidden_ids,
    )
    profile_store.put(updated)
    return updated.model_dump()


@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    p = profile_store.get(user_id)
    if not p:
        raise HTTPException(404, "Profile not found")
    return p.model_dump()


# Mount MCP at /mcp
from .mcp_server import mount_mcp  # noqa: E402
mount_mcp(app, orchestrator)

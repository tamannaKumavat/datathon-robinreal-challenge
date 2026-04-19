# AWS Instance

**IP:** `54.184.212.11` | Amazon Linux 2023 (arm64) | Working dir: `/workshop`

---

## Services

| Service | Port | Entry point |
|---------|------|-------------|
| Retrieval API | 8000 | `uvicorn serve:app` in `/workshop/datathon-robinreal-challenge` |
| MCP Orchestrator | 8081 | `uvicorn app.api:app` in `/workshop/infra_extracted` |

---

## What We Did

**VLM integration** — Pre-computed visual scores (brightness, modernity, condition, spaciousness, kitchen/bathroom appeal) from a 6-GPU run were loaded into the ranking pipeline. Extended from 3,484 → 4,274 listings by replacing the JSONL/CSV on the instance. Scores are attached to every `/pipeline_embed` response and used by the infra ranker.

**MCP orchestrator on port 8081** — Deployed a second FastAPI service exposing two MCP tools (`search_listings`, `get_user_profile`). It calls the retrieval API on port 8000, re-ranks results with VLM + user profile boosts, and writes per-listing explanations via Claude Sonnet.

**Port 8081 opened in AWS security group** — Port 8000 was already open; 8081 was not. Fixed with:
```bash
aws ec2 authorize-security-group-ingress \
  --group-id sg-097dbbab5cca5b6e7 \
  --protocol tcp --port 8081 --cidr 0.0.0.0/0
```

**Teammate MCP connection** — After opening the port, teammates connected by adding to Cursor/Claude Desktop config:
```json
{
  "mcpServers": {
    "robin-search": { "type": "http", "url": "http://54.184.212.11:8081/mcp" }
  }
}
```

**Multi-turn deduplication** — Added a rolling `seen_listing_ids` window (200 entries) to `UserProfile`. Previously shown listings get a `+0.3` penalty at rerank time. Profile is auto-updated after every `search_listings` call when a `user_id` is passed — no client-side session management needed.

**3-turn validation** — Ran a full conversation on the instance: 3 consecutive queries returned 9 unique listings with zero repeats across turns. Profile accumulated `quietness: 0.7`, `near_lake_or_green: 0.8` from the conversation.

---

## MCP Protocol Notes

The streamable-HTTP transport requires a two-step handshake:
1. `POST /mcp/` with `initialize` → capture `Mcp-Session-Id` from response headers
2. All subsequent calls include `mcp-session-id: <id>` header

The trailing slash on `/mcp/` matters — `/mcp` redirects with a 307 that some clients don't follow.

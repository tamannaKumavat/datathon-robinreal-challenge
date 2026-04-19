"""MCP server — exposes the orchestrator as tools for Claude Desktop and other MCP clients.

We use the HTTP (streamable-http) transport so the server can run on AWS
and be reached by any MCP client over HTTPS. For local Claude Desktop,
you can alternatively run `python -m app.mcp_stdio` for stdio transport.

Tools exposed:
  - search_apartments:         the main search
  - explain_listing:           get a detailed explanation for one listing
  - update_user_profile:       push favourites/hides and trigger profile rewrite
  - get_user_profile:          inspect a user profile
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import TextContent, Tool

from .orchestrator import SearchOrchestrator
from .profile_manager import InMemoryProfileStore
from .schemas import UserProfile

log = logging.getLogger(__name__)


def _build_mcp_server(orchestrator: SearchOrchestrator) -> Server:
    server = Server("robin-search")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="search_listings",
                description=(
                    "Search Swiss real-estate listings from a natural-language query. "
                    "Handles hard filters (city, price, rooms, area), soft preferences "
                    "(bright, quiet, modern, family-friendly), commute constraints "
                    "(e.g. 'within 25 min of ETH'), and automatic relaxation when "
                    "results are sparse. Optionally personalizes to a saved user profile."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's natural-language apartment query. "
                                           "English, German, or French accepted.",
                        },
                        "user_id": {
                            "type": "string",
                            "description": "Optional. If provided, the user's stored "
                                           "profile is used to fill in missing defaults.",
                        },
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 10,
                        },
                        "rich_explanations": {
                            "type": "boolean",
                            "default": False,
                            "description": "If true, generate natural-language "
                                           "summaries per listing (slower, higher cost).",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_user_profile",
                description="Return the stored preference profile for a user id, or null if none exists.",
                inputSchema={
                    "type": "object",
                    "properties": {"user_id": {"type": "string"}},
                    "required": ["user_id"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            if name == "search_listings":
                user_id = arguments.get("user_id")
                result = orchestrator.search(
                    query=arguments["query"],
                    user_id=user_id,
                    top_k=arguments.get("top_k", 10),
                    rich=arguments.get("rich_explanations", True),
                    update_profile=bool(user_id),
                )
                return [TextContent(type="text", text=_render_cards(result, user_id))]

            if name == "get_user_profile":
                p = orchestrator.profile_store.get(arguments["user_id"])
                text = "null" if p is None else p.model_dump_json(indent=2)
                return [TextContent(type="text", text=text)]

            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        except Exception as e:
            log.exception("MCP tool failed: %s", name)
            return [TextContent(type="text", text=f"Error: {e}")]

    return server


def _render_cards(result: dict[str, Any], user_id: Optional[str]) -> str:
    """Render search results as Markdown listing cards with favourite CTA."""
    lines: list[str] = []

    exp = result.get("explanations", {})
    headline = exp.get("headline", "")
    if headline:
        lines.append(f"## {headline}\n")

    caveats = exp.get("caveats") or []
    if caveats:
        lines.append("> " + " · ".join(caveats) + "\n")

    why_map = {p["listing_id"]: p["why"] for p in exp.get("per_listing", [])}

    for item in result.get("results", []):
        lid = str(item.get("listing_id", ""))
        title = item.get("title") or f"Listing {lid}"
        city = item.get("city", "")
        canton = item.get("canton", "")
        price = item.get("price_chf")
        rooms = item.get("rooms")
        area = item.get("area_sqm")
        features = item.get("features") or []
        hero = item.get("hero_image_url")
        url = item.get("original_url")
        why = why_map.get(lid, "")

        lines.append(f"---\n### {title}")
        if hero:
            lines.append(f"![{title}]({hero})")

        meta_parts = []
        if city:
            meta_parts.append(f"📍 {city}{', ' + canton if canton else ''}")
        if price:
            meta_parts.append(f"💰 CHF {price:,}/mo")
        if rooms:
            meta_parts.append(f"🛏 {rooms} rooms")
        if area:
            meta_parts.append(f"📐 {area} m²")
        if meta_parts:
            lines.append("  ".join(meta_parts))

        if features:
            lines.append("**Features:** " + " · ".join(features[:6]))

        if why:
            lines.append(f"*{why}*")

        action_parts = []
        if url:
            action_parts.append(f"[View listing]({url})")
        fav_cmd = f'`favourite listing {lid}`'
        if user_id:
            fav_cmd = f'`favourite listing {lid}`  *(calls update_user_profile)*'
        action_parts.append(f"⭐ {fav_cmd}")
        lines.append("  ".join(action_parts))
        lines.append("")

    relax = result.get("backend_relaxations_applied") or result.get("our_relaxations")
    if relax:
        lines.append(f"\n> ℹ️ Search was relaxed: {relax}")

    if user_id:
        lines.append(f"\n*Profile `{user_id}` updated — previously seen listings are downranked.*")

    return "\n".join(lines)


def mount_mcp(app, orchestrator: SearchOrchestrator) -> None:
    """Mount the MCP streamable-http handler under /mcp on the FastAPI app."""
    server = _build_mcp_server(orchestrator)
    session_manager = StreamableHTTPSessionManager(app=server, json_response=True)

    # 【核心修改】：提前调用 .run() 获取异步上下文管理器
    session_ctx = session_manager.run()

    # Start the session manager inside the FastAPI lifespan
    @app.on_event("startup")
    async def _startup():
        # 改为使用 session_ctx 来触发 __aenter__
        await session_ctx.__aenter__()

    @app.on_event("shutdown")
    async def _shutdown():
        # 改为使用 session_ctx 来触发 __aexit__
        await session_ctx.__aexit__(None, None, None)

    async def handle_mcp(scope, receive, send):
        await session_manager.handle_request(scope, receive, send)

    # Register the ASGI endpoint. FastAPI doesn't directly accept raw ASGI,
    # so we use Starlette's Mount via add_route-style wrapping.
    from starlette.routing import Mount
    app.router.routes.append(Mount("/mcp", app=handle_mcp))
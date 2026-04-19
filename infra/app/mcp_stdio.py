"""Stdio MCP transport for local Claude Desktop use.

Run with:
  python -m app.mcp_stdio
"""
from __future__ import annotations

import asyncio
import logging

from mcp.server.stdio import stdio_server

from .mcp_server import _build_mcp_server
from .orchestrator import SearchOrchestrator


async def main():
    logging.basicConfig(level=logging.INFO)
    orch = SearchOrchestrator()
    server = _build_mcp_server(orch)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())

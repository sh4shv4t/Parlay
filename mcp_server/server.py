"""
Parlay MCP server entry point.
Supports both stdio (Claude Desktop, Claude Code, any stdio MCP client)
and SSE transport (Continue.dev, Zed, HTTP-based clients).

Usage:
    python -m mcp_server.server stdio     # stdio transport (default)
    python -m mcp_server.server sse       # SSE on port 8002
"""
import sys
import logging

from .tools import mcp

logger = logging.getLogger(__name__)


def main() -> None:
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "sse":
        logger.info("Starting Parlay MCP server on SSE transport (port 8002)")
        mcp.run(transport="sse", host="0.0.0.0", port=8002)
    else:
        logger.info("Starting Parlay MCP server on stdio transport")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

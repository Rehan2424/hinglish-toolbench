"""MCP server: lets any MCP client (an IDE agent, a desktop assistant) use the model.

    htb mcp --provider local --base-url http://localhost:8080/v1

A large agent can hand a Hinglish request to the small local model, get back
validated tool calls, and decide what to do with them. Runs over stdio.
"""

from __future__ import annotations

from typing import Any

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

from . import mock_backend
from .assistant import Assistant, ChatBackend
from .tools import TOOLS


def build_server(backend: ChatBackend) -> MCPServer:
    server = MCPServer("hinglish-toolbench")
    assistant = Assistant(backend)

    @server.tool()
    def understand_request(text: str, city: str = "Delhi", now: str | None = None) -> dict[str, Any]:
        """Turn a Hinglish, Hindi or English request into validated tool calls.

        Returns the action (call, clarify or respond), the calls with their arguments,
        and any validation errors. `now` is local time as YYYY-MM-DDTHH:MM (default: IST now).
        """
        return assistant.run([{"role": "user", "content": text}], city=city, now=now).to_dict()

    @server.tool()
    def run_mock_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute one validated call against the demo backend (fake data, no side effects)."""
        return mock_backend.execute(name, arguments)

    @server.resource("htb://tools")
    def tool_schemas() -> list[dict[str, Any]]:
        """JSON schemas of the 17 supported tools."""
        return [t.json_schema() for t in TOOLS.values()]

    return server

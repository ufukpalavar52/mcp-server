"""
The MCP protocol surface.

Both handlers read from the in-memory catalogue, which the gateway publishes. This
service reaches out to nothing to answer them, so an MCP client sees exactly the set of
tools the gateway last published — no more, and never a stale copy fetched behind its
back.
"""

from __future__ import annotations

import json
import logging

from mcp.server.lowlevel import Server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)

from app.catalogue import Catalogue
from app.dispatcher import Dispatcher
from app.planner import Planner

logger = logging.getLogger(__name__)

SERVER_NAME = "mcp-server"


def build_mcp_server(
    catalogue: Catalogue, planner: Planner, dispatcher: Dispatcher
) -> Server:
    """Wire the two MCP methods this server implements onto a low level Server."""
    server: Server = Server(SERVER_NAME, version="0.1.0")

    async def handle_list_tools(_ctx, _params) -> ListToolsResult:
        """``tools/list`` — whatever the gateway last published."""
        return ListToolsResult(
            tools=[
                Tool(
                    name=definition.tool_name,
                    description=definition.tool_description or definition.name,
                    input_schema=definition.input_schema(),
                )
                for definition in catalogue.all()
            ]
        )

    async def handle_call_tool(_ctx, params: CallToolRequestParams) -> CallToolResult:
        """
        ``tools/call`` — resolve the call into a plan.

        The same planning path the gateway's REST endpoint uses, so an MCP client and
        the gateway can never be told different things about the same call.
        """
        definition = catalogue.find(params.name)

        if definition is None:
            return _result(
                {
                    "status": "error",
                    "message": (
                        f"Unknown tool: {params.name}. "
                        "The gateway may not have published its catalogue yet."
                    ),
                },
                is_error=True,
            )

        plan = await planner.plan(definition, params.arguments or {})
        dispatch = await dispatcher.dispatch(definition, plan, arguments, actor=None)

        logger.info(
            "Planned %s via MCP: status=%s actions=%d dispatch=%s",
            plan.tool, plan.status, len(plan.actions), dispatch.status,
        )

        return _result(
            {
                "status": plan.status,
                "plan": plan.model_dump(mode="json"),
                "dispatch": dispatch.model_dump(mode="json"),
            },
            # A rejected plan is a real failure of the request, and the client should
            # see it as one rather than having to parse the payload to find out.
            is_error=plan.status == "rejected",
        )

    # PaginatedRequestParams is all optional, so a tools/list with no params member
    # still reaches the handler with defaults rather than being rejected.
    server.add_request_handler("tools/list", PaginatedRequestParams, handle_list_tools)
    server.add_request_handler("tools/call", CallToolRequestParams, handle_call_tool)

    return server


def _result(payload: dict, *, is_error: bool) -> CallToolResult:
    """Every result carries JSON, so a client can parse it without branching first."""
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(payload, indent=2, ensure_ascii=False),
            )
        ],
        is_error=is_error,
    )

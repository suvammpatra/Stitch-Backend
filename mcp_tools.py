"""
Google Stitch MCP Tool Client
Wraps all MCP tool calls with:
- OAuth Bearer token injection
- Typed input validation
- Retry with exponential backoff
- 401 → token invalidation signal
- Structured error returns
"""

import asyncio
import json
import logging
from typing import Any, Optional

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger("stitch.mcp")

MCP_ENDPOINT = "https://stitch.googleapis.com/mcp"
DEFAULT_TIMEOUT = 60.0   # Stitch can be slow; 60s is realistic
MAX_MCP_TIMEOUT = 120.0  # generate_screen_from_text can take longer


# ─── Input schemas ────────────────────────────────────────────────────────────

class CreateProjectInput(BaseModel):
    display_name: str = Field(..., description="Human-readable project name")
    description: str = Field(default="", description="Optional project description")


class ListProjectsInput(BaseModel):
    page_size: int = Field(default=20, description="Max projects to return")
    page_token: str = Field(default="", description="Pagination token")


class GetProjectInput(BaseModel):
    project_id: str = Field(..., description="The Stitch project ID")


class GenerateScreenInput(BaseModel):
    project_id: str = Field(..., description="Target project ID")
    prompt: str = Field(..., description="Text description of the screen to generate")
    screen_name: str = Field(default="", description="Optional name for the screen")


class GetScreenInput(BaseModel):
    project_id: str = Field(..., description="The project ID")
    screen_id: str = Field(..., description="The screen ID to retrieve")


class ListScreensInput(BaseModel):
    project_id: str = Field(..., description="The project ID")
    page_size: int = Field(default=20, description="Max screens to return")


# ─── MCP Client ──────────────────────────────────────────────────────────────

class AuthRequired(Exception):
    """Raised when the MCP endpoint returns 401. Signals token invalidation."""
    pass


class StitchMCPClient:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self._headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    async def call_tool(
        self,
        tool_name: str,
        args: dict,
        max_retries: int = 2,
    ) -> tuple[Any, Optional[str]]:
        """
        Call an MCP tool. Returns (result, error).
        error is None on success, string on failure.
        """
        # Determine timeout — generation is slow
        timeout = MAX_MCP_TIMEOUT if "generate" in tool_name else DEFAULT_TIMEOUT

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": args,
            },
        }

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        MCP_ENDPOINT,
                        headers=self._headers,
                        json=payload,
                    )

                if resp.status_code == 401:
                    raise AuthRequired("Token rejected by Stitch MCP endpoint (401)")

                if resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    logger.warning(f"[mcp] Rate limited on {tool_name}, waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning(f"[mcp] Server error {resp.status_code} on {tool_name}, retry in {wait}s")
                    await asyncio.sleep(wait)
                    last_error = f"Stitch server error {resp.status_code}"
                    continue

                if resp.status_code != 200:
                    return None, f"Unexpected HTTP {resp.status_code}: {resp.text[:200]}"

                body = resp.json()

                if "error" in body:
                    err = body["error"]
                    return None, f"MCP error {err.get('code', '?')}: {err.get('message', str(err))}"

                # MCP result: body["result"]["content"] is list of {type, text} blocks
                result = body.get("result", {})
                content_blocks = result.get("content", [])
                if content_blocks and content_blocks[0].get("type") == "text":
                    raw_text = content_blocks[0]["text"]
                    try:
                        return json.loads(raw_text), None
                    except json.JSONDecodeError:
                        return {"text": raw_text}, None

                return result, None

            except AuthRequired:
                raise  # Propagate up — triggers token invalidation
            except httpx.TimeoutException:
                last_error = f"Timeout calling {tool_name} after {timeout}s"
                logger.warning(f"[mcp] {last_error} (attempt {attempt+1})")
                await asyncio.sleep(2 ** attempt)
            except httpx.RequestError as e:
                last_error = f"Network error: {e}"
                logger.warning(f"[mcp] {last_error} (attempt {attempt+1})")
                await asyncio.sleep(2 ** attempt)

        return None, last_error or f"Failed after {max_retries+1} attempts"

    # ─── LangChain Tool factory ───────────────────────────────────────────

    def get_langchain_tools(self) -> list[StructuredTool]:
        """Returns LangChain StructuredTools for all Stitch MCP actions."""

        async def _call(name: str, args: dict) -> str:
            result, error = await self.call_tool(name, args)
            if error:
                return f"Tool error: {error}"
            return json.dumps(result, indent=2)

        tools = [
            StructuredTool.from_function(
                coroutine=lambda **kw: _call("create_project", kw),
                name="create_project",
                description="Create a new Google Stitch project.",
                args_schema=CreateProjectInput,
            ),
            StructuredTool.from_function(
                coroutine=lambda **kw: _call("list_projects", kw),
                name="list_projects",
                description="List all Google Stitch projects for the authenticated user.",
                args_schema=ListProjectsInput,
            ),
            StructuredTool.from_function(
                coroutine=lambda **kw: _call("get_project", kw),
                name="get_project",
                description="Get details of a specific Stitch project by ID.",
                args_schema=GetProjectInput,
            ),
            StructuredTool.from_function(
                coroutine=lambda **kw: _call("generate_screen_from_text", kw),
                name="generate_screen_from_text",
                description="Generate a UI screen inside a Stitch project from a text prompt.",
                args_schema=GenerateScreenInput,
            ),
            StructuredTool.from_function(
                coroutine=lambda **kw: _call("get_screen", kw),
                name="get_screen",
                description="Retrieve a specific screen from a Stitch project.",
                args_schema=GetScreenInput,
            ),
            StructuredTool.from_function(
                coroutine=lambda **kw: _call("list_screens", kw),
                name="list_screens",
                description="List all screens in a Stitch project.",
                args_schema=ListScreensInput,
            ),
        ]
        return tools

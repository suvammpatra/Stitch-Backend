"""
LangGraph Graph — Google Stitch Agent
Nodes:
  - preprocess     : inject system prompt + history
  - agent          : Groq function-calling loop
  - tool_executor  : calls MCP tools with OAuth tokens
  - error_handler  : surfaces friendly errors
  - finalizer      : post-processes output for Telegram

State: StitchState (TypedDict)
Model: llama-3.3-70b-versatile via Groq (supports tool_use)
"""

import asyncio
import json
import logging
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from mcp_tools import StitchMCPClient
from oauth_manager import OAuthManager, TokenExpiredError

logger = logging.getLogger("stitch.graph")

SYSTEM_PROMPT = """You are a Google Stitch assistant operating via Telegram.
You help users manage their Stitch UI projects and screens using available tools.

Rules:
- Always call a tool when the user requests a project or screen action.
- After calling a tool, summarize the result clearly and concisely.
- If a tool fails, explain what went wrong and suggest next steps.
- Format all responses in Telegram-compatible Markdown.
- Never hallucinate project IDs or screen names — always retrieve them first.
- If the user's intent is ambiguous, ask ONE clarifying question.

Available actions you can take:
1. list_projects — list all Stitch projects for the user
2. create_project — create a new Stitch project
3. generate_screen_from_text — generate a UI screen from a text description
4. get_screen — retrieve details about a specific screen
5. list_screens — list all screens in a project
6. get_project — get details about a specific project
"""


# ─── State ───────────────────────────────────────────────────────────────────

class StitchState(TypedDict):
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    tool_calls_log: list[dict]       # For observability
    last_tool_error: str | None
    retry_count: int
    final_reply: str | None


# ─── Graph ───────────────────────────────────────────────────────────────────

class StitchGraph:
    MAX_TOOL_RETRIES = 2
    MAX_AGENT_ITERATIONS = 8

    def __init__(self, oauth_manager: OAuthManager):
        self.oauth = oauth_manager
        # llama-3.3-70b-versatile: best Groq model with reliable tool_use support
        # llama-3.1-8b-instant: faster/cheaper fallback
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_retries=3,
            request_timeout=30,
        )
        self.graph = self._build()

    def _build(self):
        builder = StateGraph(StitchState)

        builder.add_node("preprocess", self._preprocess_node)
        builder.add_node("agent", self._agent_node)
        builder.add_node("tool_executor", self._tool_executor_node)
        builder.add_node("error_handler", self._error_handler_node)
        builder.add_node("finalizer", self._finalizer_node)

        builder.add_edge(START, "preprocess")
        builder.add_edge("preprocess", "agent")
        builder.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {
                "tools": "tool_executor",
                "done": "finalizer",
                "error": "error_handler",
            },
        )
        builder.add_conditional_edges(
            "tool_executor",
            self._route_after_tools,
            {
                "agent": "agent",
                "error": "error_handler",
            },
        )
        builder.add_edge("error_handler", "finalizer")
        builder.add_edge("finalizer", END)

        return builder.compile()

    # ── Nodes ─────────────────────────────────────────────────────────────

    async def _preprocess_node(self, state: StitchState) -> dict:
        """Inject system prompt + historical messages."""
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        return {"messages": messages}

    async def _agent_node(self, state: StitchState) -> dict:
        """Run the LLM with tools bound. Respects iteration cap."""
        session_id = state["session_id"]
        iteration = state.get("retry_count", 0)

        if iteration >= self.MAX_AGENT_ITERATIONS:
            logger.warning(f"[agent] Max iterations hit for session {session_id}")
            return {
                "last_tool_error": "Max reasoning steps reached. Please rephrase your request.",
                "retry_count": iteration,
            }

        # Build tool definitions dynamically (with OAuth token for user)
        try:
            token = await self.oauth.get_valid_token(session_id)
        except TokenExpiredError:
            return {
                "last_tool_error": "Authentication expired. Please re-authorize Google Stitch.",
            }

        mcp = StitchMCPClient(access_token=token)
        tools = mcp.get_langchain_tools()
        llm_with_tools = self.llm.bind_tools(tools)

        try:
            response = await llm_with_tools.ainvoke(state["messages"])
            return {
                "messages": [response],
                "retry_count": iteration,
            }
        except Exception as e:
            logger.exception(f"[agent] LLM call failed: {e}")
            return {
                "last_tool_error": f"LLM error: {str(e)}",
            }

    async def _tool_executor_node(self, state: StitchState) -> dict:
        """Execute all tool calls from the latest AI message."""
        session_id = state["session_id"]
        last_message = state["messages"][-1]

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {}

        try:
            token = await self.oauth.get_valid_token(session_id)
        except TokenExpiredError as e:
            return {"last_tool_error": str(e)}

        mcp = StitchMCPClient(access_token=token)
        tool_results = []
        tool_log = list(state.get("tool_calls_log", []))

        for tc in last_message.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]

            logger.info(f"[tools] Calling {tool_name} args={tool_args} session={session_id}")

            result, error = await mcp.call_tool(
                tool_name=tool_name,
                args=tool_args,
                max_retries=self.MAX_TOOL_RETRIES,
            )

            tool_log.append({
                "tool": tool_name,
                "args": tool_args,
                "success": error is None,
                "error": error,
            })

            content = json.dumps(result) if error is None else f"ERROR: {error}"
            tool_results.append(
                ToolMessage(content=content, tool_call_id=tool_id)
            )

            if error:
                logger.warning(f"[tools] {tool_name} failed: {error}")

        return {
            "messages": tool_results,
            "tool_calls_log": tool_log,
            "last_tool_error": None,
        }

    async def _error_handler_node(self, state: StitchState) -> dict:
        error = state.get("last_tool_error", "An unknown error occurred.")
        logger.error(f"[error_handler] session={state['session_id']} error={error}")
        return {
            "final_reply": f"❌ *Error:* {error}\n\nPlease try again or contact support.",
        }

    async def _finalizer_node(self, state: StitchState) -> dict:
        """Extract the last AI text response as the reply."""
        if state.get("final_reply"):
            return {}  # Already set by error handler

        # Walk messages in reverse to find last non-tool AIMessage
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
                return {"final_reply": msg.content}

        return {"final_reply": "✅ Done. No output was returned by the tool."}

    # ── Routers ───────────────────────────────────────────────────────────

    def _route_after_agent(self, state: StitchState) -> Literal["tools", "done", "error"]:
        if state.get("last_tool_error"):
            return "error"

        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "done"

    def _route_after_tools(self, state: StitchState) -> Literal["agent", "error"]:
        if state.get("last_tool_error"):
            return "error"
        return "agent"

    # ── Public run() ──────────────────────────────────────────────────────

    async def run(self, session_id: str, message: str, history: list[dict]) -> dict:
        """
        Entrypoint called by FastAPI.
        history = [{"role": "user"|"assistant", "content": "..."}]
        """
        messages: list[BaseMessage] = []
        for turn in history[-10:]:  # Limit context window
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))

        messages.append(HumanMessage(content=message))

        initial_state: StitchState = {
            "session_id": session_id,
            "messages": messages,
            "tool_calls_log": [],
            "last_tool_error": None,
            "retry_count": 0,
            "final_reply": None,
        }

        final_state = await self.graph.ainvoke(initial_state)

        return {
            "reply": final_state.get("final_reply", ""),
            "tool_calls": final_state.get("tool_calls_log", []),
        }

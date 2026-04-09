"""
LangGraph Graph — Google Stitch Agent
Nodes:
  - agent          : Groq function-calling loop (system prompt injected here)
  - tool_executor  : calls MCP tools with OAuth tokens
  - error_handler  : surfaces friendly errors
  - finalizer      : extracts final reply text

State: StitchState (TypedDict)
Model: llama-3.3-70b-versatile via Groq (supports tool_use)

Fixes vs v1:
  - Removed separate preprocess node (caused empty-dict state update crash)
  - System prompt injected directly inside agent node before LLM call
  - _finalizer_node never returns {} — always writes final_reply
  - _tool_executor_node never returns {} — always writes tool_calls_log
  - retry_count incremented properly on each agent loop
"""

import json
import logging
from typing import Annotated, Literal, TypedDict

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

Available actions:
1. list_projects — list all Stitch projects
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
    tool_calls_log: list[dict]
    last_tool_error: str | None
    retry_count: int
    final_reply: str | None


# ─── Graph ───────────────────────────────────────────────────────────────────

class StitchGraph:
    MAX_TOOL_RETRIES = 2
    MAX_AGENT_ITERATIONS = 8

    def __init__(self, oauth_manager: OAuthManager):
        self.oauth = oauth_manager
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_retries=3,
            request_timeout=30,
        )
        self.graph = self._build()

    def _build(self):
        builder = StateGraph(StitchState)

        # No separate preprocess node — agent handles everything
        builder.add_node("agent", self._agent_node)
        builder.add_node("tool_executor", self._tool_executor_node)
        builder.add_node("error_handler", self._error_handler_node)
        builder.add_node("finalizer", self._finalizer_node)

        builder.add_edge(START, "agent")
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

    async def _agent_node(self, state: StitchState) -> dict:
        """Run Groq LLM with tools. Injects system prompt on first iteration."""
        session_id = state["session_id"]
        iteration = state.get("retry_count", 0)

        if iteration >= self.MAX_AGENT_ITERATIONS:
            logger.warning(f"[agent] Max iterations hit for session={session_id}")
            return {
                "last_tool_error": "Max reasoning steps reached. Please rephrase.",
                "retry_count": iteration + 1,
                "final_reply": None,
            }

        # Get valid OAuth token (falls back to static token)
        try:
            token = await self.oauth.get_valid_token(session_id)
        except TokenExpiredError:
            return {
                "last_tool_error": "Google Stitch authentication expired. Please re-authorize.",
                "retry_count": iteration + 1,
                "final_reply": None,
            }

        # Build tool list
        mcp = StitchMCPClient(access_token=token)
        tools = mcp.get_langchain_tools()
        llm_with_tools = self.llm.bind_tools(tools)

        # Prepend system prompt only on first iteration (avoid duplication)
        messages = state["messages"]
        if iteration == 0:
            # Check if system prompt already present
            has_system = any(isinstance(m, SystemMessage) for m in messages)
            if not has_system:
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        try:
            response = await llm_with_tools.ainvoke(messages)
            logger.info(f"[agent] iter={iteration} session={session_id} tool_calls={len(getattr(response, 'tool_calls', []))}")
            return {
                "messages": [response],
                "retry_count": iteration + 1,
                "last_tool_error": None,
            }
        except Exception as e:
            logger.exception(f"[agent] LLM call failed: {e}")
            return {
                "last_tool_error": f"LLM error: {str(e)}",
                "retry_count": iteration + 1,
                "final_reply": None,
            }

    async def _tool_executor_node(self, state: StitchState) -> dict:
        """Execute all tool calls from the latest AI message."""
        session_id = state["session_id"]
        last_message = state["messages"][-1]
        tool_log = list(state.get("tool_calls_log", []))

        # Guard: no tool calls present
        if not getattr(last_message, "tool_calls", None):
            logger.warning(f"[tools] tool_executor called but no tool_calls found")
            return {
                "tool_calls_log": tool_log,
                "last_tool_error": None,
                "messages": [],
            }

        try:
            token = await self.oauth.get_valid_token(session_id)
        except TokenExpiredError as e:
            return {
                "last_tool_error": str(e),
                "tool_calls_log": tool_log,
                "messages": [],
            }

        mcp = StitchMCPClient(access_token=token)
        tool_messages = []

        for tc in last_message.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]

            logger.info(f"[tools] calling={tool_name} session={session_id}")

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

            content = json.dumps(result) if result is not None else f"ERROR: {error}"
            tool_messages.append(ToolMessage(content=content, tool_call_id=tool_id))

            if error:
                logger.warning(f"[tools] {tool_name} failed: {error}")

        return {
            "messages": tool_messages,
            "tool_calls_log": tool_log,
            "last_tool_error": None,
        }

    async def _error_handler_node(self, state: StitchState) -> dict:
        error = state.get("last_tool_error") or "An unknown error occurred."
        logger.error(f"[error] session={state['session_id']} error={error}")
        return {
            "final_reply": f"❌ *Error:* {error}\n\nPlease try again.",
            "last_tool_error": error,
        }

    async def _finalizer_node(self, state: StitchState) -> dict:
        """Always writes final_reply — never returns empty dict."""
        # Error handler already set it
        if state.get("final_reply"):
            return {"final_reply": state["final_reply"]}

        # Find last non-empty AIMessage text
        for msg in reversed(state["messages"]):
            if (
                isinstance(msg, AIMessage)
                and isinstance(msg.content, str)
                and msg.content.strip()
            ):
                return {"final_reply": msg.content}

        return {"final_reply": "✅ Done. No text output was returned by the tool."}

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

    # ── Public entrypoint ─────────────────────────────────────────────────

    async def run(self, session_id: str, message: str, history: list[dict]) -> dict:
        """
        Called by FastAPI /invoke.
        history = [{"role": "user"|"assistant", "content": "..."}]
        """
        messages: list[BaseMessage] = []

        # Load last 10 turns of history (20 messages)
        for turn in history[-20:]:
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

        logger.info(f"[run] session={session_id} message={message[:60]!r}")

        try:
            final_state = await self.graph.ainvoke(initial_state)
        except Exception as e:
            logger.exception(f"[run] graph.ainvoke crashed: {e}")
            return {
                "reply": f"❌ Internal error: {str(e)}",
                "tool_calls": [],
            }

        return {
            "reply": final_state.get("final_reply") or "No reply generated.",
            "tool_calls": final_state.get("tool_calls_log", []),
        }

"""
Google Stitch LangGraph Backend
Production-grade FastAPI + LangGraph service
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from graph import StitchGraph
from oauth_manager import OAuthManager
from session_store import SessionStore

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stitch.main")

# ─── App lifespan ────────────────────────────────────────────────────────────
redis_client: aioredis.Redis | None = None
oauth_manager: OAuthManager | None = None
session_store: SessionStore | None = None
stitch_graph: StitchGraph | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, oauth_manager, session_store, stitch_graph

    logger.info("Starting up Stitch backend...")

    redis_url = os.environ["REDIS_URL"]  # Fail fast if not set
    redis_client = aioredis.from_url(redis_url, decode_responses=True)
    await redis_client.ping()
    logger.info("Redis connected.")

    oauth_manager = OAuthManager(redis_client)
    session_store = SessionStore(redis_client)
    stitch_graph = StitchGraph(oauth_manager=oauth_manager)

    logger.info("LangGraph compiled and ready.")
    yield

    logger.info("Shutting down...")
    await redis_client.aclose()


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Google Stitch LangGraph Backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

N8N_SECRET = os.getenv("N8N_WEBHOOK_SECRET", "")


# ─── Auth guard ──────────────────────────────────────────────────────────────
def verify_n8n_secret(x_n8n_secret: str = Header(default="")):
    if N8N_SECRET and x_n8n_secret != N8N_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")


# ─── Models ──────────────────────────────────────────────────────────────────
class InvokeRequest(BaseModel):
    session_id: str          # Telegram user ID (string)
    message: str             # User's message text (stripped of /stitch)
    telegram_chat_id: str    # For reply routing


class InvokeResponse(BaseModel):
    session_id: str
    reply: str
    tool_calls: list[dict] = []
    error: str | None = None


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        await redis_client.ping()
        return {"status": "ok", "redis": "connected"}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})


@app.post("/invoke", response_model=InvokeResponse, dependencies=[Depends(verify_n8n_secret)])
async def invoke(req: InvokeRequest):
    logger.info(f"[invoke] session={req.session_id} msg={req.message[:60]!r}")

    try:
        history = await session_store.get_history(req.session_id)

        result = await stitch_graph.run(
            session_id=req.session_id,
            message=req.message,
            history=history,
        )

        await session_store.append_turn(
            req.session_id,
            user=req.message,
            assistant=result["reply"],
        )

        return InvokeResponse(
            session_id=req.session_id,
            reply=result["reply"],
            tool_calls=result.get("tool_calls", []),
        )

    except Exception as e:
        logger.exception(f"[invoke] unhandled error for session={req.session_id}")
        return InvokeResponse(
            session_id=req.session_id,
            reply="⚠️ An internal error occurred. Please try again.",
            error=str(e),
        )


@app.post("/oauth/refresh")
async def force_refresh(user_id: str, dependencies=[Depends(verify_n8n_secret)]):
    """Manual trigger to force-refresh an OAuth token."""
    token = await oauth_manager.get_valid_token(user_id)
    return {"access_token_preview": token[:12] + "...", "user_id": user_id}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

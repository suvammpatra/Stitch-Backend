"""
Google Stitch LangGraph Backend
Production-grade FastAPI + LangGraph service
Includes OAuth2 flow for Google Stitch authorization
"""

import logging
import os
import urllib.parse
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from graph import StitchGraph
from oauth_manager import OAuthManager
from session_store import SessionStore

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stitch.main")

# ─── Globals ─────────────────────────────────────────────────────────────────
redis_client: aioredis.Redis | None = None
oauth_manager: OAuthManager | None = None
session_store: SessionStore | None = None
stitch_graph: StitchGraph | None = None

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

# Scopes required by Google Stitch MCP
# Note: auth/stitch does not exist — cloud-platform covers all Google APIs
STITCH_SCOPES = "https://www.googleapis.com/auth/cloud-platform"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, oauth_manager, session_store, stitch_graph

    logger.info("Starting up Stitch backend...")

    redis_url = os.environ["REDIS_URL"]
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
BACKEND_URL = os.getenv("BACKEND_URL", "https://stitch-backend-production.up.railway.app")


# ─── Auth guard ──────────────────────────────────────────────────────────────
def verify_n8n_secret(x_n8n_secret: str = Header(default="")):
    if N8N_SECRET and x_n8n_secret != N8N_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")


# ─── Models ──────────────────────────────────────────────────────────────────
class InvokeRequest(BaseModel):
    session_id: str
    message: str
    telegram_chat_id: str


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


# ─── OAuth2 Flow ─────────────────────────────────────────────────────────────

@app.get("/oauth/login")
async def oauth_login(user_id: str = Query(..., description="Your Telegram user ID")):
    """
    Step 1: Visit this URL in your browser to authorize Google Stitch.
    Example: https://your-backend.railway.app/oauth/login?user_id=123456789
    """
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    if not client_id:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID not configured")

    redirect_uri = f"{BACKEND_URL}/oauth/callback"

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": STITCH_SCOPES,
        "access_type": "offline",       # Gets refresh_token
        "prompt": "consent",            # Forces refresh_token every time
        "state": user_id,               # Pass Telegram user_id through OAuth flow
    }

    url = GOOGLE_AUTH_URL + "?" + urllib.parse.urlencode(params)
    logger.info(f"[oauth] Redirecting user_id={user_id} to Google consent")
    return RedirectResponse(url=url)


@app.get("/oauth/callback")
async def oauth_callback(
    code: str = Query(None),
    state: str = Query(None),
    error: str = Query(None),
):
    """
    Step 2: Google redirects here after user consents.
    Exchanges code for access + refresh tokens and stores them in Redis.
    """
    if error:
        logger.error(f"[oauth] Google returned error: {error}")
        return HTMLResponse(f"""
            <h2>❌ Authorization failed</h2>
            <p>Google returned: <code>{error}</code></p>
            <p>Please try again: <a href="/oauth/login?user_id={state}">Retry</a></p>
        """, status_code=400)

    if not code or not state:
        return HTMLResponse("<h2>❌ Missing code or state</h2>", status_code=400)

    user_id = state
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    redirect_uri = f"{BACKEND_URL}/oauth/callback"

    # Exchange authorization code for tokens
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(GOOGLE_TOKEN_URL, data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            })

        if resp.status_code != 200:
            logger.error(f"[oauth] Token exchange failed: {resp.text}")
            return HTMLResponse(f"""
                <h2>❌ Token exchange failed</h2>
                <pre>{resp.text}</pre>
            """, status_code=400)

        token_data = resp.json()

        if "refresh_token" not in token_data:
            logger.warning(f"[oauth] No refresh_token in response for user={user_id}. "
                           "User may need to revoke access at myaccount.google.com/permissions and retry.")

        # Store tokens in Redis via OAuthManager
        await oauth_manager.store_token(user_id, token_data)
        logger.info(f"[oauth] Tokens stored for user_id={user_id}")

        return HTMLResponse(f"""
            <html>
            <body style="font-family:sans-serif;max-width:500px;margin:60px auto;text-align:center">
                <h2>✅ Authorization successful!</h2>
                <p>Google Stitch is now connected for your Telegram account.</p>
                <p>Your Telegram user ID: <code>{user_id}</code></p>
                <p>Go back to Telegram and send a <code>/stitch</code> command.</p>
                <p style="color:#888;font-size:12px">You can close this tab.</p>
            </body>
            </html>
        """)

    except Exception as e:
        logger.exception(f"[oauth] Callback error: {e}")
        return HTMLResponse(f"<h2>❌ Internal error</h2><pre>{e}</pre>", status_code=500)


@app.get("/oauth/status")
async def oauth_status(user_id: str = Query(...)):
    """Check if a user has a valid stored token."""
    token_raw = await redis_client.get(f"oauth:token:{user_id}")
    if not token_raw:
        return {
            "user_id": user_id,
            "authorized": False,
            "message": f"No token found. Visit /oauth/login?user_id={user_id} to authorize.",
        }
    import json, time
    data = json.loads(token_raw)
    expires_at = data.get("expires_at", 0)
    seconds_left = int(expires_at - time.time())
    return {
        "user_id": user_id,
        "authorized": True,
        "expires_in_seconds": seconds_left,
        "has_refresh_token": bool(data.get("refresh_token")),
    }


@app.delete("/oauth/revoke")
async def oauth_revoke(user_id: str = Query(...), dependencies=[Depends(verify_n8n_secret)]):
    """Remove stored tokens for a user."""
    await oauth_manager.invalidate_token(user_id)
    return {"user_id": user_id, "revoked": True}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

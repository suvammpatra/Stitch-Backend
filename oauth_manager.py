"""
OAuth Manager
Handles Google OAuth2 token lifecycle:
- Storage in Redis with TTL
- Auto-refresh before expiry
- Retry on 401 with backoff
- Per-user token isolation
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional

import httpx
import redis.asyncio as aioredis

logger = logging.getLogger("stitch.oauth")

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
STITCH_SCOPES = "https://www.googleapis.com/auth/stitch"
# Refresh 5 minutes before actual expiry to avoid race conditions
EXPIRY_BUFFER_SECONDS = 300


class TokenExpiredError(Exception):
    pass


class TokenRefreshError(Exception):
    pass


class OAuthManager:
    """
    Centralized OAuth2 token manager backed by Redis.

    Token structure stored per user:
    {
        "access_token": "...",
        "refresh_token": "...",
        "expires_at": <unix timestamp>,
        "token_type": "Bearer"
    }

    Redis keys:
        oauth:token:{user_id}     → JSON blob, TTL = token expiry
        oauth:lock:{user_id}      → distributed lock for refresh
    """

    def __init__(self, redis: aioredis.Redis):
        self.redis = redis
        self.client_id = os.environ["GOOGLE_CLIENT_ID"]
        self.client_secret = os.environ["GOOGLE_CLIENT_SECRET"]
        # Static token used as fallback (from n8n Bearer Auth) — service account or long-lived PAT
        self.static_token = os.getenv("GOOGLE_STITCH_STATIC_TOKEN", "")

    # ── Public interface ────────────────────────────────────────────────────

    async def get_valid_token(self, user_id: str) -> str:
        """
        Returns a valid access token for user_id.
        Refreshes automatically if within EXPIRY_BUFFER_SECONDS of expiry.
        Falls back to static token if no per-user token stored.
        """
        token_data = await self._load_token(user_id)

        if not token_data:
            if self.static_token:
                logger.info(f"[oauth] No per-user token for {user_id}, using static token")
                return self.static_token
            raise TokenExpiredError(f"No token found for user {user_id}")

        expires_at = token_data.get("expires_at", 0)
        now = time.time()

        if now >= expires_at - EXPIRY_BUFFER_SECONDS:
            logger.info(f"[oauth] Token for {user_id} near/past expiry, refreshing...")
            token_data = await self._refresh_with_lock(user_id, token_data)

        return token_data["access_token"]

    async def store_token(self, user_id: str, token_response: dict):
        """
        Store a new token from OAuth callback or manual injection.
        Called once after the OAuth2 consent flow completes.
        """
        expires_in = token_response.get("expires_in", 3600)
        token_data = {
            "access_token": token_response["access_token"],
            "refresh_token": token_response.get("refresh_token", ""),
            "expires_at": time.time() + expires_in,
            "token_type": token_response.get("token_type", "Bearer"),
        }
        await self._save_token(user_id, token_data, ttl=expires_in + 600)
        logger.info(f"[oauth] Stored new token for user {user_id}")

    async def invalidate_token(self, user_id: str):
        await self.redis.delete(f"oauth:token:{user_id}")
        logger.warning(f"[oauth] Invalidated token for user {user_id}")

    # ── Internal helpers ────────────────────────────────────────────────────

    async def _load_token(self, user_id: str) -> Optional[dict]:
        raw = await self.redis.get(f"oauth:token:{user_id}")
        if raw:
            return json.loads(raw)
        return None

    async def _save_token(self, user_id: str, token_data: dict, ttl: int = 4200):
        await self.redis.setex(
            f"oauth:token:{user_id}",
            ttl,
            json.dumps(token_data),
        )

    async def _refresh_with_lock(self, user_id: str, stale_token: dict) -> dict:
        """
        Distributed lock prevents multiple concurrent refreshes for same user.
        If lock is held, wait and re-read from Redis (another worker refreshed).
        """
        lock_key = f"oauth:lock:{user_id}"
        acquired = await self.redis.set(lock_key, "1", nx=True, ex=30)

        if not acquired:
            # Another worker is refreshing — wait and re-read
            for _ in range(10):
                await asyncio.sleep(0.5)
                token_data = await self._load_token(user_id)
                if token_data and time.time() < token_data.get("expires_at", 0) - EXPIRY_BUFFER_SECONDS:
                    return token_data
            raise TokenRefreshError(f"Timeout waiting for token refresh lock for {user_id}")

        try:
            return await self._do_refresh(user_id, stale_token)
        finally:
            await self.redis.delete(lock_key)

    async def _do_refresh(self, user_id: str, stale_token: dict) -> dict:
        refresh_token = stale_token.get("refresh_token")
        if not refresh_token:
            raise TokenRefreshError(f"No refresh_token for user {user_id}")

        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(GOOGLE_TOKEN_URL, data=payload)

                if resp.status_code == 200:
                    data = resp.json()
                    # Preserve old refresh_token if new one not returned
                    if "refresh_token" not in data:
                        data["refresh_token"] = refresh_token

                    expires_in = data.get("expires_in", 3600)
                    token_data = {
                        "access_token": data["access_token"],
                        "refresh_token": data["refresh_token"],
                        "expires_at": time.time() + expires_in,
                        "token_type": data.get("token_type", "Bearer"),
                    }
                    await self._save_token(user_id, token_data, ttl=expires_in + 600)
                    logger.info(f"[oauth] Successfully refreshed token for {user_id}")
                    return token_data

                elif resp.status_code in (400, 401):
                    logger.error(f"[oauth] Refresh failed for {user_id}: {resp.text}")
                    await self.invalidate_token(user_id)
                    raise TokenRefreshError(f"Refresh rejected: {resp.text}")

                else:
                    wait = 2 ** attempt
                    logger.warning(f"[oauth] Refresh attempt {attempt+1} got {resp.status_code}, retrying in {wait}s")
                    await asyncio.sleep(wait)

            except httpx.RequestError as e:
                logger.warning(f"[oauth] Network error on refresh attempt {attempt+1}: {e}")
                await asyncio.sleep(2 ** attempt)

        raise TokenRefreshError(f"Exhausted refresh retries for {user_id}")

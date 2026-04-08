"""
Session Store
Redis-backed multi-turn conversation memory.
- Per-user conversation history (FIFO, capped)
- TTL-based session expiry (24h default)
- Serialised as JSON list of {role, content} dicts
"""

import json
import logging
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger("stitch.session")

SESSION_TTL_SECONDS = 86400   # 24 hours
MAX_TURNS = 20                # Max stored turns (user+assistant pairs)


class SessionStore:
    def __init__(self, redis: aioredis.Redis):
        self.redis = redis

    def _key(self, session_id: str) -> str:
        return f"session:history:{session_id}"

    async def get_history(self, session_id: str) -> list[dict]:
        """Returns list of {role, content} dicts, oldest first."""
        raw = await self.redis.get(self._key(session_id))
        if not raw:
            return []
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"[session] Corrupt history for {session_id}, resetting")
            await self.clear(session_id)
            return []

    async def append_turn(self, session_id: str, user: str, assistant: str):
        """Append a user+assistant exchange and reset TTL."""
        history = await self.get_history(session_id)
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": assistant})

        # Cap history to MAX_TURNS pairs (2 messages per turn)
        max_messages = MAX_TURNS * 2
        if len(history) > max_messages:
            history = history[-max_messages:]

        await self.redis.setex(
            self._key(session_id),
            SESSION_TTL_SECONDS,
            json.dumps(history),
        )

    async def clear(self, session_id: str):
        await self.redis.delete(self._key(session_id))
        logger.info(f"[session] Cleared history for {session_id}")

    async def get_all_active_sessions(self) -> list[str]:
        """For admin/monitoring use only."""
        keys = await self.redis.keys("session:history:*")
        return [k.replace("session:history:", "") for k in keys]

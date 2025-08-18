import json
from typing import Any, Dict, Optional

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

from ..core.config import Config
from ..core.models import RouterState


class StateStore:
    """Externalized state management with Redis; falls back to in-memory if Redis unavailable."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._mem: Dict[str, Dict[str, Any]] = {}
        if cfg.redis_url and redis is not None:
            self.client = redis.from_url(cfg.redis_url, decode_responses=True)
        else:
            self.client = None

    def _key(self, run_id: str) -> str:
        return f"research:{run_id}:state"

    async def load(self, run_id: str) -> Optional[RouterState]:
        if not self.client:
            data = self._mem.get(run_id)
            if not data:
                return None
            return RouterState.model_validate(data)
        s = await self.client.get(self._key(run_id))
        return RouterState.model_validate(json.loads(s)) if s else None

    async def save_cas(self, state: RouterState, expected_version: int) -> bool:
        # optimistic concurrency: only write if version matches
        state_dict = state.model_dump(mode="json")
        state_dict["version"] = expected_version + 1
        payload = json.dumps(state_dict)

        if not self.client:
            cur = self._mem.get(state.run_id)
            cur_ver = cur.get("version", -1) if cur else -1
            if cur is None or cur_ver == expected_version:
                self._mem[state.run_id] = state_dict
                return True
            return False

        key = self._key(state.run_id)
        async with self.client.pipeline() as pipe:
            success = False
            for _ in range(5):
                try:
                    await pipe.watch(key)
                    cur = await pipe.get(key)
                    cur_ver = -1
                    if cur:
                        cur_ver = json.loads(cur).get("version", -1)
                    if cur is None or cur_ver == expected_version:
                        pipe.multi()
                        pipe.set(key, payload, ex=self.cfg.state_ttl_seconds)
                        await pipe.execute()
                        success = True
                        break
                    await pipe.unwatch()
                    break
                except redis.WatchError:  # type: ignore
                    continue
            return success

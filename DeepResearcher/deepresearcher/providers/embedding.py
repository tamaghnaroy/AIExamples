import os
import asyncio
import numpy as np
from openai import AsyncOpenAI
import structlog

from ..core.models import RouterState
from ..utils.utils import hash_key

log = structlog.get_logger(__name__)

class EmbeddingsProvider:
    def __init__(self, client: AsyncOpenAI, model: str, cache_dir: str, budget_state: RouterState, run_id: str):
        self.client = client
        self.model = model
        self.dir = os.path.join(cache_dir, "emb")
        os.makedirs(self.dir, exist_ok=True)
        self._lock = asyncio.Lock()
        self.budget_state = budget_state
        self.run_id = run_id

    def _path(self, key: str) -> str:
        return os.path.join(self.dir, f"{hash_key(key)}.npy")

    async def embed(self, text: str) -> np.ndarray:
        path = self._path(text)
        async with self._lock:
            if os.path.exists(path):
                return np.load(path)
        if self.budget_state.token_budget_remaining <= 0:
            raise RuntimeError("Token budget exhausted before embedding call.")
        resp = await self.client.embeddings.create(input=[text], model=self.model)
        # TODO: Add budget tracking for embeddings
        vec = np.array(resp.data[0].embedding)
        async with self._lock:
            np.save(path, vec)
        return vec

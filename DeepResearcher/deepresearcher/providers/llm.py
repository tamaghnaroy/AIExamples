import time
import json
from typing import Optional, Dict

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..core.models import RouterState
from ..core.config import Config, estimate_cost

log = structlog.get_logger(__name__)

class LLM:
    def __init__(self, client: AsyncOpenAI, model: str, temperature: float, budget_state: RouterState, cfg: Config, run_id: str):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.budget_state = budget_state
        self.cfg = cfg
        self.run_id = run_id

    def _apply_budget(self, usage: Dict[str, int]) -> None:
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        total = pt + ct
        cost = estimate_cost(self.model, pt, ct)
        self.budget_state.tokens_used += total
        self.budget_state.cost_used += cost
        self.budget_state.token_budget_remaining = max(0, self.budget_state.token_budget_remaining - total)
        self.budget_state.cost_budget_remaining = max(0.0, self.budget_state.cost_budget_remaining - cost)

    def _deadline_exceeded(self) -> bool:
        return time.time() >= self.budget_state.time_deadline_ts

    def out_of_budget(self) -> bool:
        return (
            self.budget_state.token_budget_remaining <= 0 or
            self.budget_state.cost_budget_remaining <= 0.0 or
            self._deadline_exceeded()
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(1, 1, 10),
           retry=retry_if_exception_type(Exception), reraise=True)
    async def json_schema_complete(self, system: str, user: str, schema: type[BaseModel],
                                   temperature: Optional[float]=None) -> BaseModel:
        if self.out_of_budget():
            raise RuntimeError("Budget exhausted before LLM call.")
        messages = [{"role":"system","content":system},{"role":"user","content":user}]
        t0 = time.time()
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature if temperature is None else temperature,
            response_format={"type": "json_object"},
            messages=messages,
        )
        latency = time.time() - t0
        usage = (getattr(resp, "usage", None) or {}).__dict__ if hasattr(resp, "usage") else {}
        # `resp.usage` in openai>=1.x has attributes: prompt_tokens, completion_tokens
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) if hasattr(resp, "usage") else 0,
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0) if hasattr(resp, "usage") else 0
        }
        self._apply_budget(usage)

        raw = resp.choices[0].message.content
        try:
            data = json.loads(raw)
            try:
                out = schema.model_validate(data)
            except ValidationError as e:
                log.bind(event="llm_parse_error", run_id=self.run_id, errors=e.errors()).error("pydantic_validation_failed")
                raise
            log.bind(event="llm_call", run_id=self.run_id, model=self.model,
                     prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"],
                     latency_s=round(latency,3), cost_used=round(self.budget_state.cost_used,4)).info("llm.json_schema_complete")
            return out
        except json.JSONDecodeError as e:
            log.bind(event="llm_parse_error", run_id=self.run_id).error("non_json_output")
            raise e

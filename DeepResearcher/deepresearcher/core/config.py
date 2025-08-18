import os
import argparse
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# ==============================================================================
# 1) Config & Pricing (v3.7 adds budgets)
# ==============================================================================

load_dotenv()

def _require(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise ValueError(f"Missing required environment variable: {name}")
    return v

@dataclass
class Config:
    # Core keys
    openai_api_key: str = field(default_factory=lambda: _require("OPENAI_API_KEY"), repr=False)
    tavily_api_key: str = field(default_factory=lambda: _require("TAVILY_API_KEY"), repr=False)
    firecrawl_api_key: str = field(default_factory=lambda: _require("FIRECRAWL_API_KEY"), repr=False)

    # LLM + Embeddings
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    # Retrieval & loop
    max_questions: int = int(os.getenv("RESEARCH_MAX_QUESTIONS", "4"))
    iter_depth: int = int(os.getenv("RESEARCH_ITER_DEPTH", "6"))

    # Concurrency & timeouts
    max_concurrency: int = int(os.getenv("RESEARCH_MAX_CONCURRENCY", "5"))
    request_timeout_s: int = int(os.getenv("RESEARCH_TIMEOUT_S", "25"))

    # Token budgets
    max_tokens_per_source: int = int(os.getenv("RESEARCH_TOKENS_PER_SOURCE", "2500"))

    # BM25 params
    bm25_k1: float = float(os.getenv("RESEARCH_BM25_K1", "1.5"))
    bm25_b: float = float(os.getenv("RESEARCH_BM25_B", "0.75"))
    rrf_k: int = int(os.getenv("RESEARCH_RRF_K", "60"))

    # Caching
    cache_dir: str = os.getenv("RESEARCH_CACHE_DIR", ".cache/research")

    # Confidence learning
    conf_eps: float = float(os.getenv("RESEARCH_CONF_EPS", "0.01"))
    conf_alpha: float = float(os.getenv("RESEARCH_CONF_ALPHA", "0.2"))

    # v3.4: Verification/Critique policy
    verification_ttl_days: int = int(os.getenv("RESEARCH_VERIF_TTL_DAYS", "7"))
    require_verification_before_end: bool = os.getenv("RESEARCH_REQUIRE_VERIF", "1") == "1"
    require_critique_before_end: bool = os.getenv("RESEARCH_REQUIRE_CRITIQUE", "1") == "1"

    # v3.5: Strategy Review controls
    review_every_steps: int = int(os.getenv("RESEARCH_REVIEW_EVERY_STEPS", "3"))
    review_stagnation_tau: float = float(os.getenv("RESEARCH_REVIEW_STAGNATION_TAU", "0.8"))
    task_similarity_threshold: float = float(os.getenv("RESEARCH_TASK_SIM_THRESHOLD", "0.85"))
    max_prunes_per_review: int = int(os.getenv("RESEARCH_MAX_PRUNES_PER_REVIEW", "2"))
    min_steps_between_prunes: int = int(os.getenv("RESEARCH_MIN_STEPS_BETWEEN_PRUNES", "2"))
    max_reorder_jump: int = int(os.getenv("RESEARCH_MAX_REORDER_JUMP", "2"))

    # v3.6: Debate & paragraph confidence
    debate_top_k: int = int(os.getenv("RESEARCH_DEBATE_TOP_K", "6"))
    debate_min_independent_domains: int = int(os.getenv("RESEARCH_DEBATE_MIN_DOMAINS", "3"))
    debate_temperature_blue: float = float(os.getenv("RESEARCH_DEBATE_TEMP_BLUE", "0.4"))
    debate_temperature_red: float = float(os.getenv("RESEARCH_DEBATE_TEMP_RED", "0.8"))
    debate_temperature_mod: float = float(os.getenv("RESEARCH_DEBATE_TEMP_MOD", "0.2"))
    synthesis_max_evidence_urls_per_para: int = int(os.getenv("RESEARCH_SYNTH_MAX_URLS_PER_PARA", "5"))

    # v3.7: Budgets & observability
    token_budget_total: int = int(os.getenv("RESEARCH_TOKEN_BUDGET", "120000"))   # rough tokens cap for run
    cost_budget_total: float = float(os.getenv("RESEARCH_COST_BUDGET", "8.00"))   # USD
    time_budget_seconds: int = int(os.getenv("RESEARCH_TIME_BUDGET_S", "600"))    # 10 minutes
    log_level: str = os.getenv("RESEARCH_LOG_LEVEL", "INFO")

    # v3.7: State / Redis
    redis_url: Optional[str] = os.getenv("REDIS_URL")
    state_ttl_seconds: int = int(os.getenv("RESEARCH_STATE_TTL_S", "86400"))
    state_schema_version: str = "1"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Research Orchestrator v3.7 (budgets, logging, externalized state)")
    p.add_argument("topic", nargs="?", type=str, help="Primary research topic (interactive prompt if omitted).")
    p.add_argument("--output", type=str, default="research_result.json", help="Output JSON file path.")
    p.add_argument("--model", type=str, default=os.getenv("LLM_MODEL", "gpt-4o"), help="OpenAI chat model.")
    p.add_argument("--temp", type=float, default=float(os.getenv("LLM_TEMPERATURE", "0.0")), help="LLM temperature.")
    p.add_argument("--embed-model", type=str, default=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
                   help="Embedding model for semantic search and task similarity.")
    p.add_argument("--iter-depth", type=int, default=int(os.getenv("RESEARCH_ITER_DEPTH", "6")),
                   help="Max router iterations before budget stop.")
    p.add_argument("--max-questions", type=int, default=int(os.getenv("RESEARCH_MAX_QUESTIONS", "4")),
                   help="Number of questions for initial planning.")
    p.add_argument("--cache-dir", type=str, default=os.getenv("RESEARCH_CACHE_DIR", ".cache/research"),
                   help="Cache directory.")
    p.add_argument("--clear-cache", action="store_true", help="Clear cache before running.")
    p.add_argument("--timeout", type=int, default=int(os.getenv("RESEARCH_TIMEOUT_S", "25")),
                   help="Per-request timeout in seconds.")
    p.add_argument("--max-concurrency", type=int, default=int(os.getenv("RESEARCH_MAX_CONCURRENCY", "5")),
                   help="Max concurrent scrapes/summaries/extractions.")
    p.add_argument("--tokens-per-source", type=int, default=int(os.getenv("RESEARCH_TOKENS_PER_SOURCE", "2500")),
                   help="Approx token budget per source snippet.")
    p.add_argument("--rrf-k", type=int, default=int(os.getenv("RESEARCH_RRF_K", "60")), help="RRF fusion constant k.")
    p.add_argument("--bm25-k1", type=float, default=float(os.getenv("RESEARCH_BM25_K1", "1.5")), help="BM25 k1.")
    p.add_argument("--bm25-b", type=float, default=float(os.getenv("RESEARCH_BM25_B", "0.75")), help="BM25 b.")
    p.add_argument("--token-budget", type=int, default=int(os.getenv("RESEARCH_TOKEN_BUDGET", "120000")),
                   help="Total token budget for the run.")
    p.add_argument("--cost-budget", type=float, default=float(os.getenv("RESEARCH_COST_BUDGET", "8.00")),
                   help="Total cost budget (USD) for the run.")
    p.add_argument("--time-budget", type=int, default=int(os.getenv("RESEARCH_TIME_BUDGET_S", "600")),
                   help="Total wall-clock time budget (seconds) for the run.")
    p.add_argument("--log-level", type=str, default=os.getenv("RESEARCH_LOG_LEVEL", "INFO"), help="Log level.")
    p.add_argument("--run-id", type=str, default="", help="Resume an existing run_id (if state exists).")
    return p.parse_args()

# Pricing map (USD per 1K tokens) — v3.7 approximate, update as needed
PRICING_MAP = {
    # Chat
    "gpt-4o": {"input": 0.005, "output": 0.015},
    # Embeddings
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
}

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICING_MAP.get(model, {"input": 0.0, "output": 0.0})
    return (prompt_tokens / 1000.0) * p["input"] + (completion_tokens / 1000.0) * p["output"]

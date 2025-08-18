# deepresearcher/cli/main.py

import os
import sys
import json
import time
import uuid
import shutil
import asyncio
import argparse
from typing import Any, Dict, List, Callable, Awaitable

import structlog
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tavily import TavilyClient
from firecrawl import FirecrawlApp

from deepresearcher.core.config import Config
from deepresearcher.core.models import (
    Hypothesis, Synthesis, RouterState, ToolSpec
)
from deepresearcher.providers.llm import LLM
from deepresearcher.providers.embedding import EmbeddingsProvider
from deepresearcher.providers.search import Searcher
from deepresearcher.providers.scraper import Scraper
from deepresearcher.agents.topic_exploration import TopicExplorationAgent
from deepresearcher.agents.search_and_crawl import SearchAndCrawlAgent
from deepresearcher.agents.summarizer import SummarizerAgent
from deepresearcher.agents.analysis import AnalysisAgent
from deepresearcher.agents.synthesis import SynthesisAgent
from deepresearcher.agents.refinement import RefinementAgent
from deepresearcher.tools.tools import Tools
from deepresearcher.storage.state_store import StateStore
from deepresearcher.orchestration.orchestrator import Orchestrator, RouterAgent

load_dotenv()

log = structlog.get_logger(__name__)

def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    structlog.configure(
        processors=[
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(sys.modules['logging'], level.upper(), sys.modules['logging'].INFO)),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

def _require(name: str) -> str:
    """Get a required environment variable."""
    val = os.getenv(name)
    if val is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return val

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Deep Research Agent")
    parser.add_argument("-t", "--topic", type=str, help="Research topic")
    parser.add_argument("--run-id", type=str, help="Resume from a specific run ID")
    parser.add_argument("--output", type=str, default="output.json", help="Output file path")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model to use")
    parser.add_argument("--temp", type=float, default=0.4, help="LLM temperature")
    parser.add_argument("--embed-model", type=str, default="text-embedding-3-large", help="Embeddings model")
    parser.add_argument("--iter-depth", type=int, default=5, help="Max iterations")
    parser.add_argument("--max-questions", type=int, default=5, help="Max questions for topic exploration")
    parser.add_argument("--cache-dir", type=str, default="./.cache", help="Cache directory")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--max-concurrency", type=int, default=5, help="Max concurrent requests")
    parser.add_argument("--tokens-per-source", type=int, default=4096, help="Max tokens per scraped source")
    parser.add_argument("--rrf-k", type=int, default=20, help="RRF K value for ranking")
    parser.add_argument("--bm25-k1", type=float, default=1.5, help="BM25 k1 parameter")
    parser.add_argument("--bm25-b", type=float, default=0.75, help="BM25 b parameter")
    parser.add_argument("--token-budget", type=int, default=1000000, help="Total token budget")
    parser.add_argument("--cost-budget", type=float, default=2.0, help="Total cost budget in USD")
    parser.add_argument("--time-budget", type=int, default=3600, help="Total time budget in seconds")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the cache before running")
    return parser.parse_args()

async def main() -> None:
    args = _parse_args()
    setup_logging(args.log_level)

    cfg = Config(
        openai_api_key=_require("OPENAI_API_KEY"),
        tavily_api_key=_require("TAVILY_API_KEY"),
        firecrawl_api_key=_require("FIRECRAWL_API_KEY"),
        llm_model=args.model,
        temperature=args.temp,
        embed_model=args.embed_model,
        iter_depth=args.iter_depth,
        max_questions=args.max_questions,
        cache_dir=args.cache_dir,
        request_timeout_s=args.timeout,
        max_concurrency=args.max_concurrency,
        max_tokens_per_source=args.tokens_per_source,
        rrf_k=args.rrf_k,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        token_budget_total=args.token_budget,
        cost_budget_total=args.cost_budget,
        time_budget_seconds=args.time_budget,
        log_level=args.log_level,
    )

    if args.clear_cache and os.path.isdir(cfg.cache_dir):
        log.bind(event="cache_clear").info("cache.clearing")
        shutil.rmtree(cfg.cache_dir, ignore_errors=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)

    topic = args.topic
    if not topic:
        try:
            topic = input("Please enter the research topic: ").strip()
        except EOFError:
            topic = ""
    if not topic:
        log.bind(event="error").error("no_topic")
        raise SystemExit(2)

    run_id = args.run_id or str(uuid.uuid4())
    log.bind(event="run_alloc", run_id=run_id).info("run.allocated")

    # External state store
    store = StateStore(cfg)
    resume_state = await store.load(run_id)

    # Clients
    openai_client = AsyncOpenAI(api_key=cfg.openai_api_key)
    tavily_client = TavilyClient(api_key=cfg.tavily_api_key)
    firecrawl_client = FirecrawlApp(api_key=cfg.firecrawl_api_key)

    if resume_state is None:
        now = time.time()
        tmp_state = RouterState(
            run_id=run_id,
            state_schema_version=cfg.state_schema_version,
            topic=topic,
            version=0,
            hypothesis=Hypothesis(statement="", confidence=0.5),
            synthesis=Synthesis(text="", rationale=[]),
            step=0,
            conf_history=[0.5],
            token_budget_remaining=cfg.token_budget_total,
            cost_budget_remaining=cfg.cost_budget_total,
            time_deadline_ts=now + cfg.time_budget_seconds,
        )
        budget_state = tmp_state
    else:
        budget_state = resume_state

    sem = asyncio.Semaphore(cfg.max_concurrency)

    # Providers & Agents
    llm = LLM(openai_client, cfg.llm_model, cfg.temperature, budget_state, cfg, run_id)
    embed = EmbeddingsProvider(openai_client, cfg.embed_model, cfg.cache_dir, budget_state, run_id)
    searcher = Searcher(tavily_client, run_id=run_id)
    scraper = Scraper(firecrawl_client, cache_dir=os.path.join(cfg.cache_dir, "pages"), run_id=run_id)

    tea = TopicExplorationAgent(llm, max_questions=cfg.max_questions)
    gather = SearchAndCrawlAgent(searcher, scraper, embed, sem, cfg.request_timeout_s, cfg, run_id)
    summarize = SummarizerAgent(llm, cfg.max_tokens_per_source, run_id)
    analyze = AnalysisAgent(llm, run_id)
    synth = SynthesisAgent(llm, run_id)
    refine = RefinementAgent(conf_alpha=cfg.conf_alpha)

    tools_obj = Tools(gather, summarize, analyze, synth, refine, sem, cfg, embed, llm, run_id)

    tool_registry: Dict[str, Callable[..., Awaitable[Any]]] = {
        "search_and_crawl_tool": tools_obj.search_and_crawl_tool,
        "analyze_content_tool": tools_obj.analyze_content_tool,
        "synthesize_evidence_tool": tools_obj.synthesize_evidence_tool,
        "update_hypothesis_tool": tools_obj.update_hypothesis_tool,
        "calculator_tool": tools_obj.calculator_tool,
        "end_research_tool": tools_obj.end_research_tool,
        "add_sub_task_tool": tools_obj.add_sub_task_tool,
        "complete_current_task_tool": tools_obj.complete_current_task_tool,
        "verify_claims_tool": tools_obj.verify_claims_tool,
        "critique_evidence_tool": tools_obj.critique_evidence_tool,
        "review_and_prune_plan_tool": tools_obj.review_and_prune_plan_tool,
        "conduct_adversarial_review_tool": tools_obj.conduct_adversarial_review_tool,
    }

    toolspecs = [
        ToolSpec(name="search_and_crawl_tool", description="Search the web and scrape relevant pages for a query.",
                 arg_schema={"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
        ToolSpec(name="analyze_content_tool", description="Analyze provided text to extract claims/stats/entities.",
                 arg_schema={"type":"object","properties":{"content":{"type":"string"}},"required":["content"]}),
        ToolSpec(name="synthesize_evidence_tool", description="Write a synthesized answer from evidence + hypothesis.",
                 arg_schema={"type":"object","properties":{"evidence":{"type":"array"},"hypothesis":{"type":"object"}},"required":["evidence","hypothesis"]}),
        ToolSpec(name="update_hypothesis_tool", description="Update hypothesis confidence from current evidence.",
                 arg_schema={"type":"object","properties":{"evidence":{"type":"array"},"hypothesis":{"type":"object"}},"required":["evidence","hypothesis"]}),
        ToolSpec(name="calculator_tool", description="Evaluate a simple arithmetic expression.",
                 arg_schema={"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}),
        ToolSpec(name="end_research_tool", description="End the session with a final conclusion.",
                 arg_schema={"type":"object","properties":{"final_conclusion":{"type":"string"}},"required":["final_conclusion"]}),
        ToolSpec(name="add_sub_task_tool", description="Push a new, more specific sub-task onto the top of the task list.",
                 arg_schema={"type":"object","properties":{"task_description":{"type":"string"}},"required":["task_description"]}),
        ToolSpec(name="complete_current_task_tool", description="Mark the current top task as completed with a short summary.",
                 arg_schema={"type":"object","properties":{"summary_of_completion":{"type":"string"}},"required":["summary_of_completion"]}),
        ToolSpec(name="verify_claims_tool", description="Verify a list of claims via targeted search & weighting.",
                 arg_schema={"type":"object","properties":{"claims":{"type":"array","items":{"type":"string"}}},"required":["claims"]}),
        ToolSpec(name="critique_evidence_tool", description="Critique current evidence for bias, contradictions, staleness.",
                 arg_schema={"type":"object","properties":{"evidence":{"type":"array"},"verified":{"type":"array"}},"required":["evidence"]}),
        ToolSpec(name="review_and_prune_plan_tool", description="Strategically review the task list, prune redundant tasks, and propose stable priority changes with justification.",
                 arg_schema={"type":"object","properties":{}}),
        ToolSpec(name="conduct_adversarial_review_tool", description="Run Blue/Red debate and produce a Moderator FinalSynthesis with paragraph-level confidence.",
                 arg_schema={"type":"object","properties":{"evidence":{"type":"array"},"verified_claims":{"type":"array"},"current_synthesis":{"type":"object"},"hypothesis":{"type":"object"},"critique":{"type":"object"}},"required":["evidence","verified_claims","current_synthesis","hypothesis"]}),
    ]
    router = RouterAgent(llm, toolspecs, run_id)

    orch = Orchestrator(tea, gather, summarize, analyze, synth, refine, router, tool_registry, sem, cfg, tools_obj, store, run_id)

    result = await orch.run(topic, resume_state=resume_state)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log.bind(event="saved", run_id=run_id, path=args.output).info("output.saved")

def entry_point():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.bind(event="interrupt").warning("user.interrupt")
        raise SystemExit(130)
    except (ValueError, FileNotFoundError) as e:
        log.bind(event="fatal", err=str(e)).critical("config.error")
        raise SystemExit(1)
    except Exception as e:
        log.bind(event="fatal", err=str(e)).critical("fatal.error", exc_info=True)
        raise SystemExit(1)

if __name__ == "__main__":
    entry_point()

"""
DeepResearchClient - Class-based interface for programmatic access to Deep Research system.

This module provides a clean, easy-to-use interface for integrating Deep Research
capabilities into other Python applications.
"""

import asyncio
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import structlog
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tavily import TavilyClient
from firecrawl import FirecrawlApp

from deepresearcher.core.config import Config
from deepresearcher.core.models import (
    Hypothesis, Synthesis, RouterState, ToolSpec, ResearchResult
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

@dataclass
class ResearchConfig:
    """Configuration for research parameters."""
    # API Keys (optional - will use environment variables if not provided)
    openai_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    firecrawl_api_key: Optional[str] = None
    
    # Model settings
    llm_model: str = "gpt-4o"
    temperature: float = 0.4
    embed_model: str = "text-embedding-3-small"
    
    # Research behavior
    max_iterations: int = 6
    max_questions: int = 4
    
    # Budget controls
    token_budget: int = 120000
    cost_budget: float = 8.0
    time_budget_seconds: int = 600
    
    # Performance
    max_concurrency: int = 5
    request_timeout: int = 25
    tokens_per_source: int = 2500
    
    # Caching
    cache_dir: str = ".cache/research"
    clear_cache: bool = False
    
    # Logging
    log_level: str = "INFO"

@dataclass
class ResearchProgress:
    """Progress information for ongoing research."""
    run_id: str
    step: int
    max_steps: int
    hypothesis: str
    confidence: float
    synthesis_preview: str
    tokens_used: int
    cost_used: float
    time_elapsed: float
    current_task: Optional[str] = None
    completed_tasks: int = 0
    remaining_tasks: int = 0

@dataclass
class ResearchSynthesis:
    """Final research synthesis with metadata."""
    topic: str
    hypothesis: str
    confidence: float
    synthesis_text: str
    future_questions: List[str]
    meta: Dict[str, Any]
    run_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "topic": self.topic,
            "hypothesis": self.hypothesis,
            "confidence": self.confidence,
            "synthesis": self.synthesis_text,
            "future_questions": self.future_questions,
            "meta": self.meta,
            "run_id": self.run_id
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save synthesis to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

class DeepResearchClient:
    """
    Class-based interface for Deep Research system.
    
    Provides both synchronous and asynchronous methods for conducting AI-powered research.
    Can be easily integrated into other Python applications.
    
    Example:
        # Basic usage
        client = DeepResearchClient()
        result = client.research("AI safety in autonomous vehicles")
        
        # Advanced usage with custom config
        config = ResearchConfig(
            max_iterations=10,
            token_budget=200000,
            cost_budget=15.0
        )
        client = DeepResearchClient(config)
        result = client.research("Climate change mitigation strategies")
        
        # Async usage
        result = await client.research_async("Quantum computing applications")
        
        # With progress callback
        def on_progress(progress: ResearchProgress):
            print(f"Step {progress.step}/{progress.max_steps}: {progress.current_task}")
        
        result = client.research("Machine learning ethics", progress_callback=on_progress)
    """
    
    def __init__(self, config: Optional[ResearchConfig] = None):
        """
        Initialize the Deep Research client.
        
        Args:
            config: Optional research configuration. If None, uses defaults from environment.
        """
        self.config = config or ResearchConfig()
        self._setup_logging()
        self._validate_environment()
        self._executor = ThreadPoolExecutor(max_workers=1)
        
    def _setup_logging(self) -> None:
        """Configure structured logging."""
        timestamper = structlog.processors.TimeStamper(fmt="iso")
        structlog.configure(
            processors=[
                timestamper,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(structlog.stdlib, self.config.log_level.upper(), structlog.stdlib.INFO)
            ),
            context_class=dict,
            cache_logger_on_first_use=True,
        )
        self.log = structlog.get_logger("research_client")
        
    def _validate_environment(self) -> None:
        """Validate required environment variables."""
        required_vars = {
            "OPENAI_API_KEY": self.config.openai_api_key,
            "TAVILY_API_KEY": self.config.tavily_api_key,
            "FIRECRAWL_API_KEY": self.config.firecrawl_api_key
        }
        
        missing = []
        for var_name, config_value in required_vars.items():
            if not config_value and not os.getenv(var_name):
                missing.append(var_name)
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Please set them in your .env file or pass them in ResearchConfig."
            )
    
    def _get_api_key(self, env_var: str, config_value: Optional[str]) -> str:
        """Get API key from config or environment."""
        return config_value or os.getenv(env_var)
    
    async def _create_orchestrator(self, run_id: str) -> Orchestrator:
        """Create and configure the orchestrator with all dependencies."""
        # Create Config object
        cfg = Config(
            openai_api_key=self._get_api_key("OPENAI_API_KEY", self.config.openai_api_key),
            tavily_api_key=self._get_api_key("TAVILY_API_KEY", self.config.tavily_api_key),
            firecrawl_api_key=self._get_api_key("FIRECRAWL_API_KEY", self.config.firecrawl_api_key),
            llm_model=self.config.llm_model,
            temperature=self.config.temperature,
            embed_model=self.config.embed_model,
            iter_depth=self.config.max_iterations,
            max_questions=self.config.max_questions,
            cache_dir=self.config.cache_dir,
            request_timeout_s=self.config.request_timeout,
            max_concurrency=self.config.max_concurrency,
            max_tokens_per_source=self.config.tokens_per_source,
            token_budget_total=self.config.token_budget,
            cost_budget_total=self.config.cost_budget,
            time_budget_seconds=self.config.time_budget_seconds,
            log_level=self.config.log_level,
        )
        
        # Clear cache if requested
        if self.config.clear_cache and os.path.isdir(cfg.cache_dir):
            import shutil
            shutil.rmtree(cfg.cache_dir, ignore_errors=True)
        os.makedirs(cfg.cache_dir, exist_ok=True)
        
        # External state store
        store = StateStore(cfg)
        
        # Create initial state for budget tracking
        now = time.time()
        budget_state = RouterState(
            run_id=run_id,
            state_schema_version=cfg.state_schema_version,
            topic="",  # Will be set during research
            version=0,
            hypothesis=Hypothesis(statement="", confidence=0.5),
            synthesis=Synthesis(text="", rationale=[]),
            step=0,
            conf_history=[0.5],
            token_budget_remaining=cfg.token_budget_total,
            cost_budget_remaining=cfg.cost_budget_total,
            time_deadline_ts=now + cfg.time_budget_seconds,
        )
        
        # Clients
        openai_client = AsyncOpenAI(api_key=cfg.openai_api_key)
        tavily_client = TavilyClient(api_key=cfg.tavily_api_key)
        firecrawl_client = FirecrawlApp(api_key=cfg.firecrawl_api_key)
        
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
        
        return Orchestrator(tea, gather, summarize, analyze, synth, refine, router, tool_registry, sem, cfg, tools_obj, store, run_id)
    
    async def research_async(
        self, 
        topic: str,
        run_id: Optional[str] = None,
        progress_callback: Optional[Callable[[ResearchProgress], None]] = None
    ) -> ResearchSynthesis:
        """
        Conduct asynchronous research on a given topic.
        
        Args:
            topic: The research topic to investigate
            run_id: Optional run ID to resume previous research
            progress_callback: Optional callback function to receive progress updates
            
        Returns:
            ResearchSynthesis containing the final research results
            
        Raises:
            ValueError: If topic is empty or API keys are missing
            Exception: If research fails due to technical issues
        """
        if not topic or not topic.strip():
            raise ValueError("Research topic cannot be empty")
        
        topic = topic.strip()
        actual_run_id = run_id or str(uuid.uuid4())
        
        self.log.bind(event="research_start", run_id=actual_run_id, topic=topic).info("Starting research")
        
        try:
            # Create orchestrator
            orchestrator = await self._create_orchestrator(actual_run_id)
            
            # Set up progress monitoring if callback provided
            if progress_callback:
                # This is a simplified progress tracking - in a full implementation,
                # you might want to hook into the orchestrator's state updates
                progress = ResearchProgress(
                    run_id=actual_run_id,
                    step=0,
                    max_steps=self.config.max_iterations,
                    hypothesis="Initializing...",
                    confidence=0.5,
                    synthesis_preview="Starting research...",
                    tokens_used=0,
                    cost_used=0.0,
                    time_elapsed=0.0
                )
                progress_callback(progress)
            
            # Run research
            result_dict = await orchestrator.run(topic, resume_state=None)
            
            # Convert to ResearchSynthesis
            synthesis = ResearchSynthesis(
                topic=result_dict["topic"],
                hypothesis=result_dict["hypothesis"]["statement"],
                confidence=result_dict["hypothesis"]["confidence"],
                synthesis_text=result_dict["synthesis"]["text"],
                future_questions=result_dict["future_questions"],
                meta=result_dict["meta"],
                run_id=actual_run_id
            )
            
            self.log.bind(event="research_complete", run_id=actual_run_id).info("Research completed successfully")
            return synthesis
            
        except Exception as e:
            self.log.bind(event="research_error", run_id=actual_run_id, error=str(e)).error("Research failed")
            raise
    
    def research(
        self, 
        topic: str,
        run_id: Optional[str] = None,
        progress_callback: Optional[Callable[[ResearchProgress], None]] = None
    ) -> ResearchSynthesis:
        """
        Conduct synchronous research on a given topic.
        
        This is a blocking call that runs the async research in a thread pool.
        
        Args:
            topic: The research topic to investigate
            run_id: Optional run ID to resume previous research
            progress_callback: Optional callback function to receive progress updates
            
        Returns:
            ResearchSynthesis containing the final research results
            
        Raises:
            ValueError: If topic is empty or API keys are missing
            Exception: If research fails due to technical issues
        """
        # Run async method in thread pool to avoid blocking
        future = self._executor.submit(
            asyncio.run, 
            self.research_async(topic, run_id, progress_callback)
        )
        return future.result()
    
    async def resume_research_async(
        self, 
        run_id: str,
        new_topic: Optional[str] = None,
        progress_callback: Optional[Callable[[ResearchProgress], None]] = None
    ) -> ResearchSynthesis:
        """
        Resume an existing research session asynchronously.
        
        Args:
            run_id: The run ID of the research session to resume
            new_topic: Optional new topic (if different from original)
            progress_callback: Optional callback function to receive progress updates
            
        Returns:
            ResearchSynthesis containing the final research results
        """
        orchestrator = await self._create_orchestrator(run_id)
        
        # Load existing state
        store = StateStore(orchestrator.cfg)
        resume_state = await store.load(run_id)
        
        if resume_state is None:
            raise ValueError(f"No research session found with run_id: {run_id}")
        
        topic = new_topic or resume_state.topic
        
        self.log.bind(event="research_resume", run_id=run_id, topic=topic).info("Resuming research")
        
        try:
            result_dict = await orchestrator.run(topic, resume_state=resume_state)
            
            synthesis = ResearchSynthesis(
                topic=result_dict["topic"],
                hypothesis=result_dict["hypothesis"]["statement"],
                confidence=result_dict["hypothesis"]["confidence"],
                synthesis_text=result_dict["synthesis"]["text"],
                future_questions=result_dict["future_questions"],
                meta=result_dict["meta"],
                run_id=run_id
            )
            
            self.log.bind(event="research_resume_complete", run_id=run_id).info("Research resume completed")
            return synthesis
            
        except Exception as e:
            self.log.bind(event="research_resume_error", run_id=run_id, error=str(e)).error("Research resume failed")
            raise
    
    def resume_research(
        self, 
        run_id: str,
        new_topic: Optional[str] = None,
        progress_callback: Optional[Callable[[ResearchProgress], None]] = None
    ) -> ResearchSynthesis:
        """
        Resume an existing research session synchronously.
        
        Args:
            run_id: The run ID of the research session to resume
            new_topic: Optional new topic (if different from original)
            progress_callback: Optional callback function to receive progress updates
            
        Returns:
            ResearchSynthesis containing the final research results
        """
        future = self._executor.submit(
            asyncio.run, 
            self.resume_research_async(run_id, new_topic, progress_callback)
        )
        return future.result()
    
    async def get_research_status_async(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a research session asynchronously.
        
        Args:
            run_id: The run ID to check status for
            
        Returns:
            Dictionary with research status information, or None if not found
        """
        try:
            cfg = Config(
                openai_api_key=self._get_api_key("OPENAI_API_KEY", self.config.openai_api_key),
                tavily_api_key=self._get_api_key("TAVILY_API_KEY", self.config.tavily_api_key),
                firecrawl_api_key=self._get_api_key("FIRECRAWL_API_KEY", self.config.firecrawl_api_key),
            )
            store = StateStore(cfg)
            state = await store.load(run_id)
            
            if state is None:
                return None
                
            return {
                "run_id": state.run_id,
                "topic": state.topic,
                "step": state.step,
                "hypothesis": state.hypothesis.model_dump(),
                "synthesis_preview": state.synthesis.text[:500] + "..." if len(state.synthesis.text) > 500 else state.synthesis.text,
                "confidence": state.hypothesis.confidence,
                "tokens_used": state.tokens_used,
                "cost_used": state.cost_used,
                "time_elapsed": time.time() - state.started_ts,
                "tasks_remaining": len([t for t in state.task_list if t.status == "open"]),
                "verification_done": state.verification_done,
                "critique_done": state.critique_done,
                "debate_done": state.debate_done
            }
            
        except Exception as e:
            self.log.bind(event="status_error", run_id=run_id, error=str(e)).error("Failed to get status")
            return None
    
    def get_research_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a research session synchronously.
        
        Args:
            run_id: The run ID to check status for
            
        Returns:
            Dictionary with research status information, or None if not found
        """
        future = self._executor.submit(
            asyncio.run, 
            self.get_research_status_async(run_id)
        )
        return future.result()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self._executor.shutdown(wait=True)
    
    def close(self):
        """Explicitly close the client and cleanup resources."""
        self._executor.shutdown(wait=True)

# Convenience functions for quick usage
def research(topic: str, config: Optional[ResearchConfig] = None) -> ResearchSynthesis:
    """
    Quick research function for simple use cases.
    
    Args:
        topic: Research topic
        config: Optional configuration
        
    Returns:
        ResearchSynthesis with results
    """
    with DeepResearchClient(config) as client:
        return client.research(topic)

async def research_async(topic: str, config: Optional[ResearchConfig] = None) -> ResearchSynthesis:
    """
    Quick async research function for simple use cases.
    
    Args:
        topic: Research topic
        config: Optional configuration
        
    Returns:
        ResearchSynthesis with results
    """
    client = DeepResearchClient(config)
    try:
        return await client.research_async(topic)
    finally:
        client.close()

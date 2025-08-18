import asyncio
import json
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import structlog

from deepresearcher.agents.analysis import AnalysisAgent
from deepresearcher.agents.refinement import RefinementAgent
from deepresearcher.agents.search_and_crawl import SearchAndCrawlAgent
from deepresearcher.agents.summarizer import SummarizerAgent
from deepresearcher.agents.synthesis import SynthesisAgent
from deepresearcher.agents.topic_exploration import TopicExplorationAgent
from deepresearcher.core.config import Config
from deepresearcher.core.models import (
    CritiqueReport,
    FinalSynthesis,
    Hypothesis,
    QueryBlock,
    ResearchResult,
    RouterDecision,
    RouterState,
    Synthesis,
    ToolSpec,
    VerifiedClaim,
)
from deepresearcher.providers.llm import LLM
from deepresearcher.storage.state_store import StateStore
from deepresearcher.tools.tools import Tools

log = structlog.get_logger("orchestrator")


class RouterAgent:
    def __init__(self, llm: LLM, toolspecs: List[ToolSpec], run_id: str):
        self.llm = llm
        self.toolspecs = toolspecs
        self.run_id = run_id

    async def decide(self, state: RouterState) -> RouterDecision:
        tools_txt = "\n".join(
            f"- {t.name}: {t.description}\n  args: {json.dumps(t.arg_schema)}"
            for t in self.toolspecs
        )
        tl = "\n".join([f"{i+1}. {t.description} (status={t.status})" for i, t in enumerate(state.task_list[:8])]) or "∅"
        time_remaining = max(0.0, state.time_deadline_ts - time.time())
        system = (
            "You are the central controller of a research team. "
            "Return ONLY: up to 3 short rationale bullets, a numeric scorecard (expected_gain, cost, uncertainty, predicted_cost), "
            "and a single tool call with JSON args."
        )
        user = (
            f"Run: {state.run_id} | Step: {state.step}\n"
            f"Hypothesis (confidence={state.hypothesis.confidence:.2f}): {state.hypothesis.statement}\n\n"
            f"Budget remaining → tokens: {state.token_budget_remaining}, cost: ${state.cost_budget_remaining:.2f}, time_s: {time_remaining:.1f}\n"
            f"Current Task List (top first):\n{tl}\n\n"
            "Focus on the TOP task. Prefer lower predicted_cost tools when budgets are tight. "
            "You may: (a) search/summarize/analyze/synthesize/update, (b) add a sub-task, (c) verify/critique, "
            "(d) review/prune plan, (e) debate before ending, (f) complete current task.\n\n"
            f"Current synthesis:\n{state.synthesis.text[:1000] or 'N/A'}\n\n"
            f"Last action result:\n{state.last_action_result or 'None'}\n\n"
            f"Available tools:\n{tools_txt}\n\n"
            "Return JSON with this exact shape:\n"
            "{\n"
            '  "rationale_bullets": ["<short>", "<short>", "<short>"],\n'
            '  "scores": {"expected_gain": <0..1>, "cost": <0..1>, "uncertainty": <0..1>, "predicted_cost": <0..1>},\n'
            '  "call": {"tool": "<name>", "arguments": {...}}\n'
            "}"
        )
        decision = await self.llm.json_schema_complete(system, user, RouterDecision)
        log.bind(event="router_decision", run_id=state.run_id, step=state.step,
                 tool=decision.call.tool, scores=decision.scores,
                 bullets=decision.rationale_bullets).info("router.decide")
        return decision

class Orchestrator:
    def __init__(
        self,
        tea: TopicExplorationAgent,
        gather: SearchAndCrawlAgent,
        summarize: SummarizerAgent,
        analyze: AnalysisAgent,
        synth: SynthesisAgent,
        refine: RefinementAgent,
        router: RouterAgent,
        tool_registry: Dict[str, Callable[..., Awaitable[Any]]],
        sem: asyncio.Semaphore,
        cfg: Config,
        tools_obj: Tools,
        store: StateStore,
        run_id: str,
    ):
        self.tea = tea
        self.gather = gather
        self.summarize = summarize
        self.analyze = analyze
        self.synth = synth
        self.refine = refine
        self.router = router
        self.tool_registry = tool_registry
        self.sem = sem
        self.cfg = cfg
        self.tools_obj = tools_obj
        self.store = store
        self.run_id = run_id

    @staticmethod
    def _merge_blocks(existing: List[QueryBlock], new: List[QueryBlock]) -> List[QueryBlock]:
        seen = {src.url for blk in existing for src in blk.sources}
        merged = [*existing]
        for blk in new:
            filtered = [src for src in blk.sources if src.url not in seen]
            seen.update({src.url for src in filtered})
            merged.append(QueryBlock(query=blk.query, sources=filtered))
        return merged

    @staticmethod
    async def _extract_key_claims_from_synthesis(synthesis: Synthesis) -> List[str]:
        text = synthesis.text or ""
        candidates = re.split(r"(?<=[:\.\!\?])\s+", text)
        claims = []
        for c in candidates:
            c2 = c.strip()
            if len(c2) >= 20 and any(w in c2.lower() for w in ("therefore","we find","the evidence","shows","indicates","suggests")):
                claims.append(c2[:240])
        return claims[:5] or candidates[:3]

    def _graceful_degradation(self, state: RouterState) -> bool:
        return (
            state.token_budget_remaining <= 0 or
            state.cost_budget_remaining <= 0.0 or
            time.time() >= state.time_deadline_ts
        )

    async def _initial_seed(self, topic: str, state: RouterState) -> Tuple[List[QueryBlock], Synthesis]:
        questions, hyp = await self.tea.propose(topic)
        state.hypothesis = hyp
        log.bind(event="init", run_id=self.run_id, questions=len(questions),
                 hypothesis=hyp.statement, conf=round(hyp.confidence,2)).info("seed.proposed")
        blocks: List[QueryBlock] = []
        if questions:
            seed_blocks = await self.gather.run_for_questions(questions, k_per_query=2)
            seed_blocks = await self.summarize.enrich(seed_blocks, self.sem)
            seed_blocks = await self.analyze.enrich(seed_blocks, self.sem)
            blocks = self._merge_blocks([], seed_blocks)
        synthesis = await self.synth.synthesize(blocks, state.hypothesis) if blocks else Synthesis(text="", rationale=[])
        return blocks, synthesis

    async def run(self, topic: str, resume_state: Optional[RouterState]=None) -> Dict[str, Any]:
        t0 = time.time()
        log.bind(event="run_start", run_id=self.run_id, topic=topic).info("orchestrator.start")

        if resume_state:
            state = resume_state
            state.topic = topic
            log.bind(event="resume", run_id=self.run_id, version=state.version, step=state.step).info("state.resume")
            state_blocks: List[QueryBlock] = []
        else:
            # Construct initial state shell; full seed after first save
            now = time.time()
            state = RouterState(
                run_id=self.run_id,
                state_schema_version=self.cfg.state_schema_version,
                topic=topic,
                version=0,
                hypothesis=Hypothesis(statement="", confidence=0.5),
                synthesis=Synthesis(text="", rationale=[]),
                step=0,
                conf_history=[0.5],
                token_budget_remaining=self.cfg.token_budget_total,
                cost_budget_remaining=self.cfg.cost_budget_total,
                time_deadline_ts=now + self.cfg.time_budget_seconds,
            )
            # Initial task
            root_desc = f"Validate hypothesis: {topic}"
            await self.tools_obj.add_sub_task_tool(state=state, task_description=root_desc)
            # Save initial state
            await self.store.save_cas(state, expected_version=state.version)
            state.version += 1
            state_blocks, synthesis = await self._initial_seed(topic, state)
            state.synthesis = synthesis
            state.conf_history = [state.hypothesis.confidence]

        prev_sources_count = 0
        final_synth: Optional[FinalSynthesis] = None
        steps_budget = self.cfg.iter_depth

        # Persist state snapshot helper
        async def persist_state_snapshot(s: RouterState, note: str = ""):
            ok = await self.store.save_cas(s, expected_version=s.version)
            if ok:
                s.version += 1
            log.bind(event="state_snapshot", run_id=s.run_id, step=s.step, version=s.version,
                     hyp_conf=round(s.hypothesis.confidence,3), tokens_used=s.tokens_used,
                     cost_used=round(s.cost_used,3), note=note).info("state.save")

        # After seed, save snapshot
        await persist_state_snapshot(state, note="post_seed")

        for step in range(state.step + 1, steps_budget + 1):
            state.step = step

            # Budget check (graceful degradation)
            if self._graceful_degradation(state):
                note = "budget_exhausted"
                log.bind(event="budget", run_id=self.run_id, step=step,
                         token_rem=state.token_budget_remaining,
                         cost_rem=round(state.cost_budget_remaining,2),
                         time_rem=round(state.time_deadline_ts-time.time(),1)).warning("budget.exhausted")
                # Attempt minimal debate before ending if possible
                try:
                    if not state.debate_done and state.synthesis.text:
                        debate = await self.tools_obj.conduct_adversarial_review_tool(
                            evidence=[],  # empty if no blocks cached; keep cheap
                            verified_claims=state.verified_claims,
                            current_synthesis=state.synthesis,
                            hypothesis=state.hypothesis,
                            critique=state.critique,
                        )
                        final_synth = FinalSynthesis.model_validate(debate["moderator"])
                        state.debate_done = True
                except Exception:
                    pass
                state.last_action_result = "Run ended early due to budget constraints."
                await persist_state_snapshot(state, note=note)
                break

            # Strategy review cadence (from v3.5)
            try:
                if self.cfg.review_every_steps > 0 and step % self.cfg.review_every_steps == 0:
                    res = await self.tools_obj.review_and_prune_plan_tool(state)
                    state.last_action_result = f"Review: pruned={len(res.get('pruned', []))}, reordered={len(res.get('reordered', []))}"
            except Exception as e:
                log.bind(event="review_error", run_id=self.run_id, step=step, err=str(e)).warning("review.failed")

            # Router decision
            decision = await self.router.decide(state)
            tool_name = decision.call.tool
            args = decision.call.arguments or {}

            # Cost-aware nudge: if predicted_cost high and budgets low, prefer cheaper tool (soft guard)
            predicted_cost = float(decision.scores.predicted_cost)
            if (state.cost_budget_remaining < self.cfg.cost_budget_total * 0.2 or
                state.token_budget_remaining < self.cfg.token_budget_total * 0.2):
                if predicted_cost > 0.7 and tool_name in ("search_and_crawl_tool", "verify_claims_tool"):
                    # Soft reroute to synth/update if possible
                    if state.synthesis.text:
                        tool_name = "synthesize_evidence_tool"
                        args = {"evidence": [], "hypothesis": state.hypothesis}

            # Policy gate before ending
            if tool_name == "end_research_tool":
                if self.cfg.require_verification_before_end and not state.verification_done and state.synthesis.text:
                    tool_name = "verify_claims_tool"
                    args = {"claims": await self._extract_key_claims_from_synthesis(state.synthesis)}
                elif self.cfg.require_critique_before_end and not state.critique_done:
                    tool_name = "critique_evidence_tool"
                    args = {"evidence": [], "verified": state.verified_claims or None}
                elif not state.debate_done:
                    tool_name = "conduct_adversarial_review_tool"
                    args = {
                        "evidence": [],  # keep cheap; or pass limited blocks
                        "verified_claims": state.verified_claims,
                        "current_synthesis": state.synthesis,
                        "hypothesis": state.hypothesis,
                        "critique": state.critique
                    }

            # Enrich args for stateful tools
            if tool_name in ("add_sub_task_tool", "complete_current_task_tool", "review_and_prune_plan_tool"):
                args = {"state": state, **args}

            # Execute tool
            t1 = time.time()
            try:
                if tool_name == "search_and_crawl_tool":
                    new_blocks = await self.tool_registry[tool_name](**args)
                    # NOTE: For brevity, we do not store blocks in external storage in v3.7.
                    # If needed, plug S3 and keep references in state.meta.
                    # Here, we just count new sources for signals.
                    added_sources = sum(len(b.sources) for b in new_blocks)
                    state.new_sources_history.append(added_sources)
                    state.last_action_result = f"Gathered {added_sources} sources."

                elif tool_name == "analyze_content_tool":
                    await self.tool_registry[tool_name](**args)
                    state.new_sources_history.append(1)
                    state.last_action_result = "Analyzed inline content."

                elif tool_name == "synthesize_evidence_tool":
                    syn = await self.tool_registry[tool_name](evidence=[], hypothesis=state.hypothesis)
                    # Keep existing synthesis if empty output
                    state.synthesis = syn if syn.text else state.synthesis
                    state.last_action_result = "Synthesis updated."

                elif tool_name == "update_hypothesis_tool":
                    hyp2 = await self.tool_registry[tool_name](evidence=[], hypothesis=state.hypothesis)
                    state.hypothesis = hyp2
                    state.last_action_result = f"Hypothesis confidence now {hyp2.confidence:.2f}"

                elif tool_name in ("add_sub_task_tool", "complete_current_task_tool"):
                    state.last_action_result = await self.tool_registry[tool_name](**args)

                elif tool_name == "calculator_tool":
                    state.last_action_result = f"Calculator: {await self.tool_registry[tool_name](**args)}"

                elif tool_name == "verify_claims_tool":
                    claims_arg = args.get("claims") or (await self._extract_key_claims_from_synthesis(state.synthesis))
                    vitems = await self.tool_registry[tool_name](claims=claims_arg)
                    state.verified_claims = vitems
                    state.verification_done = True
                    state.last_action_result = f"Verified {len(vitems)} claims."

                elif tool_name == "critique_evidence_tool":
                    report = await self.tool_registry[tool_name](evidence=[], verified=state.verified_claims or None)
                    state.critique = report
                    state.critique_done = True
                    state.last_action_result = "Critique completed."

                elif tool_name == "review_and_prune_plan_tool":
                    res = await self.tool_registry[tool_name](**args)
                    state.last_action_result = f"Review: pruned={len(res.get('pruned',[]))}, reordered={len(res.get('reordered',[]))}"

                elif tool_name == "conduct_adversarial_review_tool":
                    debate = await self.tool_registry[tool_name](**args)
                    final_synth = FinalSynthesis.model_validate(debate["moderator"])
                    state.debate_done = True
                    state.last_action_result = f"Adversarial review complete: {len(final_synth.paragraphs)} paragraphs."

                elif tool_name == "end_research_tool":
                    final_text = await self.tool_registry[tool_name](**args)
                    if final_text:
                        state.synthesis = Synthesis(
                            text=(state.synthesis.text or "") + "\n\n[Final Note]\n" + str(final_text),
                            rationale=state.synthesis.rationale
                        )
                    log.bind(event="end", run_id=self.run_id, step=step).info("orchestrator.end_requested")
                    await persist_state_snapshot(state, note="end_research")
                    break

            except Exception as e:
                state.last_action_result = f"Tool error in {tool_name}: {e}"
                log.bind(event="tool_error", run_id=self.run_id, tool=tool_name, err=str(e)).error("tool.failed")

            finally:
                duration = round(time.time() - t1, 3)
                state.recent_tools.append(tool_name)
                log.bind(event="tool_done", run_id=self.run_id, tool=tool_name, latency_s=duration,
                         last_action=state.last_action_result).info("tool.finished")

            # Keep synthesis reasonably fresh if cheap
            if not state.synthesis.text:
                try:
                    state.synthesis = await self.synth.synthesize([], state.hypothesis)
                except Exception:
                    pass

            # Confidence tracking
            state.hypothesis = self.refine.update_hypothesis(state.hypothesis, [])
            state.conf_history.append(state.hypothesis.confidence)

            await persist_state_snapshot(state, note=f"post_step_{step}")

            # Early exit if no tasks
            if not state.task_list:
                log.bind(event="done", run_id=self.run_id, step=step).info("no_tasks_remaining")
                break

        elapsed = round(time.time() - t0, 2)
        footer = ""
        if self._graceful_degradation(state):
            footer = "\n\n[Note] Run ended due to budget constraints (tokens/cost/time). Output is best-effort."

        future_qs = self.refine.replan_from_gaps(self.refine.identify_gaps([]), state.hypothesis)
        result = ResearchResult(
            topic=topic,
            hypothesis=state.hypothesis,
            synthesis=Synthesis(text=(state.synthesis.text + footer).strip(), rationale=state.synthesis.rationale),
            future_questions=future_qs,
            evidence=[],  # Omitted in v3.7 to keep state light; switch to blob storage for full provenance
            final_synthesis=final_synth,
            meta={
                "elapsed_sec": elapsed,
                "run_id": self.run_id,
                "steps_used": state.step,
                "model": self.cfg.llm_model,
                "embed_model": self.cfg.embed_model,
                "budgets": {
                    "initial_tokens": self.cfg.token_budget_total,
                    "initial_cost": self.cfg.cost_budget_total,
                    "initial_time_s": self.cfg.time_budget_seconds,
                    "tokens_used": state.tokens_used,
                    "cost_used": round(state.cost_used, 4),
                    "time_used_s": round(time.time() - state.started_ts, 2),
                    "ended_due_to_budget": self._graceful_degradation(state),
                },
                "tasks": {
                    "open": [t.description for t in state.task_list if t.status == "open"],
                    "log": [e.model_dump(mode="json") for e in state.task_log],
                },
                "router": {
                    "tools_used": state.recent_tools,
                },
                "verification": [vc.model_dump(mode="json") for vc in state.verified_claims],
                "critique": state.critique.model_dump(mode="json") if state.critique else None,
                "debate_done": state.debate_done,
            },
        )
        log.bind(event="run_complete", run_id=self.run_id, elapsed=elapsed,
                 tokens_used=result.meta["budgets"]["tokens_used"],
                 cost_used=result.meta["budgets"]["cost_used"]).info("orchestrator.complete")
        return result.model_dump(mode="json")

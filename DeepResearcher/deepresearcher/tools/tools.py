import asyncio
import os
import re
import json
import time
import math
import uuid
from typing import List, Dict, Optional, Any, Tuple

import structlog
from pydantic import BaseModel

from ..agents.analysis import AnalysisAgent
from ..agents.refinement import RefinementAgent
from ..agents.search_and_crawl import SearchAndCrawlAgent
from ..agents.summarizer import SummarizerAgent
from ..agents.synthesis import SynthesisAgent
from ..core.config import Config
from ..core.models import (
    QueryBlock,
    ExtractedData,
    Hypothesis,
    Synthesis,
    TaskItem,
    TaskLogEntry,
    VerifiedClaim,
    CritiqueReport,
    SynthesisParagraph,
    FinalSynthesis,
    RouterState,
)
from ..providers.embedding import EmbeddingsProvider
from ..providers.llm import LLM
from ..utils.utils import (
    tokenize,
    hash_key,
    domain,
    authority_weight,
    freshness_score,
    bias_level,
    canonicalize,
)

log = structlog.get_logger(__name__)

def _text_overlap(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))

class Tools:
    def __init__(
        self,
        gather: SearchAndCrawlAgent,
        summarize: SummarizerAgent,
        analyze: AnalysisAgent,
        synth: SynthesisAgent,
        refine: RefinementAgent,
        sem: asyncio.Semaphore,
        cfg: Config,
        embed: EmbeddingsProvider,
        llm: LLM,
        run_id: str,
    ):
        self.gather = gather
        self.summarize = summarize
        self.analyze = analyze
        self.synth = synth
        self.refine = refine
        self.sem = sem
        self.cfg = cfg
        self.embed = embed
        self.llm = llm
        self.run_id = run_id
        self.verif_dir = os.path.join(cfg.cache_dir, "verification")
        os.makedirs(self.verif_dir, exist_ok=True)

    async def search_and_crawl_tool(self, query: str) -> List[QueryBlock]:
        blks = await self.gather.run_for_questions([query], k_per_query=2)
        blks = await self.summarize.enrich(blks, self.sem)
        blks = await self.analyze.enrich(blks, self.sem)
        log.bind(event="tool_call", run_id=self.run_id, tool="search_and_crawl_tool", query=query,
                 blocks=len(blks)).info("tool.search_and_crawl")
        return blks

    async def analyze_content_tool(self, content: str) -> ExtractedData:
        fake_url = "inline://analysis"
        summary = await self.summarize.summarize_source(fake_url, content)
        data = await self.analyze.extract_from_source(fake_url, summary)
        log.bind(event="tool_call", run_id=self.run_id, tool="analyze_content_tool").info("tool.analyze_content")
        return data

    async def synthesize_evidence_tool(self, evidence: List[QueryBlock], hypothesis: Hypothesis) -> Synthesis:
        out = await self.synth.synthesize(evidence, hypothesis)
        log.bind(event="tool_call", run_id=self.run_id, tool="synthesize_evidence_tool").info("tool.synthesize_evidence")
        return out

    async def update_hypothesis_tool(self, evidence: List[QueryBlock], hypothesis: Hypothesis) -> Hypothesis:
        out = self.refine.update_hypothesis(hypothesis, evidence)
        log.bind(event="tool_call", run_id=self.run_id, tool="update_hypothesis_tool",
                 confidence=out.confidence).info("tool.update_hypothesis")
        return out

    async def calculator_tool(self, expression: str) -> str:
        if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s%^]+", expression):
            return "Error: unsupported expression"
        try:
            res = str(eval(expression, {"__builtins__": {}}, {}))
            log.bind(event="tool_call", run_id=self.run_id, tool="calculator_tool").info("tool.calculator")
            return res
        except Exception as e:
            return f"Error: {e}"

    async def end_research_tool(self, final_conclusion: str) -> str:
        log.bind(event="tool_call", run_id=self.run_id, tool="end_research_tool").info("tool.end_research")
        return final_conclusion

    async def add_sub_task_tool(self, state: "RouterState", task_description: str) -> str:
        max_tasks = 20  # Default maximum tasks
        if len(state.task_list) >= max_tasks:
            return "Task list is full; cannot add."
        desc = (task_description or "").strip()
        if not desc:
            return "Task description is empty."
        if state.task_list and state.task_list[0].description.strip().lower() == desc.lower():
            return "Refused to add duplicate top task."
        tid = str(uuid.uuid4())
        item = TaskItem(id=tid, description=desc, created_step=state.step)
        state.task_list.insert(0, item)
        state.task_log.append(TaskLogEntry(step=state.step, action="add", task_id=tid, description=item.description))
        log.bind(event="tool_call", run_id=self.run_id, tool="add_sub_task_tool", task_id=tid).info("tool.add_sub_task")
        return f"Added sub-task: {item.description}"

    async def complete_current_task_tool(self, state: "RouterState", summary_of_completion: str) -> str:
        if not state.task_list:
            return "No task to complete."
        item = state.task_list.pop(0)
        item.status = "done"
        if summary_of_completion:
            item.notes.append(summary_of_completion.strip())
        state.task_log.append(TaskLogEntry(step=state.step, action="complete", task_id=item.id,
                                           description=item.description, summary=(summary_of_completion or "").strip()))
        log.bind(event="tool_call", run_id=self.run_id, tool="complete_current_task_tool", task_id=item.id).info("tool.complete_task")
        return f"Completed task: {item.description}"

    def _verif_cache_path(self, claim: str) -> str:
        return os.path.join(self.verif_dir, f"{_hash_key(claim)}.json")

    def _load_verif_cache(self, claim: str) -> Optional[List[VerifiedClaim]]:
        path = self._verif_cache_path(claim)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ts = data.get("_ts", 0)
            if (time.time() - ts) > self.cfg.verification_ttl_days * 86400:
                return None
            items = [VerifiedClaim.model_validate(vc) for vc in data.get("items", [])]
            return items
        except Exception:
            return None

    def _save_verif_cache(self, claim: str, items: List[VerifiedClaim]) -> None:
        path = self._verif_cache_path(claim)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"_ts": time.time(), "items": [i.model_dump(mode="json") for i in items]}, f)

    async def verify_claims_tool(self, claims: List[str]) -> List[VerifiedClaim]:
        verified: List[VerifiedClaim] = []
        for claim in claims:
            claim_norm = claim.strip()
            if not claim_norm:
                continue
            cached = self._load_verif_cache(claim_norm)
            if cached:
                verified.extend(cached)
                continue
            key_terms = " ".join(list(set(tokenize(claim_norm)))[:8])
            query_pack = [
                f"{claim_norm}",
                f"fact check {claim_norm}",
                f"evidence against {claim_norm}",
                f"{claim_norm} site:.gov OR site:.edu",
                f"{key_terms} systematic review",
            ]
            all_blocks: List[QueryBlock] = []
            for q in query_pack:
                blks = await self.gather.run_for_questions([q], k_per_query=2)
                blks = await self.summarize.enrich(blks, self.sem)
                blks = await self.analyze.enrich(blks, self.sem)
                all_blocks.extend(blks)
            support_urls: Dict[str, float] = {}
            refute_urls: Dict[str, float] = {}
            neutral_urls: Dict[str, float] = {}
            def _bump(dct: Dict[str, float], url: str, w: float):
                dct[url] = dct.get(url, 0.0) + w
            for blk in all_blocks:
                for src in blk.sources:
                    if src.error or not src.extracted:
                        continue
                    auth = src.authority or authority_weight(src.url, src.pub_date, None)
                    for c in src.extracted.claims:
                        text = c.get("text") or ""
                        pol = (c.get("polarity") or "").lower()
                        if _text_overlap(text, claim_norm) < 0.35:
                            continue
                        if pol == "support":
                            _bump(support_urls, src.url, auth)
                        elif pol == "refute":
                            _bump(refute_urls, src.url, auth)
                        else:
                            _bump(neutral_urls, src.url, auth)
            S = sum(support_urls.values()); R = sum(refute_urls.values()); U = sum(neutral_urls.values())
            T = S + R + U + 1e-6
            V = (S - R) / T
            supp_domains = {domain(u) for u in support_urls}
            ref_domains = {domain(u) for u in refute_urls}
            if V >= 0.25 and len(supp_domains) >= 2 and S >= 2.0:
                status = "Supported"
            elif V <= -0.25 and len(ref_domains) >= 2 and R >= 2.0:
                status = "Refuted"
            elif (abs(V) < 0.25) and (S + R >= 2.0):
                status = "Contested"
            else:
                status = "Unclear"
            vc = VerifiedClaim(
                claim=claim_norm,
                status=status,
                supporting_urls=sorted(support_urls, key=lambda u: support_urls[u], reverse=True)[:6],
                refuting_urls=sorted(refute_urls, key=lambda u: refute_urls[u], reverse=True)[:6],
                score=round(V, 3),
                method={"query_pack": query_pack, "S": S, "R": R, "U": U}
            )
            verified.append(vc)
            self._save_verif_cache(claim_norm, [vc])
        log.bind(event="tool_call", run_id=self.run_id, tool="verify_claims_tool", claims=len(claims),
                 verified=len(verified)).info("tool.verify_claims")
        return verified

    async def critique_evidence_tool(self, evidence: List[QueryBlock], verified: Optional[List[VerifiedClaim]]=None) -> CritiqueReport:
        urls = []
        auth_scores: Dict[str, float] = {}
        freshness_vals: List[float] = []
        bias_med_count = 0
        by_domain: Dict[str, int] = {}
        for b in evidence:
            for s in b.sources:
                if s.error or not s.url: continue
                urls.append(s.url)
                d = domain(s.url); by_domain[d] = by_domain.get(d, 0) + 1
                auth = s.authority if s.authority is not None else authority_weight(s.url, s.pub_date, None)
                auth_scores[s.url] = auth
                freshness_vals.append(s.freshness if s.freshness is not None else freshness_score(s.pub_date, s.event_date))
                if (s.bias or bias_level(s.url)) in ("med", "high"): bias_med_count += 1
        biases: List[str] = []
        if urls:
            top_dom, top_cnt = max(by_domain.items(), key=lambda kv: kv[1])
            if top_cnt / max(1, len(urls)) > 0.6:
                biases.append(f"Single-source concentration: {int(100*top_cnt/len(urls))}% from {top_dom}.")
        if bias_med_count / max(1, len(urls)) > 0.5:
            biases.append("Vendor/PR dominance: majority of sources show vendor-like bias signals.")
        if freshness_vals:
            freshness_median = sorted(freshness_vals)[len(freshness_vals)//2]
            if freshness_median < 0.4:
                biases.append("Corpus is stale (low median freshness).")
        contradictions: List[Dict[str, str]] = []
        seen_claims: List[Tuple[str, str]] = []
        for b in evidence:
            for s in b.sources:
                if s.extracted:
                    for c in s.extracted.claims:
                        t = (c.get("text") or "").strip()
                        pol = (c.get("polarity") or "").lower()
                        if not t: continue
                        norm = " ".join(tokenize(t))[:160]
                        for prev_norm, prev_pol in seen_claims:
                            if _text_overlap(norm, prev_norm) > 0.75 and prev_pol != pol:
                                contradictions.append({"a": s.url, "b": "previous", "note": f"Opposing polarities on: {t[:80]}..."})
                        seen_claims.append((norm, pol))
        top_authority = sorted(auth_scores.items(), key=lambda kv: kv[1], reverse=True)[:10]
        authority_ranking = [{"url": u, "authority": round(a, 2)} for u, a in top_authority]
        gaps: List[str] = []
        if verified:
            for vc in verified:
                if vc.status in ("Contested", "Unclear"):
                    gaps.append(f"Claim unresolved ({vc.status}): {vc.claim}")
        recommended_tasks: List[str] = []
        if biases:
            recommended_tasks.append("Add sub-task: expand domain diversity to reduce source concentration.")
        if gaps:
            recommended_tasks.extend([f"Add sub-task: targeted verification on '{g[:64]}...'" for g in gaps[:3]])
        if not verified:
            recommended_tasks.append("Add sub-task: run verify_claims_tool on key synthesis claims before concluding.")
        report = CritiqueReport(
            biases=biases, contradictions=contradictions, authority_ranking=authority_ranking,
            gaps=gaps, recommended_tasks=recommended_tasks
        )
        log.bind(event="tool_call", run_id=self.run_id, tool="critique_evidence_tool",
                 biases=len(biases), contradictions=len(contradictions)).info("tool.critique_evidence")
        return report

    def _evidence_authority_map(self, evidence: List[QueryBlock]) -> Dict[str, float]:
        amap: Dict[str, float] = {}
        for b in evidence:
            for s in b.sources:
                if s.url:
                    amap[s.url] = float(s.authority if s.authority is not None else authority_weight(s.url, s.pub_date, None))
        return amap

    def _verification_weight(self, url: str, verified: List[VerifiedClaim]) -> float:
        score = 0.0; found = False
        for vc in verified:
            if url in vc.supporting_urls: score += 1.0; found = True
            if url in vc.refuting_urls: score -= 0.5; found = True
        return score if found else 0.0

    def _para_confidence(self, urls: List[str], amap: Dict[str, float], verified: List[VerifiedClaim], contradiction_penalty: float) -> float:
        U = [u for u in urls if u in amap]
        if not U:
            return 0.0
        A_vals = [amap[u] for u in U]
        V_vals = [self._verification_weight(u, verified) for u in U]
        quality = sum(0.6*a + 0.4*v for a, v in zip(A_vals, V_vals)) / (len(U) + 1e-6)
        doms = {domain(u) for u in U}
        D = math.log(1 + len(doms)) / math.log(1 + 5)
        conf = min(1.0, quality * (0.7 + 0.3*D)) - contradiction_penalty
        return float(max(0.0, min(1.0, conf)))

    async def conduct_adversarial_review_tool(
        self,
        evidence: List[QueryBlock],
        verified_claims: List[VerifiedClaim],
        current_synthesis: Synthesis,
        hypothesis: Hypothesis,
        critique: Optional[CritiqueReport] = None,
    ) -> Dict[str, Any]:
        amap = self._evidence_authority_map(evidence)
        all_sources: List[Tuple[str, float]] = []
        seen = set()
        for b in evidence:
            for s in b.sources:
                if s.url and s.url not in seen:
                    seen.add(s.url)
                    all_sources.append((s.url, amap.get(s.url, 0.3)))
        all_sources.sort(key=lambda kv: kv[1], reverse=True)
        picked, dom_seen = [], set()
        for u, a in all_sources:
            d = domain(u)
            if d in dom_seen: continue
            picked.append((u, a)); dom_seen.add(d)
            if len(picked) >= self.cfg.debate_top_k: break
        urls_for_debate = [u for u, _ in picked]
        evidence_snippets: Dict[str, str] = {}
        for b in evidence:
            for s in b.sources:
                if s.url in urls_for_debate and s.summary:
                    evidence_snippets[s.url] = s.summary
        verified_map = {
            "supported": list({u for vc in verified_claims for u in vc.supporting_urls}),
            "refuted": list({u for vc in verified_claims for u in vc.refuting_urls}),
            "statuses": [{ "claim": vc.claim, "status": vc.status, "score": vc.score } for vc in verified_claims]
        }
        min_domains = self.cfg.debate_min_independent_domains
        class _SideResp(BaseModel):
            points: List[Dict[str, Any]]
        blue_sys = "You are the Pro-Hypothesis debater. Use ONLY provided evidence bundle. Return strictly valid JSON."
        blue_user = (
            f"Hypothesis (conf={hypothesis.confidence:.2f}): {hypothesis.statement}\n\n"
            f"Evidence: {json.dumps(urls_for_debate)}\n\n"
            f"Summaries: {json.dumps({u: evidence_snippets.get(u,'')[:500] for u in urls_for_debate})[:3500]}\n\n"
            f"Verified: {json.dumps(verified_map)[:1500]}\n\n"
            f"Rule: cite at least {min_domains} independent domains; do not introduce new facts.\n"
            'Return JSON: {"points":[{"text":"...", "urls":["..."], "strength":0.0}]}'
        )
        blue = await self.llm.json_schema_complete(blue_sys, blue_user, _SideResp, temperature=self.cfg.debate_temperature_blue)
        red_sys = "You are the Anti-Hypothesis debater. Attack weaknesses using ONLY provided evidence. Return strictly valid JSON."
        red_user = blue_user.replace("Pro-Hypothesis", "Anti-Hypothesis")
        red = await self.llm.json_schema_complete(red_sys, red_user, _SideResp, temperature=self.cfg.debate_temperature_red)
        class _ModeratorResp(BaseModel):
            paragraphs: List[Dict[str, Any]]
            executive_summary: str
        moderator_sys = "You are an impartial judge. Weigh arguments based on quality of evidence. Return strictly valid JSON."
        moderator_user = (
            f"Blue points: {json.dumps(blue.model_dump(mode='json'))[:4000]}\n\n"
            f"Red points: {json.dumps(red.model_dump(mode='json'))[:4000]}\n\n"
            f"Authority map: {json.dumps({u: round(amap.get(u,0.3),2) for u in urls_for_debate})}\n"
            f"Verified map: {json.dumps(verified_map)[:1500]}\n"
            f"Current synthesis (context): {current_synthesis.text[:1200]}\n"
            f"Guidance: produce 3–6 paragraphs. Each paragraph must list 1–{self.cfg.synthesis_max_evidence_urls_per_para} supporting URLs from the bundle (distinct domains preferred). "
            "Do NOT invent citations.\n"
            'Return JSON: {"paragraphs":[{"text":"...", "evidence_urls":["..."]}], "executive_summary":"..."}'
        )
        mod = await self.llm.json_schema_complete(moderator_sys, moderator_user, _ModeratorResp, temperature=self.cfg.debate_temperature_mod)
        paragraphs: List[SynthesisParagraph] = []
        contradiction_penalty = 0.0
        if critique and critique.contradictions:
            contradiction_penalty = min(0.3, 0.05 * len(critique.contradictions))
        for p in mod.paragraphs[:6]:
            raw_urls = [canonicalize(u) for u in (p.get("evidence_urls") or [])]
            dedup = []; seen_d = set()
            for u in raw_urls:
                d = domain(u)
                if u in amap and d not in seen_d:
                    dedup.append(u); seen_d.add(d)
                if len(dedup) >= self.cfg.synthesis_max_evidence_urls_per_para:
                    break
            conf = self._para_confidence(dedup, amap, verified_claims, contradiction_penalty)
            text = (p.get("text") or "").strip()
            if text and dedup:
                paragraphs.append(SynthesisParagraph(text=text, confidence_score=conf, evidence_urls=dedup))
        final = FinalSynthesis(paragraphs=paragraphs, executive_summary=(mod.executive_summary or "").strip()[:1200])
        log.bind(event="tool_call", run_id=self.run_id, tool="conduct_adversarial_review_tool",
                 paragraphs=len(paragraphs)).info("tool.debate")
        return {"moderator": final.model_dump(mode="json")}

    async def review_and_prune_plan_tool(self, state: "RouterState", current_evidence: List[QueryBlock]) -> str:
        """Review current research plan and prune/reorder tasks based on progress and evidence quality."""
        if not state.task_list:
            return "No tasks to review."
        
        # Simple pruning logic - remove duplicate or low-value tasks
        seen_descriptions = set()
        pruned_tasks = []
        pruned_count = 0
        
        for task in state.task_list:
            desc_lower = task.description.lower()
            # Check for duplicates or very similar tasks
            is_duplicate = any(_text_overlap(desc_lower, seen) > 0.8 for seen in seen_descriptions)
            
            if not is_duplicate and pruned_count < self.cfg.max_prunes_per_review:
                pruned_tasks.append(task)
                seen_descriptions.add(desc_lower)
            else:
                pruned_count += 1
        
        # Update task list
        original_count = len(state.task_list)
        state.task_list = pruned_tasks
        
        log.bind(event="tool_call", run_id=self.run_id, tool="review_and_prune_plan_tool",
                 original_tasks=original_count, remaining_tasks=len(pruned_tasks)).info("tool.review_and_prune")
        
        return f"Reviewed plan: kept {len(pruned_tasks)} tasks, pruned {original_count - len(pruned_tasks)} duplicates/low-value tasks."

from typing import List

import structlog
from pydantic import BaseModel

from ..core.models import QueryBlock, Hypothesis, Synthesis
from ..providers.llm import LLM

log = structlog.get_logger(__name__)

class SynthesisAgent:
    def __init__(self, llm: LLM, run_id: str):
        self.llm = llm
        self.run_id = run_id

    async def synthesize(self, blocks: List[QueryBlock], hyp: Hypothesis) -> Synthesis:
        lines = []
        for b in blocks:
            for s in b.sources:
                if s.extracted:
                    for c in s.extracted.claims[:3]:
                        lines.append(f"- {c.get('text','').strip()} (polarity={c.get('polarity','')}) [{s.url}]")
                    for st in s.extracted.stats[:2]:
                        metric = st.get("metric", "")
                        val = st.get("value", "")
                        unit = st.get("unit", "")
                        lines.append(f"- {metric}: {val}{(' '+unit) if unit else ''} [{s.url}]")
        evidence = "\n".join(lines[:200])

        class _SynthResp(BaseModel):
            text: str

        system = "You are a rigorous research writer. Output clear, concise synthesis as plain text within JSON."
        user = (
            f"Working hypothesis (confidence={hyp.confidence:.2f}): {hyp.statement}\n\n"
            "Using the following structured evidence (claims & stats with provenance), write a synthesized answer that:\n"
            "- States current consensus/uncertainty\n- Highlights support/refutation of the hypothesis\n"
            "- Notes key caveats\n- Ends with 3 bullet recommendations for next steps\n\n"
            f"EVIDENCE:\n{evidence}\n"
            'Return JSON with a single field: {"text": "..."}'
        )
        resp = await self.llm.json_schema_complete(system, user, _SynthResp)
        text = resp.text
        rationale = []
        for b in blocks:
            for s in b.sources:
                if s.extracted and (s.extracted.claims or s.extracted.stats):
                    rationale.append(s.url)
        seen = set(); rat = []
        for u in rationale:
            if u not in seen:
                seen.add(u); rat.append(u)
            if len(rat) >= 50: break
        log.bind(event="tool_call", run_id=self.run_id, tool="synthesize", rationale=len(rat)).info("synthesis.synthesize")
        return Synthesis(text=text, rationale=rat)

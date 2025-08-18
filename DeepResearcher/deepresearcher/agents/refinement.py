from typing import List, Tuple

from ..core.models import QueryBlock, Hypothesis

class RefinementAgent:
    def __init__(self, conf_alpha: float):
        self.conf_alpha = conf_alpha

    @staticmethod
    def _count_support_refute(blocks: List[QueryBlock]) -> Tuple[int, int]:
        sup, ref = 0, 0
        for b in blocks:
            for s in b.sources:
                if s.extracted:
                    for c in s.extracted.claims:
                        pol = (c.get("polarity") or "").lower()
                        if pol == "support":
                            sup += 1
                        elif pol == "refute":
                            ref += 1
        return sup, ref

    def update_hypothesis(self, hyp: Hypothesis, blocks: List[QueryBlock]) -> Hypothesis:
        S, C = self._count_support_refute(blocks)
        denom = max(1.0, S + C)
        delta = self.conf_alpha * (S - C) / denom
        hyp.confidence = float(max(0.0, min(1.0, hyp.confidence + delta)))
        return hyp

    @staticmethod
    def identify_gaps(blocks: List[QueryBlock]) -> List[str]:
        has_accuracy = any(
            (st.get("metric", "").lower() == "accuracy")
            for b in blocks for s in b.sources if s.extracted for st in s.extracted.stats
        )
        needs = []
        if not has_accuracy:
            needs.append("Report quantitative performance (e.g., accuracy on benchmark Z) with sample size and CI.")
        has_ablation = any(
            ("ablation" in (c.get("text","").lower()))
            for b in blocks for s in b.sources if s.extracted for c in s.extracted.claims
        )
        if not has_ablation:
            needs.append("Locate ablation or causal evidence isolating the effect of X on Y.")
        return needs[:max(1, len(needs))]

    @staticmethod
    def replan_from_gaps(gaps: List[str], hyp: Hypothesis) -> List[str]:
        qs = []
        for g in gaps:
            qs.append(f"{g} In particular, evaluate the working hypothesis: {hyp.statement}")
        return qs or [f"Find the strongest evidence for/against: {hyp.statement}"]

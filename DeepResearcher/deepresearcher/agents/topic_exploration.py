from typing import List, Tuple

from ..core.models import Hypothesis, QuestionsPayload, HypothesisPayload
from ..providers.llm import LLM

class TopicExplorationAgent:
    def __init__(self, llm: LLM, max_questions: int):
        self.llm = llm
        self.max_questions = max_questions

    async def propose(self, topic: str) -> Tuple[List[str], Hypothesis]:
        q_sys = "You are a precise research planner. Output strictly valid JSON."
        q_user = (
            f"Generate 3–{self.max_questions} broad, open-ended research questions covering applications, "
            f"challenges, impacts, and future trends.\n\nTopic: {topic}\n\n"
            'Return JSON: {"questions": ["...","..."]}'
        )
        questions = (await self.llm.json_schema_complete(q_sys, q_user, QuestionsPayload)).questions

        h_sys = "You are a careful scientist. Output strictly valid JSON."
        h_user = (
            "Propose a single, testable working hypothesis about the topic with an initial confidence in [0,1].\n"
            f"Topic: {topic}\n\nReturn JSON: {{\"statement\":\"...\",\"confidence\":0.5}}"
        )
        hp = await self.llm.json_schema_complete(h_sys, h_user, HypothesisPayload)
        return [q.strip() for q in questions if q.strip()], Hypothesis(statement=hp.statement.strip(), confidence=float(hp.confidence))

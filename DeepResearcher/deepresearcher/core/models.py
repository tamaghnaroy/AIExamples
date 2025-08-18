import time
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict

# ==============================================================================
# 3) Utilities & Schemas (abridged from v3.6)
# ==============================================================================

class LLMPrice(BaseModel):
    prompt: float
    completion: float

class SummaryPayload(BaseModel):
    bullets: List[str]

class QuestionsPayload(BaseModel):
    questions: List[str]

class HypothesisPayload(BaseModel):
    statement: str
    confidence: float

class Claim(BaseModel):
    text: str
    polarity: str

class Stat(BaseModel):
    metric: str
    value: Any
    unit: Optional[str] = None
    n: Optional[Any] = None
    ci: Optional[List[float]] = None

class Entity(BaseModel):
    type: str
    name: str

class Date(BaseModel):
    type: str
    value: str

class ExtractedData(BaseModel):
    claims: List[Claim] = Field(default_factory=list)
    stats: List[Stat] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    dates: List[Date] = Field(default_factory=list)
    url: Optional[str] = None

class Synthesis(BaseModel):
    text: str
    rationale: List[str]

class Hypothesis(BaseModel):
    statement: str
    confidence: float = 0.5

class EnrichedSource(BaseModel):
    model_config = ConfigDict(extra="allow")
    url: str
    content: str
    error: Optional[str] = None
    publisher: Optional[str] = None
    pub_date: Optional[str] = None
    event_date: Optional[str] = None
    evidence_type: Optional[str] = None
    bias: Optional[str] = None
    authority: Optional[float] = None
    freshness: Optional[float] = None
    summary: Optional[str] = None
    extracted: Optional[ExtractedData] = None

class QueryBlock(BaseModel):
    query: str
    sources: List[EnrichedSource]

class VerifiedClaim(BaseModel):
    claim: str
    status: str
    supporting_urls: List[str]
    refuting_urls: List[str]
    score: float
    method: Dict[str, Any]

class CritiqueReport(BaseModel):
    biases: List[str]
    contradictions: List[Dict[str, Any]]
    authority_ranking: List[Dict[str, Any]]
    gaps: List[str]
    recommended_tasks: List[str]

class SynthesisParagraph(BaseModel):
    text: str
    confidence_score: float
    evidence_urls: List[str]

class FinalSynthesis(BaseModel):
    paragraphs: List[SynthesisParagraph]
    executive_summary: str

class ToolSpec(BaseModel):
    name: str
    description: str
    arg_schema: Dict[str, Any]

class ToolCall(BaseModel):
    tool: str
    arguments: Optional[Dict[str, Any]] = None

class RouterScores(BaseModel):
    expected_gain: float
    cost: float
    uncertainty: float
    predicted_cost: float

class RouterDecision(BaseModel):
    rationale_bullets: List[str]
    scores: RouterScores
    call: ToolCall

class TaskItem(BaseModel):
    id: str
    description: str
    status: str = "open"
    created_step: int
    notes: List[str] = Field(default_factory=list)

class TaskLogEntry(BaseModel):
    step: int
    action: str
    task_id: str
    description: str
    summary: Optional[str] = None

class RouterState(BaseModel):
    run_id: str
    state_schema_version: str
    topic: str
    version: int
    hypothesis: Hypothesis
    synthesis: Synthesis
    step: int
    conf_history: List[float]
    token_budget_remaining: int
    cost_budget_remaining: float
    time_deadline_ts: float
    task_list: List[TaskItem] = Field(default_factory=list)
    task_log: List[TaskLogEntry] = Field(default_factory=list)
    verified_claims: List[VerifiedClaim] = Field(default_factory=list)
    critique: Optional[CritiqueReport] = None
    new_sources_history: List[int] = Field(default_factory=list)
    recent_tools: List[str] = Field(default_factory=list)
    last_action_result: Optional[str] = None
    verification_done: bool = False
    critique_done: bool = False
    debate_done: bool = False
    started_ts: float = Field(default_factory=time.time)
    tokens_used: int = 0
    cost_used: float = 0.0

class ResearchResult(BaseModel):
    topic: str
    hypothesis: Hypothesis
    synthesis: Synthesis
    future_questions: List[str]
    evidence: List[QueryBlock]
    final_synthesis: Optional[FinalSynthesis] = None
    meta: Dict[str, Any]

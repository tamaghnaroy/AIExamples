from typing import Dict, Any, TypedDict, List, Optional

class GraphState(TypedDict):
    """
    Represents the state of the Deep Planning LangGraph as it passes through each node.
    Contains the user's inputs, generated documents, and control flags.
    """
    initial_idea: str
    qna_history: Dict[str, Any]
    prd_document: str
    tech_design_document: str
    user_feedback: str
    is_docs_approved: bool
    actionable_notes: str  # The NOTES.md content
    testing_plan: str  # The TESTING_PLAN.md content
    final_package: Dict[str, str]  # Filename -> Content

from typing import Dict, Any, TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .graph_state import GraphState
from .agents.safe_product_interviewer import SafeProductInterviewerAgent
from .agents.prd_generator import PRDGeneratorAgent
from .agents.tech_design_generator import TechDesignGeneratorAgent
from .agents.technical_manager import TechnicalManagerAgent
from .agents.test_developer import TestDeveloperAgent
from .agents.final_assembler import FinalAssemblerAgent

def create_deep_planning_graph() -> StateGraph:
    """Create the Deep Planning LangGraph with all agent nodes."""
    
    # Initialize agents
    product_interviewer = SafeProductInterviewerAgent()
    prd_generator = PRDGeneratorAgent()
    tech_design_generator = TechDesignGeneratorAgent()
    technical_manager = TechnicalManagerAgent()
    test_developer = TestDeveloperAgent()
    final_assembler = FinalAssemblerAgent()
    
    # Define state flow graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("product_interviewer", product_interviewer.run)
    workflow.add_node("prd_generator", prd_generator.run)
    workflow.add_node("tech_design_generator", tech_design_generator.run)
    workflow.add_node("user_review", lambda state: {"is_docs_approved": True})  # Mock for now, would be interactive
    workflow.add_node("technical_manager", technical_manager.run)
    workflow.add_node("test_developer", test_developer.run)
    workflow.add_node("final_assembler", final_assembler.run)
    
    # Define the conditional logic for review feedback loop
    def after_review(state: GraphState) -> Literal["revise", "continue"]:
        """Determine whether to continue or revise based on user feedback."""
        if state.get("is_docs_approved", False):
            return "continue"
        return "revise"
    
    # Define edges (transitions between nodes)
    workflow.add_edge("product_interviewer", "prd_generator")
    workflow.add_edge("prd_generator", "tech_design_generator")
    workflow.add_edge("tech_design_generator", "user_review")
    workflow.add_conditional_edges(
        "user_review",
        after_review,
        {
            "revise": "prd_generator",  # Loop back for revisions
            "continue": "technical_manager"  # Continue if approved
        }
    )
    workflow.add_edge("technical_manager", "test_developer")
    workflow.add_edge("test_developer", "final_assembler")
    workflow.add_edge("final_assembler", END)
    
    # Set entry point
    workflow.set_entry_point("product_interviewer")
    
    # Compile the graph
    return workflow.compile()

def initialize_state(idea: str) -> GraphState:
    """Initialize the graph state with the user's initial idea."""
    return {
        "initial_idea": idea,
        "qna_history": {},
        "prd_document": "",
        "tech_design_document": "",
        "user_feedback": "",
        "is_docs_approved": False,
        "actionable_notes": "",
        "testing_plan": "",
        "final_package": {}
    }

import sys
import os
import logging
from typing import Dict, Any
from langchain.prompts import PromptTemplate

from ProductDesigner.graph_state import GraphState
from ProductDesigner.agents.base_agent import BaseAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FinalAssemblerAgent(BaseAgent):
    """
    Acts as the Packager, bundling all generated artifacts into a final,
    clean output for the user.
    """
    
    def __init__(self):
        """Initialize the Final Assembler Agent."""
        super().__init__()
    
    def run(self, state: GraphState) -> Dict[str, Any]:
        """Bundle all artifacts into a final package."""
        logging.info("---ASSEMBLING FINAL PACKAGE---")
        
        prompt_template_str = (
            "You are the Final Assembler responsible for packaging all project artifacts "
            "into a clean, organized final output for the user.\n"
            "Review the following artifacts:\n"
            "PRD Document:\n{prd_document}\n"
            "Technical Design Document:\n{tech_design_document}\n"
            "Implementation Notes:\n{actionable_notes}\n"
            "Testing Plan:\n{testing_plan}\n"
            "Create a brief executive summary (README.md) that ties all these documents together "
            "and explains how they form a complete project blueprint. This summary should help the "
            "user understand how to use these documents effectively to implement their project."
        )
        
        prompt_template = PromptTemplate.from_template(prompt_template_str)
        
        params = {
            "prd_document": state.prd_document,
            "tech_design_document": state.tech_design_document,
            "actionable_notes": state.actionable_notes,
            "testing_plan": state.testing_plan
        }

        executive_summary = self.llm_service.execute(
            prompt_template,
            params,
            fallback_response="Failed to generate the executive summary."
        )
        
        final_package = {
            "README.md": executive_summary,
            "PRD.md": state.prd_document,
            "TechDesignDoc.md": state.tech_design_document,
            "NOTES.md": state.actionable_notes,
            "TESTING_PLAN.md": state.testing_plan
        }
        
        logging.info("Final package assembled successfully.")
        return {"final_package": final_package}


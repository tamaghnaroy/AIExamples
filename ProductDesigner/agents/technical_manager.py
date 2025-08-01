import sys
import os
import logging
from typing import Dict, Any
from langchain.prompts import PromptTemplate

from ProductDesigner.graph_state import GraphState
from ProductDesigner.agents.base_agent import BaseAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TechnicalManagerAgent(BaseAgent):
    """
    Acts as a meticulous Technical Project Manager, synthesizing the approved PRD and TDD
    into a step-by-step implementation guide (NOTES.md) for a code-generation LLM.
    """
    
    def __init__(self):
        """Initialize the Technical Manager Agent."""
        super().__init__()
    
    def run(self, state: GraphState) -> Dict[str, Any]:
        """Generate the actionable notes based on the PRD and TDD."""
        logging.info("---GENERATE ACTIONABLE NOTES (IMPLEMENTATION GUIDE)---")
        
        prompt_template_str = (
            "You are a meticulous Technical Project Manager tasked with creating "
            "a detailed step-by-step implementation guide based on approved PRD and TDD documents.\n"
            "PRD Content:\n{prd_document}\n"
            "Technical Design Document:\n{tech_design_document}\n"
            "Your task is to synthesize these documents into an actionable NOTES.md file "
            "that serves as a precise implementation guide for a code-generation LLM. "
            "Include step-by-step project structure creation with shell commands, file-by-file population "
            "instructions with exact code, and final integration steps. Be extremely precise and detailed."
        )
        
        prompt_template = PromptTemplate.from_template(prompt_template_str)
        
        params = {
            "prd_document": state.prd_document,
            "tech_design_document": state.tech_design_document
        }

        actionable_notes = self.llm_service.execute(
            prompt_template,
            params,
            fallback_response="Failed to generate the actionable notes."
        )
        
        logging.info("Actionable notes generated successfully.")
        return {"actionable_notes": actionable_notes}


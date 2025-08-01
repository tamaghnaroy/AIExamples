import sys
import os
import logging
from typing import Dict, Any
from langchain.prompts import PromptTemplate

from ProductDesigner.graph_state import GraphState
from ProductDesigner.agents.base_agent import BaseAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestDeveloperAgent(BaseAgent):
    """
    Acts as a dedicated QA Engineer focused on Test-Driven Development (TDD),
    generating a comprehensive testing plan based on the PRD and implementation details.
    """
    
    def __init__(self):
        """Initialize the Test Developer Agent."""
        super().__init__()
    
    def run(self, state: GraphState) -> Dict[str, Any]:
        """Generate the testing plan based on the PRD, TDD, and implementation notes."""
        logging.info("---GENERATE TESTING PLAN---")
        
        prompt_template_str = (
            "You are a dedicated QA Engineer focused on Test-Driven Development. "
            "Your task is to create a comprehensive testing plan based on the PRD, "
            "Technical Design Document, and implementation details.\n"
            "PRD Content:\n{prd_document}\n"
            "Technical Design Document:\n{tech_design_document}\n"
            "Implementation Notes:\n{actionable_notes}\n"
            "Create a detailed TESTING_PLAN.md document that includes sections for Testing Frameworks, "
            "Unit Tests, Integration Tests, End-to-End (E2E) Tests, and Test Data Strategy. "
            "The plan should be comprehensive and provide clear guidance for implementation."
        )
        
        prompt_template = PromptTemplate.from_template(prompt_template_str)
        
        params = {
            "prd_document": state.prd_document,
            "tech_design_document": state.tech_design_document,
            "actionable_notes": state.actionable_notes
        }

        testing_plan = self.llm_service.execute(
            prompt_template,
            params,
            fallback_response="Failed to generate the testing plan."
        )
        
        logging.info("Testing plan generated successfully.")
        return {"testing_plan": testing_plan}


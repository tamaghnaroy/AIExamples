import sys
import os
import logging
from typing import Dict, Any
from langchain.prompts import PromptTemplate

from ProductDesigner.graph_state import GraphState
from ProductDesigner.agents.base_agent import BaseAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TechDesignGeneratorAgent(BaseAgent):
    """
    Acts as a Senior Full-Stack Architect, creating a highly detailed Technical Design Document
    that is verbose, unambiguous, and complete.
    """
    
    def __init__(self):
        """Initialize the Technical Design Generator Agent."""
        super().__init__()
    
    def _format_qa_history(self, qna_history: Dict[str, Any]) -> str:
        """Format the Q&A history for display in the prompt."""
        if not qna_history:
            return "No questions were answered in the interview process."
        
        formatted = ""
        for idx, (question, data) in enumerate(qna_history.items()):
            answer = data.get('refined_answer', data.get('llm_recommendation', 'No answer provided.'))
            formatted += f"Q{idx+1}: {question}\n"
            formatted += f"A{idx+1}: {answer}\n\n"
        
        return formatted.strip()
    
    def run(self, state: GraphState) -> Dict[str, Any]:
        """Generate the Technical Design Document based on the PRD and interview Q&A."""
        logging.info("---GENERATE TECHNICAL DESIGN DOCUMENT---")
        
        qa_history_formatted = self._format_qa_history(state.qna_history)
        
        prompt_template_str = (
            "You are a Senior Full-Stack Architect tasked with creating a comprehensive "
            "Technical Design Document (TDD) based on a PRD and product interview Q&A.\n"
            "Initial idea: {initial_idea}\n"
            "Q&A Interview Results:\n{qa_history_formatted}\n"
            "PRD Content:\n{prd_document}\n"
            "User feedback (if any):\n{user_feedback}\n"
            "Your task is to create an extremely detailed TDD in markdown format. "
            "Be verbose, unambiguous, and complete. Include sections like Tech Stack Specification, "
            "Architecture, API and Data Modeling, Component & Logic Breakdown, and Detailed Implementation Strategies. "
            "This document should be so detailed that a developer could implement it directly."
        )
        
        prompt_template = PromptTemplate.from_template(prompt_template_str)
        
        params = {
            "initial_idea": state.initial_idea,
            "qa_history_formatted": qa_history_formatted,
            "prd_document": state.prd_document,
            "user_feedback": state.user_feedback
        }

        tech_design_content = self.llm_service.execute(
            prompt_template,
            params,
            fallback_response="Failed to generate the Technical Design Document."
        )
        
        logging.info("Technical Design Document generated successfully.")
        return {"tech_design_document": tech_design_content}


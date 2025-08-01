import sys
import os
import logging
from typing import Dict, Any
from langchain.prompts import PromptTemplate

from ProductDesigner.graph_state import GraphState
from ProductDesigner.agents.base_agent import BaseAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PRDGeneratorAgent(BaseAgent):
    """
    Acts as the Product Owner, transforming structured Q&A data 
    into a formal Product Requirements Document (PRD) for the MVP.
    """
    
    def __init__(self):
        """Initialize the PRD Generator Agent."""
        super().__init__()
    
    def _format_qa_history(self, qna_history: Dict[str, Any]) -> str:
        """Format the Q&A history for display in the prompt."""
        if not qna_history:
            return "No questions were answered in the interview process."
        
        formatted = ""
        for idx, (question, data) in enumerate(qna_history.items()):
            # Extract the final answer from the nested dictionary
            answer = data.get('refined_answer', data.get('llm_recommendation', 'No answer provided.'))
            formatted += f"Q{idx+1}: {question}\n"
            formatted += f"A{idx+1}: {answer}\n\n"
        
        return formatted.strip()
    
    def run(self, state: GraphState) -> Dict[str, Any]:
        """Generate the PRD based on the interview Q&A."""
        logging.info("---GENERATE PRD DOCUMENT---")
        
        qa_history_formatted = self._format_qa_history(state.qna_history)
        
        prompt_template_str = (
            "You are an expert Product Owner tasked with creating a formal "
            "Product Requirements Document (PRD) for an MVP based on a Q&A interview.\n"
            "Initial idea: {initial_idea}\n"
            "Q&A Interview Results:\n{qa_history_formatted}\n"
            "User feedback (if any):\n{user_feedback}\n"
            "Your task is to create a comprehensive PRD document in markdown format. "
            "The document should include sections like Executive Summary, Product Vision, "
            "Target Audience, User Stories, Technical Requirements, and Success Metrics. "
            "Focus on clarity, completeness, and providing a solid foundation for technical design.\n"
            "Generate the complete PRD.md file content:"
        )
        
        prompt_template = PromptTemplate.from_template(prompt_template_str)
        
        params = {
            "initial_idea": state.initial_idea,
            "qa_history_formatted": qa_history_formatted,
            "user_feedback": state.user_feedback
        }

        prd_content = self.llm_service.execute(
            prompt_template,
            params,
            fallback_response="Failed to generate the PRD document."
        )
        
        logging.info("PRD document generated successfully.")
        return {"prd_document": prd_content}


import sys
import os
from typing import Dict, Any
from langchain.prompts import PromptTemplate

# Add project root to path for imports to support different execution contexts
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Robustly import GraphState and LLMService
from ProductDesigner.core.llm_service import LLMService
from ProductDesigner.graph_state import GraphState

class BaseAgent:
    """Base class for all agents, now using a centralized LLMService."""

    def __init__(self):
        """
        Initialize the base agent.
        It now uses LLMService for all language model interactions.
        """
        self.llm_service = LLMService()
        # Prompt template can be overridden by subclasses
        self.prompt_template = PromptTemplate.from_template("")

    def _execute_llm_call(self, state: GraphState, prompt_template: PromptTemplate, fallback_response: str = "Error processing request.") -> str:
        """
        A wrapper to execute the LLM call using the LLMService.
        This centralizes the execution logic for all agents.

        Args:
            state (GraphState): The current graph state, converted to a dict for the prompt.
            prompt_template (PromptTemplate): The specific prompt template for the call.
            fallback_response (str): The response to return if the LLM call fails.

        Returns:
            str: The LLM's response or the fallback response.
        """
        params = state.dict()
        return self.llm_service.execute(prompt_template, params, fallback_response)

    def run(self, state: GraphState) -> Dict[str, Any]:
        """
        Process the current state and return an updated state dictionary.
        This method must be implemented by all subclasses.

        Args:
            state (GraphState): The current state of the workflow.

        Returns:
            Dict[str, Any]: A dictionary containing the updated parts of the state.
        """
        raise NotImplementedError("Subclasses must implement the 'run' method.")


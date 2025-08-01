"""
Tests for the BaseAgent class.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agents.base_agent import BaseAgent
from graph_state import GraphState


class TestBaseAgent:
    """Test cases for BaseAgent functionality."""
    
    def test_initialization_default_params(self, mock_openai_client):
        """Test BaseAgent initialization with default parameters."""
        agent = BaseAgent()
        
        # Test that the agent has an LLM instance
        assert agent.llm is not None
        assert hasattr(agent, 'prompt_template')
    
    def test_initialization_custom_params(self, mock_openai_client):
        """Test BaseAgent initialization with custom parameters."""
        agent = BaseAgent(model="gpt-3.5-turbo", temperature=0.5)
        
        # Test that the agent was created successfully
        assert agent.llm is not None
        assert hasattr(agent, 'prompt_template')
    
    def test_setup_chain(self, mock_openai_client):
        """Test that setup_chain creates a processing chain."""
        agent = BaseAgent()
        chain = agent.setup_chain()
        
        # Test that a chain was created
        assert chain is not None
    
    def test_run_method_not_implemented(self, mock_openai_client):
        """Test that run method raises NotImplementedError in base class."""
        agent = BaseAgent()
        test_state = {
            "initial_idea": "Test idea",
            "qna_history": {},
            "prd_document": "",
            "tech_design_document": "",
            "user_feedback": "",
            "is_docs_approved": False,
            "actionable_notes": "",
            "testing_plan": "",
            "final_package": {}
        }
        
        with pytest.raises(NotImplementedError):
            agent.run(test_state)
    
    def test_multiple_agents_independence(self, mock_openai_client):
        """Test that multiple agent instances are independent."""
        agent1 = BaseAgent(model="gpt-4", temperature=0.1)
        agent2 = BaseAgent(model="gpt-3.5-turbo", temperature=0.9)
        
        # Test that agents have different LLM instances
        assert agent1.llm != agent2.llm
    
    @patch('agents.base_agent.ChatOpenAI')
    def test_llm_initialization_failure(self, mock_chat_openai):
        """Test handling of LLM initialization failure."""
        mock_chat_openai.side_effect = Exception("Failed to initialize LLM")
        
        with pytest.raises(Exception) as exc_info:
            BaseAgent()
        
        assert "Failed to initialize LLM" in str(exc_info.value)

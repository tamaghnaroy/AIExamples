"""
Pytest configuration and shared fixtures for Deep Planning tests.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables for testing
load_dotenv()

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client to avoid real API calls."""
    with patch('os.environ.get') as mock_env:
        mock_env.return_value = 'test-api-key'
        
        # Create a comprehensive mock for the chain invoke method
        mock_response = Mock()
        mock_response.content = "Mocked LLM response"
        
        # Mock at the chain level - patch the invoke method directly on any chain
        with patch('langchain_core.runnables.base.Runnable.invoke') as mock_invoke:
            mock_invoke.return_value = "Mocked LLM response"
            yield mock_invoke

@pytest.fixture
def mock_search_tool():
    """Mock search functionality for testing without web requests."""
    # Create a simple mock that can be used in place of any search tool
    mock_search = Mock()
    mock_search.run.return_value = "Mock search results for testing"
    
    # Patch multiple possible search imports
    with patch('agents.product_interviewer.DuckDuckGoSearchRun', return_value=mock_search), \
         patch('duckduckgo_search.DDGS', return_value=mock_search), \
         patch('langchain_community.tools.DuckDuckGoSearchRun', return_value=mock_search):
        yield mock_search

@pytest.fixture
def sample_project_idea():
    """Sample project idea for testing."""
    return "A web application that helps users track their daily habits and visualize progress over time with beautiful charts and reminders."

@pytest.fixture
def sample_qa_responses():
    """Sample Q&A responses for testing."""
    return {
        "target_audience": "Health-conscious individuals aged 25-45 who want to build better habits",
        "core_features": "Habit tracking, progress visualization, reminders, streak tracking, analytics",
        "technical_requirements": "Web application with responsive design, data persistence, notifications",
        "success_metrics": "User engagement, habit completion rates, retention metrics",
        "timeline": "3-month development cycle with MVP in 6 weeks"
    }

@pytest.fixture
def sample_graph_state():
    """Sample graph state for testing."""
    from graph_state import GraphState
    
    return GraphState(
        initial_idea="Test project idea",
        current_step="product_interviewer",
        qa_responses={},
        prd_content="",
        tech_design_content="",
        implementation_notes="",
        testing_plan="",
        final_package={},
        user_feedback="",
        needs_human_input=False,
        step_completed=False,
        error_message=""
    )

@pytest.fixture
def mock_file_operations():
    """Mock file operations for testing."""
    with patch('builtins.open', create=True) as mock_open, \
         patch('os.makedirs') as mock_makedirs, \
         patch('os.path.exists') as mock_exists:
        
        mock_exists.return_value = True
        yield {
            'open': mock_open,
            'makedirs': mock_makedirs,
            'exists': mock_exists
        }

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables."""
    test_env = {
        'OPENAI_API_KEY': 'test-api-key',
        'MODEL_NAME': 'gpt-4-turbo',
        'TEMPERATURE': '0.1'
    }
    
    with patch.dict(os.environ, test_env):
        yield

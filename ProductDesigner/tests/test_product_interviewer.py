"""
Tests for the SafeProductInterviewerAgent class.
"""

import pytest
from unittest.mock import patch, MagicMock
from agents.safe_product_interviewer import SafeProductInterviewerAgent

# Mock state for use in tests
@pytest.fixture
def mock_state():
    return {"initial_idea": "A smart umbrella that predicts rain", "qna_history": {}}

class TestSafeProductInterviewerAgent:
    """Test cases for SafeProductInterviewerAgent functionality."""

    def test_initialization(self):
        """Test SafeProductInterviewerAgent initialization."""
        agent = SafeProductInterviewerAgent()
        assert hasattr(agent, 'llm_service')
        assert len(agent.questions) == 13
        assert agent.current_question_idx == 0
        assert agent.web_search_enabled is not None

    def test_get_planning_questions(self):
        """Test that all 13 planning questions are properly defined."""
        agent = SafeProductInterviewerAgent()
        questions = agent._get_planning_questions()
        assert len(questions) == 13
        assert all(isinstance(q, str) for q in questions)

    @patch('agents.safe_product_interviewer.DuckDuckGoSearchRun.run')
    def test_search_web_for_context(self, mock_search_run):
        """Test web search functionality."""
        mock_search_run.return_value = "Mock search results"
        agent = SafeProductInterviewerAgent()
        results = agent.search_web_for_context("test query")
        assert results == "Mock search results"
        mock_search_run.assert_called_once_with("test query")

    @patch('agents.safe_product_interviewer.SafeProductInterviewerAgent._run_llm_step')
    def test_generate_initial_answer(self, mock_llm_step, mock_state):
        """Test initial answer generation using state dictionary."""
        mock_llm_step.return_value = "Initial answer"
        agent = SafeProductInterviewerAgent()
        answer = agent.generate_initial_answer(mock_state, "Test question", "Search results")
        assert answer == "Initial answer"
        mock_llm_step.assert_called_once()

    @patch('agents.safe_product_interviewer.SafeProductInterviewerAgent._run_llm_step')
    def test_critique_answer(self, mock_llm_step, mock_state):
        """Test answer critique functionality using state dictionary."""
        mock_llm_step.return_value = "This is a critique."
        agent = SafeProductInterviewerAgent()
        critique = agent.critique_answer(mock_state, "Test question", "Initial answer")
        assert critique == "This is a critique."
        mock_llm_step.assert_called_once()

    @patch('agents.safe_product_interviewer.SafeProductInterviewerAgent._run_llm_step')
    def test_refine_answer(self, mock_llm_step, mock_state):
        """Test answer refinement using state dictionary."""
        mock_llm_step.return_value = "Refined answer"
        agent = SafeProductInterviewerAgent()
        refined = agent.refine_answer(mock_state, "Test question", "Initial answer", "Critique")
        assert refined == "Refined answer"
        mock_llm_step.assert_called_once()

    @patch('agents.safe_product_interviewer.SafeProductInterviewerAgent._run_llm_step')
    def test_present_to_user(self, mock_llm_step, mock_state):
        """Test user presentation formatting using state dictionary."""
        mock_llm_step.return_value = "User-friendly presentation"
        agent = SafeProductInterviewerAgent()
        presentation = agent.present_to_user(mock_state, "Test question", "Refined answer")
        assert presentation == "User-friendly presentation"
        mock_llm_step.assert_called_once()

    @patch('agents.safe_product_interviewer.SafeProductInterviewerAgent.search_web_for_context')
    @patch('agents.safe_product_interviewer.SafeProductInterviewerAgent.generate_initial_answer')
    @patch('agents.safe_product_interviewer.SafeProductInterviewerAgent.critique_answer')
    @patch('agents.safe_product_interviewer.SafeProductInterviewerAgent.refine_answer')
    @patch('agents.safe_product_interviewer.SafeProductInterviewerAgent.present_to_user')
    def test_process_question_with_self_critique_flow(self, mock_present, mock_refine, mock_critique, mock_generate, mock_search, mock_state):
        """Test the complete self-critique process flow with state dictionary."""
        mock_search.return_value = "Search results"
        mock_generate.return_value = "Initial answer"
        mock_critique.return_value = "Critique"
        mock_refine.return_value = "Refined answer"
        mock_present.return_value = "User presentation"

        agent = SafeProductInterviewerAgent()
        result = agent.process_question_with_self_critique(mock_state, "Test question")

        mock_search.assert_called_once()
        mock_generate.assert_called_once_with(mock_state, "Test question", "Search results")
        mock_critique.assert_called_once_with(mock_state, "Test question", "Initial answer")
        mock_refine.assert_called_once_with(mock_state, "Test question", "Initial answer", "Critique")
        mock_present.assert_called_once_with(mock_state, "Test question", "Refined answer")

        assert result["refined_answer"] == "Refined answer"
        assert result["user_presentation"] == "User presentation"

    @patch('agents.safe_product_interviewer.SafeProductInterviewerAgent.process_question_with_self_critique')
    def test_run_method(self, mock_process_question, mock_state):
        """Test the run method's logic."""
        mock_process_question.return_value = {
            "refined_answer": "Final refined answer",
            "user_presentation": "Final user presentation",
            "search_results": "Final search results",
            "critique": "Final critique",
        }

        agent = SafeProductInterviewerAgent()
        result = agent.run(mock_state)

        assert 'qna_history' in result
        assert len(result['qna_history']) == 1
        question = agent.questions[0]
        assert result['qna_history'][question]['llm_recommendation'] == "Final refined answer"
        mock_process_question.assert_called_once_with(mock_state, question)

    def test_run_method_completes_all_questions(self, mock_state):
        """Test that the run method iterates through all questions and terminates."""
        agent = SafeProductInterviewerAgent()
        # Set current_question_idx to the last question
        agent.current_question_idx = len(agent.questions)
        result = agent.run(mock_state)
        assert result == {"qna_history": {}}

    @patch('agents.safe_product_interviewer.SafeProductInterviewerAgent.process_question_with_self_critique')
    def test_run_method_error_handling(self, mock_process_question, mock_state):
        """Test error handling in the run method."""
        mock_process_question.return_value = {"error": "An unexpected error occurred"}
        agent = SafeProductInterviewerAgent()
        result = agent.run(mock_state)
        # Should return the initial empty history on error
        assert result == {"qna_history": {}}

    @patch('threading.Thread')
    @patch('agents.safe_product_interviewer.SafeProductInterviewerAgent.get_question_recommendation')
    def test_prefetch_next_recommendation_starts_thread(self, mock_get_recommendation, mock_thread, mock_state):
        """Test that prefetching starts a background thread for the next question."""
        agent = SafeProductInterviewerAgent()
        agent.prefetch_next_recommendation(1, mock_state)

        mock_thread.assert_called_once()
        thread_instance = mock_thread.return_value
        thread_instance.start.assert_called_once()

        # To verify the thread's action, we can execute the target function
        call_kwargs = mock_thread.call_args.kwargs
        prefetch_task = call_kwargs['target']
        prefetch_task()

        mock_get_recommendation.assert_called_once_with(2, mock_state, progress_callback=None)

    @patch('threading.Thread')
    def test_prefetch_skips_if_last_question(self, mock_thread, mock_state):
        """Test that prefetching is skipped if it's the last question."""
        agent = SafeProductInterviewerAgent()
        last_question_num = len(agent.questions)
        agent.prefetch_next_recommendation(last_question_num, mock_state)
        mock_thread.assert_not_called()

    @patch('threading.Thread')
    def test_prefetch_skips_if_already_cached(self, mock_thread, mock_state):
        """Test that prefetching is skipped if the recommendation is already cached."""
        agent = SafeProductInterviewerAgent()
        with patch.object(agent, 'is_recommendation_ready', return_value=True):
            agent.prefetch_next_recommendation(1, mock_state)
            mock_thread.assert_not_called()

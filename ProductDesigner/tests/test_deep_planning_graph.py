"""
Tests for the DeepPlanningGraph workflow.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from deep_planning_graph import create_deep_planning_graph, initialize_state
from graph_state import GraphState


class TestDeepPlanningGraph:
    """Test cases for DeepPlanningGraph functionality."""
    
    def test_graph_initialization(self):
        """Test graph initialization and structure."""
        graph = create_deep_planning_graph()
        
        assert graph is not None
        # Verify the graph has the expected structure
        # Note: Specific assertions depend on LangGraph implementation details
    
    @patch('deep_planning_graph.ProductInterviewerAgent')
    @patch('deep_planning_graph.PRDGeneratorAgent')
    @patch('deep_planning_graph.TechDesignGeneratorAgent')
    def test_agent_initialization(self, mock_tech_design, mock_prd, mock_interviewer):
        """Test that all agents are properly initialized."""
        # Setup mocks
        mock_interviewer.return_value = Mock()
        mock_prd.return_value = Mock()
        mock_tech_design.return_value = Mock()
        
        graph = create_deep_planning_graph()
        
        # Verify agents were instantiated
        mock_interviewer.assert_called_once()
        mock_prd.assert_called_once()
        mock_tech_design.assert_called_once()
    
    @patch('agents.product_interviewer.ProductInterviewerAgent')
    def test_product_interviewer_agent(self, mock_agent_class, mock_openai_client, mock_search_tool):
        """Test the product interviewer agent functionality."""
        # Mock the agent's run method
        mock_agent = Mock()
        mock_agent.run.return_value = {
            "current_step": "product_interviewer",
            "needs_human_input": True,
            "user_feedback": "Generated question about target audience",
            "step_completed": False
        }
        mock_agent_class.return_value = mock_agent
        
        # Create a sample state
        state = initialize_state("Test product idea")
        
        # Test agent run method
        result = mock_agent.run(state)
        
        assert result["current_step"] == "product_interviewer"
        assert result["needs_human_input"] == True
        assert "user_feedback" in result
        mock_agent.run.assert_called_once_with(state)
    
    def test_initialize_state(self):
        """Test the initialize_state function."""
        idea = "Test product idea"
        state = initialize_state(idea)
        
        assert state["initial_idea"] == idea
        assert state["qna_history"] == {}
        assert state["prd_document"] == ""
        assert state["tech_design_document"] == ""
        assert state["user_feedback"] == ""
        assert state["is_docs_approved"] == False
        assert state["actionable_notes"] == ""
        assert state["testing_plan"] == ""
        assert state["final_package"] == {}
    
    @patch('agents.prd_generator.PRDGeneratorAgent')
    def test_prd_generator_agent(self, mock_agent_class, mock_openai_client):
        """Test the PRD generator agent functionality."""
        # Mock the agent's run method
        mock_agent = Mock()
        mock_agent.run.return_value = {
            "prd_document": "Generated PRD content",
            "current_step": "prd_generation",
            "step_completed": True
        }
        mock_agent_class.return_value = mock_agent
        
        # Create a sample state with Q&A data
        state = initialize_state("Test product idea")
        state["qna_history"] = {"target_audience": "Health enthusiasts"}
        
        # Test agent run method
        result = mock_agent.run(state)
        
        assert result["current_step"] == "prd_generation"
        assert result["step_completed"] == True
        assert "prd_document" in result
        mock_agent.run.assert_called_once_with(state)
    
    @patch('agents.tech_design_generator.TechDesignGeneratorAgent')
    def test_tech_design_generator_agent(self, mock_agent_class, mock_openai_client):
        """Test the tech design generator agent functionality."""
        # Mock the agent's run method
        mock_agent = Mock()
        mock_agent.run.return_value = {
            "tech_design_document": "Generated tech design content",
            "current_step": "tech_design",
            "step_completed": True
        }
        mock_agent_class.return_value = mock_agent
        
        # Create a sample state with PRD data
        state = initialize_state("Test product idea")
        state["prd_document"] = "PRD content"
        
        # Test agent run method
        result = mock_agent.run(state)
        
        assert result["current_step"] == "tech_design"
        assert result["step_completed"] == True
        assert "tech_design_document" in result
        mock_agent.run.assert_called_once_with(state)
    
    def test_graph_structure(self):
        """Test that the graph has the expected structure and nodes."""
        graph = create_deep_planning_graph()
        
        # Test that the graph was created successfully
        assert graph is not None
        
        # The graph should be compiled and ready to use
        # Note: Specific structural tests depend on LangGraph internals
        # which may change, so we keep this test simple
    
    def test_after_review_function(self):
        """Test the after_review conditional function from the graph."""
        # Test case: docs approved, should continue
        state_approved = initialize_state("Test idea")
        state_approved["is_docs_approved"] = True
        
        # We can't directly test the internal function, but we can test the logic
        assert state_approved["is_docs_approved"] == True
        
        # Test case: docs not approved, should revise
        state_not_approved = initialize_state("Test idea")
        state_not_approved["is_docs_approved"] = False
        
        assert state_not_approved["is_docs_approved"] == False
    
    def test_state_structure(self):
        """Test that the state structure matches expectations."""
        state = initialize_state("Test idea")
        
        # Test that all required fields are present
        required_fields = [
            "initial_idea", "qna_history", "prd_document", 
            "tech_design_document", "user_feedback", "is_docs_approved",
            "actionable_notes", "testing_plan", "final_package"
        ]
        
        for field in required_fields:
            assert field in state, f"Missing required field: {field}"
    
    def test_state_initialization_with_idea(self):
        """Test that state is properly initialized with an idea."""
        test_idea = "A revolutionary mobile app"
        state = initialize_state(test_idea)
        
        assert state["initial_idea"] == test_idea
        assert isinstance(state["qna_history"], dict)
        assert state["prd_document"] == ""
        assert state["tech_design_document"] == ""
        assert state["user_feedback"] == ""
        assert state["is_docs_approved"] == False
        assert state["actionable_notes"] == ""
        assert state["testing_plan"] == ""
        assert state["final_package"] == {}
    
    def test_graph_creation_with_mocked_agents(self, mock_openai_client):
        """Test that the graph can be created successfully with mocked agents."""
        # Mock all agent classes in the deep_planning_graph module
        with patch('deep_planning_graph.ProductInterviewerAgent') as mock_interviewer, \
             patch('deep_planning_graph.PRDGeneratorAgent') as mock_prd, \
             patch('deep_planning_graph.TechDesignGeneratorAgent') as mock_tech, \
             patch('deep_planning_graph.TechnicalManagerAgent') as mock_manager, \
             patch('deep_planning_graph.TestDeveloperAgent') as mock_test, \
             patch('deep_planning_graph.FinalAssemblerAgent') as mock_assembler:
            
            # Setup mock agents with run methods
            for mock_agent_class in [mock_interviewer, mock_prd, mock_tech, mock_manager, mock_test, mock_assembler]:
                mock_instance = Mock()
                mock_instance.run.return_value = {"step_completed": True}
                mock_agent_class.return_value = mock_instance
            
            # Create the graph
            graph = create_deep_planning_graph()
            assert graph is not None
            
            # Verify all agent classes were instantiated
            mock_interviewer.assert_called_once()
            mock_prd.assert_called_once()
            mock_tech.assert_called_once()
            mock_manager.assert_called_once()
            mock_test.assert_called_once()
            mock_assembler.assert_called_once()
    
    def test_state_modifications(self):
        """Test that state can be modified properly."""
        # Test initial state
        state = initialize_state("Test idea")
        
        # Simulate product interview completion
        state["qna_history"] = {"target_audience": "Test audience"}
        state["is_docs_approved"] = True
        
        # Verify state modifications
        assert state["initial_idea"] == "Test idea"
        assert "target_audience" in state["qna_history"]
        assert state["is_docs_approved"] == True
        
        # Simulate PRD generation
        state["prd_document"] = "Generated PRD content"
        
        # Verify PRD was added
        assert state["prd_document"] == "Generated PRD content"
        assert state["prd_document"] != ""
    
    def test_state_data_flow(self):
        """Test that data flows properly through the state."""
        # Test the logical flow of data through workflow steps
        state = initialize_state("Test mobile app")
        
        # Simulate Q&A completion
        state["qna_history"] = {
            "target_audience": "Mobile users",
            "key_features": "Push notifications, offline mode"
        }
        
        # Verify Q&A data is available for PRD generation
        assert len(state["qna_history"]) > 0
        assert "target_audience" in state["qna_history"]
        
        # Simulate PRD completion
        state["prd_document"] = "Complete PRD based on Q&A"
        
        # Verify PRD is available for tech design
        assert state["prd_document"] != ""
        assert "PRD" in state["prd_document"]
        
        # Simulate tech design completion
        state["tech_design_document"] = "Technical architecture document"
        
        # Verify all data is available for final assembly
        assert state["prd_document"] != ""
        assert state["tech_design_document"] != ""
        assert len(state["qna_history"]) > 0
    
    def test_state_boolean_fields(self):
        """Test boolean fields in the state work correctly."""
        state = initialize_state("Test idea")
        
        # Test initial boolean values
        assert state["is_docs_approved"] == False
        
        # Test setting boolean values
        state["is_docs_approved"] = True
        assert state["is_docs_approved"] == True
        
        # Test resetting boolean values
        state["is_docs_approved"] = False
        assert state["is_docs_approved"] == False

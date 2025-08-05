#!/usr/bin/env python3
"""
Test script for the ResearchSubgraphAgent.
This script demonstrates how to use the new research agent.
"""

import sys
import os
import asyncio
import logging

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.research_subgraph_agent import ResearchSubgraphAgent
from graph_state import GraphState

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_research_agent():
    """Test the ResearchSubgraphAgent with a sample product idea."""
    
    # Create the research agent
    research_agent = ResearchSubgraphAgent()
    
    # Create a sample state with an initial idea
    sample_idea = "A mobile app that helps users track their daily water intake and sends personalized hydration reminders based on their activity level, weather, and health goals"
    
    # Create GraphState object (assuming it has an initial_idea attribute)
    state = GraphState()
    state.initial_idea = sample_idea
    
    print(f"Testing ResearchSubgraphAgent with idea: {sample_idea}")
    print("=" * 80)
    
    try:
        # Run the research agent
        result = research_agent.run(state)
        
        print("\n" + "=" * 80)
        print("RESEARCH RESULTS:")
        print("=" * 80)
        
        if "error" in result:
            print(f"Error occurred: {result['error']}")
        else:
            print(f"Research completed: {result.get('research_completed', False)}")
            
            if "research_results" in result:
                research_data = result["research_results"]
                print(f"Research depth: {research_data.get('research_depth', 'N/A')}")
                print(f"Total learnings: {research_data.get('total_learnings', 'N/A')}")
                print(f"Initial queries count: {len(research_data.get('initial_queries', []))}")
                
            if "research_knowledge_base" in result:
                print("\nKNOWLEDGE BASE:")
                print("-" * 40)
                print(result["research_knowledge_base"][:500] + "..." if len(result["research_knowledge_base"]) > 500 else result["research_knowledge_base"])
                
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_individual_agents():
    """Test individual sub-agents separately."""
    
    research_agent = ResearchSubgraphAgent()
    sample_idea = "A task management app with AI-powered priority suggestions"
    
    print("Testing individual sub-agents:")
    print("=" * 50)
    
    try:
        # Test search query generation
        print("1. Testing search query generation...")
        websearch = research_agent.generate_search_queries_agent(sample_idea)
        print(f"Generated {len(websearch.search_queries)} queries")
        
        # Test knowledge base generation with mock data
        print("\n2. Testing knowledge base generation...")
        mock_learnings = [
            "Topic: Task management apps\nResult: Popular apps include Todoist, Asana, Trello",
            "Topic: AI priority suggestions\nResult: Machine learning can analyze task patterns"
        ]
        knowledge_base = research_agent.generate_knowledge_base_agent(sample_idea, mock_learnings)
        print(f"Knowledge base generated (length: {len(knowledge_base)} chars)")
        
        print("\nIndividual agent tests completed successfully!")
        
    except Exception as e:
        print(f"Individual agent test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("ResearchSubgraphAgent Test Suite")
    print("=" * 50)
    
    # Test individual agents first (safer)
    test_individual_agents()
    
    print("\n" + "=" * 50)
    print("Note: Full integration test requires OpenAI API key and web search capabilities.")
    print("To run full test, ensure OPENAI_API_KEY environment variable is set.")
    
    # Uncomment the line below to run full integration test
    # test_research_agent()

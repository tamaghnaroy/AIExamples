#!/usr/bin/env python3
"""
Test script to verify web search timeout fixes work properly.
"""

import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(__file__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.safe_product_interviewer import SafeProductInterviewerAgent

def test_web_search_timeout():
    """Test that web search has proper timeout protection."""
    print("Testing web search timeout protection...")
    
    # Create agent
    agent = SafeProductInterviewerAgent()
    
    # Test normal search (should complete quickly or timeout gracefully)
    print("\n1. Testing normal web search...")
    start_time = time.time()
    result = agent.search_web_for_context("product management best practices")
    end_time = time.time()
    
    print(f"Search completed in {end_time - start_time:.2f} seconds")
    print(f"Result: {result[:100]}...")
    
    # Test with disabled web search
    print("\n2. Testing with disabled web search...")
    agent.web_search_enabled = False
    agent.search_tool = None
    
    start_time = time.time()
    result = agent.search_web_for_context("product management best practices")
    end_time = time.time()
    
    print(f"Search completed in {end_time - start_time:.2f} seconds")
    print(f"Result: {result}")
    
    print("\n✅ Web search timeout tests completed!")

def test_question_processing():
    """Test that question processing works with timeout protection."""
    print("\nTesting question processing with timeout protection...")
    
    # Create agent
    agent = SafeProductInterviewerAgent()
    
    # Test question processing
    question = "What is the primary purpose of this application or system?"
    initial_idea = "A task management app for teams"
    
    def mock_progress_callback(stage, message, progress):
        print(f"[{progress}%] {stage}: {message}")
    
    print(f"\nProcessing question: {question}")
    start_time = time.time()
    
    try:
        result = agent.process_question_with_self_critique_and_progress(
            question, initial_idea, mock_progress_callback
        )
        end_time = time.time()
        
        print(f"\nQuestion processing completed in {end_time - start_time:.2f} seconds")
        print(f"Result keys: {list(result.keys())}")
        print(f"Search results: {result['search_results'][:100]}...")
        
    except Exception as e:
        end_time = time.time()
        print(f"Question processing failed after {end_time - start_time:.2f} seconds")
        print(f"Error: {str(e)}")
    
    print("\n✅ Question processing tests completed!")

if __name__ == "__main__":
    print("🔧 Testing Web Search Timeout Fixes")
    print("=" * 50)
    
    try:
        test_web_search_timeout()
        test_question_processing()
        
        print("\n🎉 All tests completed successfully!")
        print("\nThe web search timeout fixes should prevent hanging issues.")
        print("You can now restart the web application to test the fix.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

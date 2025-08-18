"""
Basic usage examples for the DeepResearchClient class-based interface.

This file demonstrates various ways to use the Deep Research system programmatically
from other Python applications.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import deepresearcher
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepresearcher.api import DeepResearchClient
from deepresearcher.api.research_client import ResearchConfig, ResearchProgress

def example_1_basic_usage():
    """Example 1: Basic synchronous research."""
    print("=== Example 1: Basic Usage ===")
    
    # Simple usage with default configuration
    client = DeepResearchClient()
    
    try:
        result = client.research("Latest developments in quantum computing")
        
        print(f"Research completed!")
        print(f"Topic: {result.topic}")
        print(f"Hypothesis: {result.hypothesis}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Synthesis length: {len(result.synthesis_text)} characters")
        print(f"Future questions: {len(result.future_questions)}")
        
        # Save results
        result.save_to_file("quantum_computing_research.json")
        print("Results saved to quantum_computing_research.json")
        
    finally:
        client.close()

def example_2_custom_config():
    """Example 2: Research with custom configuration."""
    print("\n=== Example 2: Custom Configuration ===")
    
    # Custom configuration for more thorough research
    config = ResearchConfig(
        max_iterations=10,           # More thorough research
        token_budget=200000,         # Higher token budget
        cost_budget=15.0,           # Higher cost budget
        time_budget_seconds=1200,   # 20 minutes
        max_questions=6,            # More initial questions
        temperature=0.2,            # More focused responses
        log_level="DEBUG"           # Verbose logging
    )
    
    with DeepResearchClient(config) as client:
        result = client.research("AI safety in autonomous vehicles")
        
        print(f"Thorough research completed!")
        print(f"Steps used: {result.meta['steps_used']}")
        print(f"Tokens used: {result.meta['budgets']['tokens_used']}")
        print(f"Cost used: ${result.meta['budgets']['cost_used']:.2f}")
        print(f"Time used: {result.meta['budgets']['time_used_s']:.1f}s")

def example_3_with_progress():
    """Example 3: Research with progress tracking."""
    print("\n=== Example 3: Progress Tracking ===")
    
    def progress_handler(progress: ResearchProgress):
        """Handle progress updates during research."""
        print(f"Step {progress.step}/{progress.max_steps} | "
              f"Confidence: {progress.confidence:.2f} | "
              f"Tokens: {progress.tokens_used} | "
              f"Cost: ${progress.cost_used:.2f}")
        
        if progress.current_task:
            print(f"  Current: {progress.current_task}")
    
    config = ResearchConfig(
        max_iterations=5,
        token_budget=80000,
        cost_budget=5.0
    )
    
    with DeepResearchClient(config) as client:
        result = client.research(
            "Impact of renewable energy on grid stability",
            progress_callback=progress_handler
        )
        
        print(f"\nFinal synthesis preview:")
        print(result.synthesis_text[:300] + "..." if len(result.synthesis_text) > 300 else result.synthesis_text)

async def example_4_async_usage():
    """Example 4: Asynchronous research."""
    print("\n=== Example 4: Async Usage ===")
    
    config = ResearchConfig(
        max_iterations=4,
        token_budget=60000,
        cost_budget=3.0
    )
    
    client = DeepResearchClient(config)
    
    try:
        # Multiple concurrent research tasks
        import asyncio
        
        topics = [
            "Machine learning interpretability",
            "Blockchain scalability solutions",
            "Gene therapy recent breakthroughs"
        ]
        
        # Run multiple research tasks concurrently
        tasks = [client.research_async(topic) for topic in topics]
        results = await asyncio.gather(*tasks)
        
        for i, result in enumerate(results):
            print(f"\nResearch {i+1}: {result.topic}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Run ID: {result.run_id}")
            
            # Save each result
            filename = f"research_{i+1}_{result.run_id[:8]}.json"
            result.save_to_file(filename)
            print(f"Saved to: {filename}")
            
    finally:
        client.close()

def example_5_resume_research():
    """Example 5: Resume previous research."""
    print("\n=== Example 5: Resume Research ===")
    
    client = DeepResearchClient()
    
    try:
        # Start initial research
        print("Starting initial research...")
        result1 = client.research("Artificial general intelligence timeline")
        initial_run_id = result1.run_id
        
        print(f"Initial research completed. Run ID: {initial_run_id}")
        print(f"Initial synthesis length: {len(result1.synthesis_text)}")
        
        # Resume and continue research
        print("\nResuming research for deeper investigation...")
        result2 = client.resume_research(
            run_id=initial_run_id,
            new_topic="AGI timeline with focus on safety considerations"
        )
        
        print(f"Resumed research completed!")
        print(f"Final synthesis length: {len(result2.synthesis_text)}")
        print(f"Improvement: {len(result2.synthesis_text) - len(result1.synthesis_text)} characters added")
        
    finally:
        client.close()

def example_6_status_checking():
    """Example 6: Check research status."""
    print("\n=== Example 6: Status Checking ===")
    
    client = DeepResearchClient()
    
    try:
        # This would typically be used with a long-running research
        # For demo, we'll just show how to check status
        result = client.research("Edge computing vs cloud computing trade-offs")
        
        # Check final status
        status = client.get_research_status(result.run_id)
        if status:
            print(f"Research Status for {status['run_id'][:8]}:")
            print(f"  Topic: {status['topic']}")
            print(f"  Step: {status['step']}")
            print(f"  Confidence: {status['confidence']:.2f}")
            print(f"  Tokens used: {status['tokens_used']}")
            print(f"  Cost used: ${status['cost_used']:.2f}")
            print(f"  Time elapsed: {status['time_elapsed']:.1f}s")
            print(f"  Tasks remaining: {status['tasks_remaining']}")
        
    finally:
        client.close()

if __name__ == "__main__":
    print("DeepResearchClient Examples")
    print("=" * 50)
    
    # Check if environment is set up
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY", "FIRECRAWL_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"⚠️  Missing environment variables: {', '.join(missing)}")
        print("Please set these in your .env file before running examples.")
        sys.exit(1)
    
    try:
        # Run synchronous examples
        example_1_basic_usage()
        example_2_custom_config()
        example_3_with_progress()
        example_5_resume_research()
        example_6_status_checking()
        
        # Run async example
        import asyncio
        asyncio.run(example_4_async_usage())
        
        print("\n✅ All examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

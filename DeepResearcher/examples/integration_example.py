"""
Integration example showing how to use DeepResearchClient in a larger application.

This demonstrates how you might integrate the research client into an existing
Python application, web service, or data pipeline.
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

# Add the parent directory to the path so we can import deepresearcher
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepresearcher.api import DeepResearchClient
from deepresearcher.api.research_client import ResearchConfig, ResearchProgress, ResearchSynthesis

class ResearchService:
    """
    Example service class that integrates Deep Research capabilities.
    
    This could be part of a larger application that needs AI research capabilities.
    """
    
    def __init__(self, config: Optional[ResearchConfig] = None):
        """Initialize the research service."""
        self.client = DeepResearchClient(config)
        self.active_research: Dict[str, Dict[str, Any]] = {}
    
    def start_background_research(self, topic: str, research_id: Optional[str] = None) -> str:
        """
        Start research in the background and return immediately with research ID.
        
        Args:
            topic: Research topic
            research_id: Optional custom research ID
            
        Returns:
            Research ID for tracking progress
        """
        research_id = research_id or f"research_{int(time.time())}"
        
        def progress_tracker(progress: ResearchProgress):
            self.active_research[research_id] = {
                "status": "running",
                "progress": progress,
                "topic": topic,
                "started_at": time.time()
            }
        
        # Start research in background thread
        def run_research():
            try:
                result = self.client.research(topic, progress_callback=progress_tracker)
                self.active_research[research_id] = {
                    "status": "completed",
                    "result": result,
                    "topic": topic,
                    "completed_at": time.time()
                }
            except Exception as e:
                self.active_research[research_id] = {
                    "status": "failed",
                    "error": str(e),
                    "topic": topic,
                    "failed_at": time.time()
                }
        
        import threading
        thread = threading.Thread(target=run_research, daemon=True)
        thread.start()
        
        return research_id
    
    def get_research_progress(self, research_id: str) -> Optional[Dict[str, Any]]:
        """Get progress of ongoing research."""
        return self.active_research.get(research_id)
    
    def get_completed_research(self, research_id: str) -> Optional[ResearchSynthesis]:
        """Get completed research results."""
        research_data = self.active_research.get(research_id)
        if research_data and research_data["status"] == "completed":
            return research_data["result"]
        return None
    
    def batch_research(self, topics: List[str]) -> List[ResearchSynthesis]:
        """
        Conduct research on multiple topics sequentially.
        
        Args:
            topics: List of research topics
            
        Returns:
            List of research results
        """
        results = []
        for i, topic in enumerate(topics):
            print(f"Researching {i+1}/{len(topics)}: {topic}")
            result = self.client.research(topic)
            results.append(result)
        return results
    
    async def batch_research_async(self, topics: List[str]) -> List[ResearchSynthesis]:
        """
        Conduct research on multiple topics concurrently.
        
        Args:
            topics: List of research topics
            
        Returns:
            List of research results
        """
        tasks = [self.client.research_async(topic) for topic in topics]
        return await asyncio.gather(*tasks)
    
    def comparative_research(self, base_topic: str, variations: List[str]) -> Dict[str, ResearchSynthesis]:
        """
        Conduct comparative research on topic variations.
        
        Args:
            base_topic: Base research topic
            variations: List of topic variations to compare
            
        Returns:
            Dictionary mapping variation to research result
        """
        results = {}
        
        # Research base topic
        print(f"Researching base topic: {base_topic}")
        results["base"] = self.client.research(base_topic)
        
        # Research variations
        for variation in variations:
            full_topic = f"{base_topic} - {variation}"
            print(f"Researching variation: {full_topic}")
            results[variation] = self.client.research(full_topic)
        
        return results
    
    def close(self):
        """Close the research service."""
        self.client.close()

def example_integration_in_web_app():
    """Example of how to integrate in a web application."""
    print("\n=== Web App Integration Example ===")
    
    # This would typically be in your web app's service layer
    research_service = ResearchService()
    
    try:
        # Simulate web app workflow
        user_query = "Benefits and risks of CRISPR gene editing"
        
        # Start research in background
        research_id = research_service.start_background_research(user_query)
        print(f"Started background research: {research_id}")
        
        # Simulate checking progress (in real app, this would be via API endpoints)
        import time
        for _ in range(5):
            time.sleep(2)
            progress = research_service.get_research_progress(research_id)
            if progress:
                status = progress["status"]
                print(f"Research status: {status}")
                
                if status == "completed":
                    result = research_service.get_completed_research(research_id)
                    print(f"Research completed! Synthesis length: {len(result.synthesis_text)}")
                    break
                elif status == "failed":
                    print(f"Research failed: {progress['error']}")
                    break
                elif status == "running" and "progress" in progress:
                    prog = progress["progress"]
                    print(f"  Step {prog.step}, Confidence: {prog.confidence:.2f}")
    
    finally:
        research_service.close()

def example_data_pipeline():
    """Example of using research in a data processing pipeline."""
    print("\n=== Data Pipeline Integration Example ===")
    
    # Simulate a data pipeline that needs research capabilities
    config = ResearchConfig(
        max_iterations=3,  # Quick research for pipeline
        token_budget=50000,
        cost_budget=2.0,
        time_budget_seconds=300  # 5 minutes max
    )
    
    research_service = ResearchService(config)
    
    try:
        # Simulate processing a list of topics from a data source
        topics_to_research = [
            "5G network security vulnerabilities",
            "IoT device privacy concerns",
            "Edge AI processing advantages"
        ]
        
        # Process each topic
        research_results = []
        for topic in topics_to_research:
            print(f"Processing: {topic}")
            
            result = research_service.client.research(topic)
            
            # Extract key insights for pipeline
            pipeline_data = {
                "topic": result.topic,
                "key_insights": result.synthesis_text[:500],  # First 500 chars
                "confidence": result.confidence,
                "research_quality": "high" if result.confidence > 0.7 else "medium" if result.confidence > 0.5 else "low",
                "run_id": result.run_id,
                "processed_at": time.time()
            }
            
            research_results.append(pipeline_data)
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Quality: {pipeline_data['research_quality']}")
        
        # Save pipeline results
        with open("pipeline_research_results.json", "w") as f:
            json.dump(research_results, f, indent=2)
        
        print(f"\nPipeline completed! Processed {len(research_results)} topics")
        
    finally:
        research_service.close()

def example_research_comparison():
    """Example of comparative research analysis."""
    print("\n=== Comparative Research Example ===")
    
    config = ResearchConfig(
        max_iterations=4,
        token_budget=100000,
        cost_budget=6.0
    )
    
    research_service = ResearchService(config)
    
    try:
        # Compare different approaches to a problem
        base_topic = "Sustainable energy storage"
        variations = [
            "battery technology focus",
            "hydrogen fuel cell focus", 
            "pumped hydro focus",
            "compressed air focus"
        ]
        
        results = research_service.comparative_research(base_topic, variations)
        
        print("\nComparative Analysis Results:")
        print("-" * 40)
        
        for variation, result in results.items():
            print(f"\n{variation.upper()}:")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Synthesis length: {len(result.synthesis_text)} chars")
            print(f"  Future questions: {len(result.future_questions)}")
            
            # Save individual results
            filename = f"comparison_{variation.replace(' ', '_')}.json"
            result.save_to_file(filename)
        
        print(f"\nComparison completed! {len(results)} research variants analyzed")
        
    finally:
        research_service.close()

if __name__ == "__main__":
    print("DeepResearchClient Integration Examples")
    print("=" * 60)
    
    # Check environment
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY", "FIRECRAWL_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"⚠️  Missing environment variables: {', '.join(missing)}")
        print("Please set these in your .env file before running examples.")
        sys.exit(1)
    
    try:
        # Run examples (comment out as needed for testing)
        example_1_basic_usage()
        example_2_custom_config()
        example_3_with_progress()
        
        # Async example
        asyncio.run(example_4_async_usage())
        
        example_integration_in_web_app()
        example_data_pipeline()
        example_research_comparison()
        
        print("\n🎉 All integration examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Integration example failed: {e}")
        import traceback
        traceback.print_exc()

"""
Simple test script to verify the DeepResearchClient API interface works correctly.

This script tests basic functionality without running full research (to avoid API costs).
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:    
        from deepresearcher import DeepResearchClient, ResearchConfig, ResearchSynthesis, ResearchProgress
        print("✅ Main imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    try:
        from deepresearcher.api import DeepResearchClient as APIClient
        from deepresearcher.api.research_client import ResearchConfig as APIConfig
        print("✅ API imports successful")
    except ImportError as e:
        print(f"❌ API import failed: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration creation and validation."""
    print("\nTesting configuration...")
    
    try:
        from deepresearcher import ResearchConfig
        
        # Test default config
        config = ResearchConfig()
        print(f"✅ Default config created: {config.llm_model}")
        
        # Test custom config
        custom_config = ResearchConfig(
            llm_model="gpt-4o",
            max_iterations=5,
            token_budget=50000,
            cost_budget=3.0
        )
        print(f"✅ Custom config created: {custom_config.max_iterations} iterations")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_client_creation():
    """Test client creation without API calls."""
    print("\nTesting client creation...")
    
    try:
        from deepresearcher import DeepResearchClient, ResearchConfig
        
        # Test with environment variables (will fail if not set, but that's expected)
        config = ResearchConfig(
            openai_api_key="test_key",
            tavily_api_key="test_key", 
            firecrawl_api_key="test_key"
        )
        
        client = DeepResearchClient(config)
        print("✅ Client created successfully")
        
        # Test client methods exist
        assert hasattr(client, 'research'), "Missing research method"
        assert hasattr(client, 'research_async'), "Missing research_async method"
        assert hasattr(client, 'resume_research'), "Missing resume_research method"
        assert hasattr(client, 'get_research_status'), "Missing get_research_status method"
        assert hasattr(client, 'close'), "Missing close method"
        print("✅ All required methods present")
        
        client.close()
        print("✅ Client closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Client creation test failed: {e}")
        return False

def test_data_structures():
    """Test data structure creation."""
    print("\nTesting data structures...")
    
    try:
        from deepresearcher import ResearchSynthesis, ResearchProgress
        
        # Test ResearchSynthesis
        synthesis = ResearchSynthesis(
            topic="Test topic",
            hypothesis="Test hypothesis", 
            confidence=0.8,
            synthesis_text="Test synthesis",
            future_questions=["Question 1", "Question 2"],
            meta={"test": "data"},
            run_id="test_run_123"
        )
        
        # Test methods
        data_dict = synthesis.to_dict()
        assert "topic" in data_dict, "Missing topic in dict"
        assert data_dict["confidence"] == 0.8, "Incorrect confidence value"
        print("✅ ResearchSynthesis works correctly")
        
        # Test ResearchProgress
        progress = ResearchProgress(
            run_id="test_run",
            step=3,
            max_steps=10,
            hypothesis="Test hypothesis",
            confidence=0.7,
            synthesis_preview="Preview text",
            tokens_used=1000,
            cost_used=0.5,
            time_elapsed=120.0
        )
        
        assert progress.step == 3, "Incorrect step value"
        assert progress.confidence == 0.7, "Incorrect confidence value"
        print("✅ ResearchProgress works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structures test failed: {e}")
        return False

def test_convenience_functions():
    """Test convenience functions (without actually calling them)."""
    print("\nTesting convenience functions...")
    
    try:
        from deepresearcher.api.research_client import research, research_async
        
        # Just check they exist and are callable
        assert callable(research), "research function not callable"
        assert callable(research_async), "research_async function not callable"
        print("✅ Convenience functions available")
        
        return True
        
    except Exception as e:
        print(f"❌ Convenience functions test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("DeepResearchClient API Interface Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configuration,
        test_client_creation,
        test_data_structures,
        test_convenience_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print("Test failed, stopping...")
            break
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All API interface tests passed!")
        print("\nThe DeepResearchClient is ready to use!")
        print("See API_USAGE_GUIDE.md for detailed usage instructions.")
        print("See examples/ directory for complete usage examples.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

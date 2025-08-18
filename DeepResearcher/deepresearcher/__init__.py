"""
DeepResearcher - AI-powered research orchestrator with multi-agent workflow.

This package provides both CLI and programmatic interfaces for conducting
comprehensive AI research on any topic.
"""

from .api import DeepResearchClient
from .api.research_client import ResearchConfig, ResearchSynthesis, ResearchProgress

__version__ = "3.7.0"
__all__ = [
    "DeepResearchClient", 
    "ResearchConfig", 
    "ResearchSynthesis", 
    "ResearchProgress"
]
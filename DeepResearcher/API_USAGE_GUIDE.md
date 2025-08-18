# DeepResearchClient API Usage Guide

The `DeepResearchClient` provides a clean, class-based interface for integrating Deep Research capabilities into other Python applications. This guide covers installation, configuration, and usage patterns.

## Quick Start

```python
from deepresearcher.api import DeepResearchClient

# Basic usage
client = DeepResearchClient()
result = client.research("AI safety in autonomous vehicles")
print(f"Research complete: {result.synthesis_text[:200]}...")
client.close()
```

## Installation & Setup

1. **Environment Variables**: Set up your `.env` file with required API keys:
```bash
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here  
FIRECRAWL_API_KEY=your_firecrawl_key_here
```

2. **Dependencies**: Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration

### ResearchConfig Options

```python
from deepresearcher.api.research_client import ResearchConfig

config = ResearchConfig(
    # API Keys (optional - uses environment if not provided)
    openai_api_key="sk-...",
    tavily_api_key="tvly-...",
    firecrawl_api_key="fc-...",
    
    # Model settings
    llm_model="gpt-4o",                    # OpenAI model
    temperature=0.4,                       # Creativity vs consistency
    embed_model="text-embedding-3-small",  # Embedding model
    
    # Research behavior
    max_iterations=6,                      # Max research steps
    max_questions=4,                       # Initial questions generated
    
    # Budget controls
    token_budget=120000,                   # Total token limit
    cost_budget=8.0,                       # Total cost limit (USD)
    time_budget_seconds=600,               # Time limit (seconds)
    
    # Performance
    max_concurrency=5,                     # Concurrent operations
    request_timeout=25,                    # Request timeout (seconds)
    tokens_per_source=2500,                # Tokens per document
    
    # Caching
    cache_dir=".cache/research",           # Cache directory
    clear_cache=False,                     # Clear cache on start
    
    # Logging
    log_level="INFO"                       # DEBUG, INFO, WARNING, ERROR
)
```

## Core Methods

### Synchronous Research

```python
# Basic research
client = DeepResearchClient()
result = client.research("quantum computing applications")

# With custom configuration
config = ResearchConfig(max_iterations=10, cost_budget=15.0)
client = DeepResearchClient(config)
result = client.research("climate change mitigation")

# With progress tracking
def on_progress(progress):
    print(f"Step {progress.step}: {progress.current_task}")

result = client.research("AI ethics", progress_callback=on_progress)
```

### Asynchronous Research

```python
import asyncio

async def async_research():
    client = DeepResearchClient()
    result = await client.research_async("machine learning interpretability")
    return result

# Multiple concurrent research
async def concurrent_research():
    client = DeepResearchClient()
    topics = ["AI safety", "Quantum computing", "Gene therapy"]
    
    tasks = [client.research_async(topic) for topic in topics]
    results = await asyncio.gather(*tasks)
    return results
```

### Resume Research

```python
# Start research
result1 = client.research("artificial general intelligence")
run_id = result1.run_id

# Resume later with additional focus
result2 = client.resume_research(
    run_id=run_id,
    new_topic="AGI with focus on safety considerations"
)
```

### Status Checking

```python
# Check research status
status = client.get_research_status(run_id)
if status:
    print(f"Step: {status['step']}")
    print(f"Confidence: {status['confidence']}")
    print(f"Tokens used: {status['tokens_used']}")
    print(f"Cost used: ${status['cost_used']:.2f}")
```

## Data Structures

### ResearchSynthesis

The main result object containing research findings:

```python
@dataclass
class ResearchSynthesis:
    topic: str                    # Research topic
    hypothesis: str              # Final hypothesis
    confidence: float            # Confidence score (0-1)
    synthesis_text: str          # Main research synthesis
    future_questions: List[str]  # Suggested follow-up questions
    meta: Dict[str, Any]        # Metadata (budgets, timing, etc.)
    run_id: str                 # Unique research session ID
    
    # Methods
    def to_dict() -> Dict[str, Any]     # Convert to dictionary
    def save_to_file(filepath: str)     # Save to JSON file
```

### ResearchProgress

Progress information for ongoing research:

```python
@dataclass
class ResearchProgress:
    run_id: str
    step: int
    max_steps: int
    hypothesis: str
    confidence: float
    synthesis_preview: str
    tokens_used: int
    cost_used: float
    time_elapsed: float
    current_task: Optional[str]
    completed_tasks: int
    remaining_tasks: int
```

## Usage Patterns

### 1. Context Manager Pattern

```python
with DeepResearchClient(config) as client:
    result = client.research("renewable energy trends")
    # Client automatically closed
```

### 2. Service Integration Pattern

```python
class MyApplication:
    def __init__(self):
        self.research_client = DeepResearchClient()
    
    def analyze_market_trend(self, topic: str):
        research = self.research_client.research(f"Market trends in {topic}")
        return self.process_research_for_business(research)
    
    def cleanup(self):
        self.research_client.close()
```

### 3. Batch Processing Pattern

```python
def research_multiple_topics(topics: List[str]) -> List[ResearchSynthesis]:
    client = DeepResearchClient()
    try:
        results = []
        for topic in topics:
            result = client.research(topic)
            results.append(result)
        return results
    finally:
        client.close()
```

### 4. Background Task Pattern

```python
import threading
from queue import Queue

def background_research_worker(topic_queue: Queue, result_queue: Queue):
    client = DeepResearchClient()
    try:
        while True:
            topic = topic_queue.get()
            if topic is None:  # Shutdown signal
                break
            
            result = client.research(topic)
            result_queue.put(result)
            topic_queue.task_done()
    finally:
        client.close()

# Start background worker
topic_queue = Queue()
result_queue = Queue()
worker = threading.Thread(target=background_research_worker, args=(topic_queue, result_queue))
worker.start()

# Submit research tasks
topic_queue.put("AI in healthcare")
topic_queue.put("Blockchain in supply chain")
```

## Error Handling

```python
from deepresearcher.api import DeepResearchClient
from deepresearcher.api.research_client import ResearchConfig

try:
    client = DeepResearchClient()
    result = client.research("complex research topic")
    
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Research failed: {e}")
finally:
    if 'client' in locals():
        client.close()
```

## Best Practices

### 1. Resource Management
- Always call `client.close()` or use context managers
- Set appropriate budgets to prevent runaway costs
- Use caching for repeated research topics

### 2. Configuration
- Set realistic budgets based on research complexity
- Use higher iterations for thorough research
- Adjust temperature for creativity vs consistency

### 3. Integration
- Use async methods for better performance in async applications
- Implement progress callbacks for long-running research
- Cache results to avoid duplicate research

### 4. Error Handling
- Always wrap research calls in try-catch blocks
- Check budget constraints before starting research
- Validate API keys and environment setup

## Performance Considerations

- **Concurrency**: Adjust `max_concurrency` based on your system and API limits
- **Caching**: Enable caching for repeated research to save costs
- **Budgets**: Set appropriate token/cost/time budgets to prevent overruns
- **Timeouts**: Adjust request timeouts based on network conditions

## Integration Examples

See the `examples/` directory for complete integration examples:
- `basic_usage.py` - Simple usage patterns
- `integration_example.py` - Advanced integration scenarios

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all required environment variables are set
2. **Budget Exceeded**: Increase budgets or reduce research scope
3. **Network Timeouts**: Increase request timeout or check network connectivity
4. **Import Errors**: Ensure all dependencies are installed

### Debug Mode

Enable debug logging for troubleshooting:

```python
config = ResearchConfig(log_level="DEBUG")
client = DeepResearchClient(config)
```

## API Reference

### DeepResearchClient Methods

- `research(topic, run_id=None, progress_callback=None)` - Synchronous research
- `research_async(topic, run_id=None, progress_callback=None)` - Asynchronous research  
- `resume_research(run_id, new_topic=None, progress_callback=None)` - Resume existing research
- `resume_research_async(run_id, new_topic=None, progress_callback=None)` - Resume async
- `get_research_status(run_id)` - Get research status
- `get_research_status_async(run_id)` - Get research status async
- `close()` - Clean up resources

### Convenience Functions

- `research(topic, config=None)` - Quick synchronous research
- `research_async(topic, config=None)` - Quick asynchronous research

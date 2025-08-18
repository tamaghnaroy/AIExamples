# DeepResearcher

An advanced AI-powered research orchestrator that conducts comprehensive, multi-agent research on any topic using web search, content analysis, and intelligent synthesis.

## 🚀 Features

- **Multi-Agent Architecture**: Specialized agents for topic exploration, search & crawl, summarization, analysis, synthesis, and refinement
- **Intelligent Orchestration**: RouterAgent coordinates research workflow with dynamic tool selection
- **Advanced Search & Retrieval**: Integration with Tavily search and Firecrawl scraping
- **Hypothesis-Driven Research**: Maintains and refines research hypotheses with confidence tracking
- **Budget Management**: Token, cost, and time budget controls with real-time monitoring
- **Adversarial Review**: Blue/Red team debate system for evidence validation
- **Caching & Resume**: Persistent state management with Redis support and resume capability
- **Structured Output**: JSON-formatted research results with detailed evidence chains

## 🏗️ Architecture

### Core Components

- **Orchestrator**: Main workflow coordinator managing agent interactions
- **RouterAgent**: LLM-powered agent that selects appropriate tools based on research context
- **StateStore**: Persistent state management with Redis backend
- **Budget Tracking**: Real-time monitoring of token usage, costs, and time limits

### Specialized Agents

- **TopicExplorationAgent**: Generates research questions and exploration strategies
- **SearchAndCrawlAgent**: Web search and content scraping with semantic ranking
- **SummarizerAgent**: Content summarization with token limit management
- **AnalysisAgent**: Deep content analysis for claims, statistics, and entities
- **SynthesisAgent**: Evidence synthesis and hypothesis refinement
- **RefinementAgent**: Confidence scoring and research quality assessment

### Advanced Tools

- **Verification Tools**: Claim verification with targeted search
- **Critique Tools**: Evidence bias and contradiction detection
- **Adversarial Review**: Multi-perspective debate and synthesis
- **Task Management**: Dynamic sub-task creation and completion tracking

## 📦 Installation

### Prerequisites

- Python 3.8+
- API Keys for:
  - OpenAI (GPT-4/GPT-3.5)
  - Tavily (Web Search)
  - Firecrawl (Web Scraping)
- Optional: Redis server for state persistence

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DeepResearcher
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   TAVILY_API_KEY=your_tavily_api_key
   FIRECRAWL_API_KEY=your_firecrawl_api_key
   
   # Optional Redis configuration
   REDIS_URL=redis://localhost:6379
   
   # Optional model configuration
   LLM_MODEL=gpt-4o
   EMBED_MODEL=text-embedding-3-large
   ```
   
   **⚠️ Security Note**: Never commit your `.env` file to version control. Add it to `.gitignore`.

## 🚀 Usage

### Command Line Interface

**Basic Usage:**
```bash
python deepresearcher.py -t "Your research topic"
```

**Advanced Options:**
```bash
python deepresearcher.py \
  --topic "Climate change impact on agriculture" \
  --model gpt-4o \
  --temp 0.4 \
  --iter-depth 8 \
  --max-questions 6 \
  --token-budget 500000 \
  --cost-budget 5.0 \
  --time-budget 7200 \
  --output research_results.json
```

**Resume Previous Research:**
```bash
python deepresearcher.py --run-id <previous-run-id>
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--topic` | Research topic or question | Required |
| `--model` | OpenAI model to use | `gpt-4o` |
| `--temp` | LLM temperature (0.0-1.0) | `0.4` |
| `--iter-depth` | Maximum research iterations | `5` |
| `--max-questions` | Max exploration questions | `5` |
| `--token-budget` | Total token budget | `1,000,000` |
| `--cost-budget` | Total cost budget (USD) | `2.0` |
| `--time-budget` | Time budget (seconds) | `3600` |
| `--max-concurrency` | Concurrent requests | `5` |
| `--cache-dir` | Cache directory | `./.cache` |
| `--clear-cache` | Clear cache before running | `false` |

## 📊 Output Format

DeepResearcher generates structured JSON output containing:

```json
{
  "run_id": "unique-run-identifier",
  "topic": "Research topic",
  "final_synthesis": {
    "text": "Comprehensive research synthesis",
    "confidence": 0.85,
    "evidence_count": 25
  },
  "hypothesis": {
    "statement": "Final research hypothesis",
    "confidence": 0.82
  },
  "evidence": [
    {
      "source": "URL or document",
      "content": "Relevant content excerpt",
      "claims": ["Extracted claims"],
      "confidence": 0.9
    }
  ],
  "verified_claims": [
    {
      "claim": "Specific claim",
      "verification_score": 0.88,
      "supporting_sources": ["source1", "source2"]
    }
  ],
  "budget_usage": {
    "tokens_used": 45000,
    "cost_incurred": 1.25,
    "time_elapsed": 1800
  }
}
```

## 🔧 Advanced Features

### Adversarial Review System

The system includes a sophisticated adversarial review process:
- **Blue Team**: Argues for current hypothesis
- **Red Team**: Challenges and critiques evidence
- **Moderator**: Synthesizes balanced final conclusion

### Budget Management

Comprehensive budget tracking prevents runaway costs:
- **Token Budget**: Tracks input/output tokens across all API calls
- **Cost Budget**: Real-time cost calculation based on model pricing
- **Time Budget**: Automatic termination after specified duration

### Caching & Performance

- **Embedding Cache**: Persistent vector embeddings for content
- **Page Cache**: Scraped content caching to avoid re-fetching
- **State Persistence**: Full research state saved for resumption

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=deepresearcher --cov-report=html

# Run specific test file
pytest tests/test_orchestrator.py -v
```

## 📁 Project Structure

```
DeepResearcher/
├── deepresearcher/
│   ├── agents/           # Specialized research agents
│   ├── cli/             # Command-line interface
│   ├── core/            # Core models and configuration
│   ├── orchestration/   # Workflow orchestration
│   ├── providers/       # External service providers
│   ├── storage/         # State persistence
│   ├── tools/           # Research tools and utilities
│   └── utils/           # Helper utilities
├── tests/               # Test suite
├── .cache/             # Local cache directory
├── requirements.txt    # Python dependencies
└── deepresearcher.py   # Main entry point
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Dependencies

### Core Dependencies
- **OpenAI**: LLM and embeddings
- **Tavily**: Web search API
- **Firecrawl**: Web scraping and content extraction
- **Pydantic**: Data validation and serialization
- **Structlog**: Structured logging

### Optional Dependencies
- **Redis**: State persistence (recommended for production)
- **Pandas/NumPy**: Data processing and analysis
- **Pytest**: Testing framework

## 🚨 Important Notes

- **API Costs**: Monitor your usage as research can consume significant tokens
- **Rate Limits**: Respect API rate limits; adjust concurrency settings if needed
- **Cache Management**: Regular cache cleanup recommended for long-term usage
- **Security**: Never commit API keys; use environment variables or `.env` files

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation and examples
- Review test cases for usage patterns

---

**DeepResearcher** - Transforming how we conduct comprehensive AI-powered research.

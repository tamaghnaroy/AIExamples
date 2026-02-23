# Prompt Universe Generator

A module for generating and iteratively improving a "prompt universe" + "tool universe" for an FX/macro hedge fund LLM agent.

## Overview

This module creates:
1. **Prompt Universe**: Realistic user prompts from FX market participants (traders, PMs, risk, quant, operations)
2. **Tool Universe**: Structured list of tools the agent will call to answer those prompts
3. **Iterative Evaluator Loop**: LLM-driven scoring, gap detection, and enrichment

## Installation

Ensure you have the required dependencies:

```bash
pip install openai python-dotenv
```

## Configuration

Create a `.env` file with:

```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4
```

The module searches for `.env` in:
1. Module directory (`FinAnalytics/AI/prompt_universe/.env`)
2. `microtasks/pca_time_series_analysis/.env`
3. Current working directory

## Usage

### Run from command line:

```bash
python -m FinAnalytics.AI.prompt_universe.runner --run
```

### Options:

- `--run`: Start the generation loop
- `--max-iterations N`: Maximum iterations (default: 100)
- `--seed N`: Random seed for reproducibility (default: 42)

### Run programmatically:

```python
from FinAnalytics.AI.prompt_universe.runner import Runner

runner = Runner(seed=42)
prompts, tools = runner.run(max_iterations=100)
```

## Output Files

All files are stored in `FinAnalytics/AI/prompt_universe/data/`:

### 1. `prompt_file.jsonl`
JSON Lines file with prompts:
```json
{
  "prompt_id": "P000001",
  "prompt_text": "...",
  "category": "market_query",
  "difficulty": 3,
  "persona": "spot_trader",
  "tags": ["fx", "spot"],
  "created_in_iteration": 0
}
```

### 2. `prompt_category.csv`
Tabular view with columns:
- `prompt_id`, `category`, `difficulty`, `persona`
- `difficulty_rationale`, `category_rationale`

### 3. `prompt_matrix.csv`
Matrix with:
- Rows = difficulty levels (1-5)
- Columns = categories
- Values = prompt counts

### 4. `tool_calls.json`
Array of tool definitions:
```json
{
  "tool_name": "get_fx_spot_history",
  "tool_description": "...",
  "tool_inputs": {...},
  "tool_outputs": {...},
  "examples": [...]
}
```

## Categories

Initial categories include:
- `market_query`, `historical_query`, `pricing`, `historical_pricing`
- `risk_greeks`, `scenario_analysis`, `vol_surface_analysis`
- `trade_discovery`, `backtesting`, `hedging`
- `portfolio_analytics`, `pnl_attribution`, `carry_roll_analysis`
- `execution_microstructure`, `liquidity_transaction_cost`
- `event_risk_calendar`, `relative_value`, `basis_cross_currency`
- `rates_fx_linked_analysis`, `data_quality_reconciliation`, `ops_workflow_query`

## Difficulty Scale

| Level | Description |
|-------|-------------|
| 1 | Simple factual or single-series query |
| 2 | Requires basic transformation, simple comparisons |
| 3 | Multi-step analysis across multiple series/instruments |
| 4 | Requires structured reasoning + scenario/simulation + constraints |
| 5 | Open-ended research / complex portfolio + risk + constraints |

## Personas

- Portfolio Manager, Spot Trader, Options Trader
- Risk Manager, Quant, Research Analyst, Macro Analyst
- RV Trader, Vol RV Trader, Event Trader, Scalper

## Stopping Criteria

### Prompt Generation Stops When:
- Every cell (difficulty × category) has >= 100 prompts
- LLM overall_score >= 9

### Tool Validation Stops When:
- After 10 validation iterations
- Aggregated `cannot_answer` list is empty

## Module Structure

```
prompt_universe/
├── __init__.py      # Package exports
├── config.py        # Configuration and constants
├── generator.py     # Prompt generation
├── matrix.py        # Matrix computation
├── tools.py         # Tool file management
├── evaluator.py     # LLM scoring calls
├── runner.py        # Iteration orchestrator CLI
├── README.md        # This file
├── data/            # Generated data files
└── iterations/      # Iteration artifacts and logs
```

## Iteration Artifacts

Each iteration saves to `iterations/iter_NNN/`:
- `prompt_evaluation.json`: LLM prompt scoring
- `tool_evaluation.json`: LLM tool scoring
- `matrix_snapshot.json`: Matrix state
- `stats.json`: Iteration statistics

Logs are saved to `iterations/logs/` for auditability.

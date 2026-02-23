"""
Configuration and constants for the prompt universe module.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Search in multiple locations
ENV_PATHS = [
    Path(__file__).parent / '.env',
    Path(__file__).parent.parent.parent.parent / 'microtasks' / 'pca_time_series_analysis' / '.env',
    Path.cwd() / '.env',
]

for env_path in ENV_PATHS:
    if env_path.exists():
        load_dotenv(env_path)
        break

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '').strip().strip('"')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4').strip().strip('"')

# Random seed for determinism
RANDOM_SEED = 42

# Module paths
MODULE_DIR = Path(__file__).parent
DATA_DIR = MODULE_DIR / 'data'
ITERATIONS_DIR = MODULE_DIR / 'iterations'

# File paths
PROMPT_FILE = DATA_DIR / 'prompt_file.jsonl'
PROMPT_CATEGORY_FILE = DATA_DIR / 'prompt_category.csv'
PROMPT_MATRIX_FILE = DATA_DIR / 'prompt_matrix.csv'
TOOL_CALLS_FILE = DATA_DIR / 'tool_calls.json'

# Categories (initial set)
CATEGORIES = [
    'market_query',
    'historical_query',
    'pricing',
    'historical_pricing',
    'risk_greeks',
    'scenario_analysis',
    'vol_surface_analysis',
    'trade_discovery',
    'backtesting',
    'hedging',
    'portfolio_analytics',
    'pnl_attribution',
    'carry_roll_analysis',
    'execution_microstructure',
    'liquidity_transaction_cost',
    'event_risk_calendar',
    'relative_value',
    'basis_cross_currency',
    'rates_fx_linked_analysis',
    'data_quality_reconciliation',
    'ops_workflow_query',
]

# Difficulty scale (1-5)
DIFFICULTY_SCALE = {
    1: "simple factual or single-series query",
    2: "requires basic transformation, simple comparisons",
    3: "multi-step analysis across multiple series / instruments",
    4: "requires structured reasoning + scenario/simulation + constraints",
    5: "open-ended research / complex portfolio + risk + constraints",
}

# Personas
PERSONAS = [
    'Portfolio Manager',
    'Spot Trader',
    'Options Trader',
    'Risk Manager',
    'Quant',
    'Research Analyst',
    'Macro Analyst',
    'RV Trader',
    'Vol RV Trader',
    'Event Trader',
    'Scalper',
]

# Stopping criteria
MIN_PROMPTS_PER_CELL = 100
MIN_OVERALL_SCORE = 9
TOOL_VALIDATION_ITERATIONS = 10
TOOL_VALIDATION_SAMPLE_SIZE = 100

# Initial generation targets
INITIAL_PROMPT_COUNT = 200
INITIAL_TOOL_COUNT = 30

# LLM retry settings
MAX_LLM_RETRIES = 3

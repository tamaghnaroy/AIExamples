"""
Evaluator module for LLM-driven scoring and gap detection.
"""

import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from openai import OpenAI

try:
    from .config import (
        OPENAI_API_KEY, OPENAI_MODEL, RANDOM_SEED,
        CATEGORIES, DIFFICULTY_SCALE, PERSONAS,
        ITERATIONS_DIR, MAX_LLM_RETRIES,
        TOOL_VALIDATION_SAMPLE_SIZE
    )
    from .logging_config import get_logger
except ImportError:
    from config import (
        OPENAI_API_KEY, OPENAI_MODEL, RANDOM_SEED,
        CATEGORIES, DIFFICULTY_SCALE, PERSONAS,
        ITERATIONS_DIR, MAX_LLM_RETRIES,
        TOOL_VALIDATION_SAMPLE_SIZE
    )
    from logging_config import get_logger

logger = get_logger('evaluator')


class Evaluator:
    """LLM-driven evaluation of prompt and tool universes."""
    
    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        random.seed(seed)
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.log_dir = ITERATIONS_DIR / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.total_api_calls = 0
        self.total_api_time = 0.0
        logger.info(f"Evaluator initialized with seed={seed}, model={self.model}")
        logger.info(f"LLM request/response logs will be saved to {self.log_dir}")
        
    def _log_request_response(self, request: Dict, response: Dict, prefix: str = "eval"):
        """Log LLM request/response to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = self.log_dir / f"{prefix}_{timestamp}.json"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "request": request,
                "response": response
            }, f, indent=2)
        logger.debug(f"Logged request/response to {log_file.name}")
    
    def _call_llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
        retries: int = MAX_LLM_RETRIES,
        log_prefix: str = "eval"
    ) -> Dict[str, Any]:
        """Call LLM and parse JSON response with retry logic."""
        request = {"system": system_prompt, "user": user_prompt}
        
        for attempt in range(retries):
            try:
                logger.debug(f"[{log_prefix}] LLM API call attempt {attempt+1}/{retries}")
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    seed=self.seed + attempt
                )
                
                elapsed = time.time() - start_time
                self.total_api_calls += 1
                self.total_api_time += elapsed
                
                content = response.choices[0].message.content
                result = json.loads(content)
                
                tokens_used = getattr(response.usage, 'total_tokens', 'N/A')
                logger.debug(f"[{log_prefix}] LLM call successful in {elapsed:.2f}s, tokens={tokens_used}")
                
                self._log_request_response(request, result, log_prefix)
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"[{log_prefix}] JSON parse error on attempt {attempt+1}: {e}")
                if attempt < retries - 1:
                    user_prompt = f"{user_prompt}\n\nPREVIOUS RESPONSE WAS INVALID JSON. Return valid JSON only."
                else:
                    logger.error(f"[{log_prefix}] Failed to parse JSON after {retries} attempts")
                    self._log_request_response(request, {"error": str(e)}, log_prefix)
                    raise ValueError(f"Failed to parse JSON after {retries} attempts: {e}")
            except Exception as e:
                logger.warning(f"[{log_prefix}] API error on attempt {attempt+1}: {type(e).__name__}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                logger.error(f"[{log_prefix}] API call failed after {retries} attempts")
                self._log_request_response(request, {"error": str(e)}, log_prefix)
                raise
        return {}
    
    def score_prompt_universe(
        self,
        categories: List[str],
        matrix: Dict[int, Dict[str, int]],
        sampled_prompts: Dict[str, Dict[int, List[Dict[str, Any]]]]
    ) -> Dict[str, Any]:
        """Score the prompt universe and identify gaps."""
        
        logger.info("=== SCORING PROMPT UNIVERSE ===")
        total_prompts = sum(sum(matrix[d].values()) for d in matrix)
        logger.info(f"Total prompts: {total_prompts}, Categories: {len(categories)}")
        
        matrix_text = "Prompt Matrix (rows=difficulty, cols=categories):\n"
        for diff in sorted(matrix.keys()):
            row_data = {cat: matrix[diff].get(cat, 0) for cat in categories}
            row_total = sum(row_data.values())
            matrix_text += f"Difficulty {diff}: {json.dumps(row_data)}\n"
            logger.debug(f"  Difficulty {diff}: {row_total} prompts")
        
        samples_text = "Sample prompts per cell (up to 5 each):\n"
        sample_count = 0
        for cat, diffs in list(sampled_prompts.items())[:10]:
            for diff, prompts in diffs.items():
                for p in prompts[:2]:
                    samples_text += f"[{cat}/D{diff}] {p.get('prompt_text', '')[:150]}...\n"
                    sample_count += 1
        logger.debug(f"Including {sample_count} sample prompts in evaluation")
        
        system_prompt = """You are an expert evaluator of FX/macro hedge fund prompt datasets.

Evaluate the prompt universe for:
1. Coverage across categories and difficulty levels
2. Realism and quality of prompts
3. Missing categories or personas
4. Overall completeness

Return strict JSON only."""

        user_prompt = f"""Evaluate this prompt universe:

CATEGORIES: {json.dumps(categories)}

PERSONAS: {json.dumps(PERSONAS)}

DIFFICULTY SCALE: {json.dumps(DIFFICULTY_SCALE)}

{matrix_text}

{samples_text}

Return JSON:
{{
  "overall_score": <0-10>,
  "missing_categories": ["category1", "category2"],
  "missing_personas": ["persona1"],
  "notes": "short critique",
  "recommended_new_categories": ["new_cat1", "new_cat2"]
}}"""

        result = self._call_llm_json(system_prompt, user_prompt, log_prefix="prompt_eval")
        
        score = result.get('overall_score', 0)
        missing_cats = result.get('missing_categories', [])
        new_cats = result.get('recommended_new_categories', [])
        logger.info(f"Prompt universe score: {score}/10")
        logger.info(f"Missing categories: {missing_cats}")
        logger.info(f"Recommended new categories: {new_cats}")
        logger.debug(f"Evaluator notes: {result.get('notes', 'N/A')}")
        
        return result
    
    def score_tool_set(
        self,
        tools: List[Dict[str, Any]],
        categories: List[str],
        matrix_summary: str
    ) -> Dict[str, Any]:
        """Score the tool set completeness."""
        
        logger.info("=== SCORING TOOL SET ===")
        logger.info(f"Evaluating {len(tools)} tools against {len(categories)} categories")
        
        tools_text = json.dumps(tools, indent=2)
        
        system_prompt = """You are an expert evaluator of FX/macro hedge fund tool APIs.

Evaluate the tool set for:
1. Coverage of all prompt categories
2. Completeness for answering realistic queries
3. Redundancy or gaps
4. Quality of tool definitions

Return strict JSON only."""

        user_prompt = f"""Evaluate this tool set:

CATEGORIES TO COVER: {json.dumps(categories)}

PROMPT COVERAGE SUMMARY:
{matrix_summary}

CURRENT TOOLS:
{tools_text}

Return JSON:
{{
  "tool_set_score": <0-10>,
  "missing_tooling_for_categories": {{
    "category_name": ["tool idea 1", "tool idea 2"]
  }},
  "redundant_tools": ["tool_name1"],
  "notes": "short critique"
}}"""

        result = self._call_llm_json(system_prompt, user_prompt, log_prefix="tool_eval")
        
        score = result.get('tool_set_score', 0)
        missing = result.get('missing_tooling_for_categories', {})
        redundant = result.get('redundant_tools', [])
        logger.info(f"Tool set score: {score}/10")
        logger.info(f"Categories with missing tools: {len(missing)}")
        if missing:
            for cat, ideas in list(missing.items())[:5]:
                logger.debug(f"  {cat}: {ideas}")
        logger.info(f"Redundant tools: {redundant}")
        
        return result
    
    def validate_tools_against_prompts(
        self,
        prompts: List[Dict[str, Any]],
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check if tools can answer the given prompts."""
        
        logger.info(f"=== VALIDATING TOOLS AGAINST {len(prompts)} PROMPTS ===")
        logger.debug(f"Using {len(tools)} tools for validation")
        
        prompts_text = json.dumps([
            {"prompt_id": p["prompt_id"], "prompt_text": p["prompt_text"], 
             "category": p["category"], "difficulty": p["difficulty"]}
            for p in prompts
        ], indent=2)
        
        tools_text = json.dumps(tools, indent=2)
        
        system_prompt = """You are an expert at evaluating whether a set of tools can answer FX/macro hedge fund queries.

For each prompt, determine if the available tools are sufficient to answer it.
If not, suggest what tool is missing.

Return strict JSON only."""

        user_prompt = f"""Evaluate if these tools can answer these prompts:

PROMPTS:
{prompts_text}

AVAILABLE TOOLS:
{tools_text}

Return JSON:
{{
  "cannot_answer": [
    {{
      "prompt_id": "P000123",
      "reason": "missing tool for ...",
      "suggested_tool": {{
        "tool_name": "suggested_tool_name",
        "tool_description": "what it should do",
        "tool_inputs": {{"param": "type"}},
        "tool_outputs": {{"schema": "table", "fields": ["field1"]}}
      }}
    }}
  ]
}}

If all prompts can be answered, return {{"cannot_answer": []}}"""

        result = self._call_llm_json(system_prompt, user_prompt, log_prefix="tool_validation")
        
        cannot_answer = result.get('cannot_answer', [])
        logger.info(f"Validation result: {len(cannot_answer)}/{len(prompts)} prompts cannot be answered")
        if cannot_answer:
            for item in cannot_answer[:5]:
                logger.debug(f"  {item.get('prompt_id')}: {item.get('reason', 'N/A')[:80]}")
            if len(cannot_answer) > 5:
                logger.debug(f"  ... and {len(cannot_answer) - 5} more")
        
        return result
    
    def sample_prompts_uniformly(
        self,
        prompts: List[Dict[str, Any]],
        categories: List[str],
        sample_size: int = TOOL_VALIDATION_SAMPLE_SIZE
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Sample prompts uniformly across categories and difficulties."""
        
        logger.info(f"Sampling {sample_size} prompts uniformly from {len(prompts)} total")
        
        grouped = {}
        for p in prompts:
            key = (p.get('category', ''), p.get('difficulty', 1))
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(p)
        
        cells = [(cat, diff) for cat in categories for diff in range(1, 6)]
        quota_per_cell = max(1, sample_size // len(cells))
        remainder = sample_size - (quota_per_cell * len(cells))
        
        logger.debug(f"Sampling from {len(cells)} cells, ~{quota_per_cell} per cell")
        
        sampled = []
        redistribution_log = {}
        
        random.shuffle(cells)
        
        for i, (cat, diff) in enumerate(cells):
            target = quota_per_cell + (1 if i < remainder else 0)
            available = grouped.get((cat, diff), [])
            
            if len(available) >= target:
                sampled.extend(random.sample(available, target))
            else:
                sampled.extend(available)
                shortfall = target - len(available)
                redistribution_log[(cat, diff)] = {
                    "requested": target,
                    "available": len(available),
                    "shortfall": shortfall
                }
        
        if len(sampled) < sample_size:
            all_remaining = [p for p in prompts if p not in sampled]
            needed = sample_size - len(sampled)
            if all_remaining:
                sampled.extend(random.sample(all_remaining, min(needed, len(all_remaining))))
        
        logger.info(f"Sampled {len(sampled)} prompts, {len(redistribution_log)} cells had shortfalls")
        if redistribution_log:
            total_shortfall = sum(v['shortfall'] for v in redistribution_log.values())
            logger.debug(f"Total shortfall: {total_shortfall} prompts")
        
        return sampled, redistribution_log
    
    def save_iteration_results(
        self,
        iteration: int,
        prompt_eval: Dict[str, Any],
        tool_eval: Dict[str, Any],
        matrix: Dict[int, Dict[str, int]],
        stats: Dict[str, Any]
    ):
        """Save iteration results to disk."""
        iter_dir = ITERATIONS_DIR / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving iteration {iteration} results to {iter_dir}")
        
        with open(iter_dir / "prompt_evaluation.json", 'w', encoding='utf-8') as f:
            json.dump(prompt_eval, f, indent=2)
        
        with open(iter_dir / "tool_evaluation.json", 'w', encoding='utf-8') as f:
            json.dump(tool_eval, f, indent=2)
        
        with open(iter_dir / "matrix_snapshot.json", 'w', encoding='utf-8') as f:
            json.dump(matrix, f, indent=2, default=str)
        
        with open(iter_dir / "stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Iteration {iteration} results saved: prompt_eval, tool_eval, matrix, stats")

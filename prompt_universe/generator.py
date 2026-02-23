"""
Prompt generation module for creating FX/macro hedge fund prompts.
"""

import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI

try:
    from .config import (
        OPENAI_API_KEY, OPENAI_MODEL, RANDOM_SEED,
        CATEGORIES, DIFFICULTY_SCALE, PERSONAS,
        PROMPT_FILE, DATA_DIR, MAX_LLM_RETRIES
    )
    from .logging_config import get_logger
except ImportError:
    from config import (
        OPENAI_API_KEY, OPENAI_MODEL, RANDOM_SEED,
        CATEGORIES, DIFFICULTY_SCALE, PERSONAS,
        PROMPT_FILE, DATA_DIR, MAX_LLM_RETRIES
    )
    from logging_config import get_logger

logger = get_logger('generator')


class PromptGenerator:
    """Generates realistic FX/macro hedge fund prompts using LLM."""
    
    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        random.seed(seed)
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.prompt_counter = 0
        self.total_api_calls = 0
        self.total_api_time = 0.0
        logger.info(f"PromptGenerator initialized with seed={seed}, model={self.model}")
        
    def _get_next_prompt_id(self) -> str:
        """Generate next unique prompt ID."""
        self.prompt_counter += 1
        return f"P{self.prompt_counter:06d}"
    
    def _call_llm_json(self, system_prompt: str, user_prompt: str, retries: int = MAX_LLM_RETRIES, context: str = "") -> Dict[str, Any]:
        """Call LLM and parse JSON response with retry logic."""
        for attempt in range(retries):
            try:
                logger.debug(f"[{context}] LLM API call attempt {attempt+1}/{retries}")
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 if attempt == 0 else 0.5,
                    seed=self.seed + attempt
                )
                
                elapsed = time.time() - start_time
                self.total_api_calls += 1
                self.total_api_time += elapsed
                
                content = response.choices[0].message.content
                result = json.loads(content)
                
                tokens_used = getattr(response.usage, 'total_tokens', 'N/A')
                logger.debug(f"[{context}] LLM call successful in {elapsed:.2f}s, tokens={tokens_used}")
                
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"[{context}] JSON parse error on attempt {attempt+1}: {e}")
                if attempt < retries - 1:
                    user_prompt = f"{user_prompt}\n\nPREVIOUS RESPONSE WAS INVALID JSON. Return valid JSON only."
                else:
                    logger.error(f"[{context}] Failed to parse JSON after {retries} attempts")
                    raise ValueError(f"Failed to parse JSON after {retries} attempts: {e}")
            except Exception as e:
                logger.warning(f"[{context}] API error on attempt {attempt+1}: {type(e).__name__}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                logger.error(f"[{context}] API call failed after {retries} attempts")
                raise
        return {}
    
    def generate_prompts_batch(
        self,
        category: str,
        difficulty: int,
        count: int,
        iteration: int,
        existing_prompts: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate a batch of prompts for a specific category and difficulty."""
        
        context = f"batch_{category}_D{difficulty}"
        logger.info(f"[{context}] Generating {count} prompts for category='{category}', difficulty={difficulty}, iteration={iteration}")
        
        existing_prompts = existing_prompts or []
        existing_sample = random.sample(existing_prompts, min(5, len(existing_prompts))) if existing_prompts else []
        logger.debug(f"[{context}] Using {len(existing_sample)} existing prompts as negative examples")
        
        system_prompt = f"""You are an expert at creating realistic prompts that FX/macro hedge fund professionals would ask an AI assistant.

DIFFICULTY SCALE:
{json.dumps(DIFFICULTY_SCALE, indent=2)}

PERSONAS: {', '.join(PERSONAS)}

You must generate prompts that:
- Are realistic, jargon-friendly, hedge-fund style
- Match the specified category and difficulty level
- Cover different personas naturally
- Are diverse and non-repetitive

Return a JSON object with a "prompts" array containing exactly {count} prompt objects."""

        user_prompt = f"""Generate {count} unique prompts for:
- Category: {category}
- Difficulty: {difficulty} ({DIFFICULTY_SCALE[difficulty]})

{"Avoid similarity to these existing prompts:" + json.dumps(existing_sample) if existing_sample else ""}

Return JSON format:
{{
  "prompts": [
    {{
      "prompt_text": "the actual prompt text",
      "persona": "one of the personas",
      "tags": ["relevant", "tags"],
      "difficulty_rationale": "brief explanation of why this difficulty",
      "category_rationale": "brief explanation of why this category"
    }}
  ]
}}"""

        result = self._call_llm_json(system_prompt, user_prompt, context=context)
        
        prompts = []
        raw_prompts = result.get("prompts", [])
        logger.debug(f"[{context}] LLM returned {len(raw_prompts)} prompts (requested {count})")
        
        for i, p in enumerate(raw_prompts):
            prompt_id = self._get_next_prompt_id()
            prompt_text = p.get("prompt_text", "")
            
            if not prompt_text:
                logger.warning(f"[{context}] Prompt {i+1} has empty text, skipping")
                continue
                
            prompt_obj = {
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "category": category,
                "difficulty": difficulty,
                "persona": p.get("persona", random.choice(PERSONAS)),
                "tags": p.get("tags", [category]),
                "created_in_iteration": iteration,
                "difficulty_rationale": p.get("difficulty_rationale", ""),
                "category_rationale": p.get("category_rationale", "")
            }
            prompts.append(prompt_obj)
            logger.debug(f"[{context}] Created prompt {prompt_id}: {prompt_text[:60]}...")
        
        logger.info(f"[{context}] Successfully generated {len(prompts)}/{count} prompts")
        return prompts
    
    def generate_initial_prompts(self, total_count: int, iteration: int = 0) -> List[Dict[str, Any]]:
        """Generate initial seed prompts distributed across categories and difficulties."""
        
        logger.info(f"=== INITIAL PROMPT GENERATION: target={total_count} prompts ===")
        
        all_prompts = []
        cells = [(cat, diff) for cat in CATEGORIES for diff in range(1, 6)]
        prompts_per_cell = max(1, total_count // len(cells))
        remainder = total_count - (prompts_per_cell * len(cells))
        
        logger.info(f"Distribution: {len(cells)} cells, ~{prompts_per_cell} prompts/cell, remainder={remainder}")
        
        random.shuffle(cells)
        
        for i, (category, difficulty) in enumerate(cells):
            count = prompts_per_cell + (1 if i < remainder else 0)
            if count > 0:
                logger.info(f"[PROGRESS] Cell {i+1}/{len(cells)}: {category}/D{difficulty} - generating {count} prompts")
                try:
                    batch = self.generate_prompts_batch(
                        category=category,
                        difficulty=difficulty,
                        count=count,
                        iteration=iteration
                    )
                    all_prompts.extend(batch)
                    logger.info(f"[PROGRESS] Cell {i+1}/{len(cells)} complete. Total prompts so far: {len(all_prompts)}")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to generate prompts for {category}/D{difficulty}: {e}")
                    raise
        
        logger.info(f"=== INITIAL GENERATION COMPLETE: {len(all_prompts)}/{total_count} prompts ===")
        logger.info(f"API stats: {self.total_api_calls} calls, {self.total_api_time:.2f}s total time")
        return all_prompts
    
    def generate_targeted_prompts(
        self,
        gaps: Dict[str, Dict[int, int]],
        iteration: int,
        existing_prompts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate prompts to fill specific gaps in the matrix."""
        
        total_gaps = sum(sum(d.values()) for d in gaps.values())
        num_cells = sum(len(d) for d in gaps.values())
        logger.info(f"=== TARGETED PROMPT GENERATION: {total_gaps} prompts across {num_cells} cells ===")
        
        all_prompts = []
        existing_texts = [p.get("prompt_text", "") for p in existing_prompts]
        logger.debug(f"Using {len(existing_texts)} existing prompts as negative examples")
        
        cell_idx = 0
        for category, difficulties in gaps.items():
            for difficulty, needed in difficulties.items():
                if needed > 0:
                    cell_idx += 1
                    logger.info(f"[PROGRESS] Gap {cell_idx}/{num_cells}: {category}/D{difficulty} - need {needed} prompts")
                    try:
                        batch = self.generate_prompts_batch(
                            category=category,
                            difficulty=int(difficulty),
                            count=needed,
                            iteration=iteration,
                            existing_prompts=existing_texts
                        )
                        all_prompts.extend(batch)
                        logger.info(f"[PROGRESS] Gap {cell_idx}/{num_cells} filled. Total new prompts: {len(all_prompts)}")
                    except Exception as e:
                        logger.error(f"[ERROR] Failed to fill gap {category}/D{difficulty}: {e}")
                        raise
        
        logger.info(f"=== TARGETED GENERATION COMPLETE: {len(all_prompts)}/{total_gaps} prompts ===")
        logger.info(f"API stats: {self.total_api_calls} calls, {self.total_api_time:.2f}s total time")
        return all_prompts
    
    def load_existing_prompts(self) -> List[Dict[str, Any]]:
        """Load existing prompts from file."""
        logger.info(f"Loading existing prompts from {PROMPT_FILE}")
        prompts = []
        if PROMPT_FILE.exists():
            with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        prompts.append(json.loads(line))
            if prompts:
                max_id = max(int(p['prompt_id'][1:]) for p in prompts)
                self.prompt_counter = max_id
                logger.info(f"Loaded {len(prompts)} existing prompts, max_id={max_id}")
            else:
                logger.info("Prompt file exists but is empty")
        else:
            logger.info("No existing prompt file found, starting fresh")
        return prompts
    
    def save_prompts(self, prompts: List[Dict[str, Any]], append: bool = False):
        """Save prompts to JSONL file."""
        mode = 'a' if append else 'w'
        logger.info(f"Saving {len(prompts)} prompts to {PROMPT_FILE} (mode={mode})")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(PROMPT_FILE, mode, encoding='utf-8') as f:
            for prompt in prompts:
                save_obj = {
                    "prompt_id": prompt["prompt_id"],
                    "prompt_text": prompt["prompt_text"],
                    "category": prompt["category"],
                    "difficulty": prompt["difficulty"],
                    "persona": prompt["persona"],
                    "tags": prompt["tags"],
                    "created_in_iteration": prompt["created_in_iteration"]
                }
                f.write(json.dumps(save_obj) + '\n')
        logger.info(f"Successfully saved {len(prompts)} prompts")

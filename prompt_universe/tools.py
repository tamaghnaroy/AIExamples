"""
Tool management module for FX/macro hedge fund agent tools.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI

try:
    from .config import (
        OPENAI_API_KEY, OPENAI_MODEL, RANDOM_SEED,
        TOOL_CALLS_FILE, DATA_DIR, MAX_LLM_RETRIES,
        CATEGORIES
    )
    from .logging_config import get_logger
except ImportError:
    from config import (
        OPENAI_API_KEY, OPENAI_MODEL, RANDOM_SEED,
        TOOL_CALLS_FILE, DATA_DIR, MAX_LLM_RETRIES,
        CATEGORIES
    )
    from logging_config import get_logger

logger = get_logger('tools')


class ToolManager:
    """Manages tool definitions for the FX agent."""
    
    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.total_api_calls = 0
        self.total_api_time = 0.0
        logger.info(f"ToolManager initialized with seed={seed}, model={self.model}")
        
    def _call_llm_json(self, system_prompt: str, user_prompt: str, retries: int = MAX_LLM_RETRIES, context: str = "tools") -> Dict[str, Any]:
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
                    time.sleep(2 ** attempt)
                    continue
                logger.error(f"[{context}] API call failed after {retries} attempts")
                raise
        return {}
    
    def generate_initial_tools(self, count: int = 30) -> List[Dict[str, Any]]:
        """Generate initial set of tool definitions."""
        
        logger.info(f"=== INITIAL TOOL GENERATION: target={count} tools ===")
        
        system_prompt = """You are an expert at designing tool APIs for FX/macro hedge fund AI assistants.

You must create tool definitions that:
- Are realistic and useful for hedge fund operations
- Cover data retrieval, analytics, pricing, risk, and workflow needs
- Have clear input/output schemas
- Include practical examples

Return a JSON object with a "tools" array."""

        user_prompt = f"""Generate {count} tool definitions for an FX/macro hedge fund AI assistant.

Categories to cover: {', '.join(CATEGORIES)}

Each tool should have this structure:
{{
  "tool_name": "snake_case_name",
  "tool_description": "Clear description of what the tool does",
  "tool_inputs": {{
    "param_name": "type and description"
  }},
  "tool_outputs": {{
    "schema": "table|scalar|object|array",
    "fields": ["field1", "field2"]
  }},
  "examples": [
    {{
      "inputs": {{"param": "value"}},
      "output_description": "What the output looks like"
    }}
  ]
}}

Return JSON format:
{{
  "tools": [...]
}}"""

        result = self._call_llm_json(system_prompt, user_prompt, context="initial_tools")
        tools = result.get("tools", [])
        logger.info(f"=== INITIAL TOOL GENERATION COMPLETE: {len(tools)}/{count} tools ===")
        for tool in tools:
            logger.debug(f"  Tool: {tool.get('tool_name', 'unnamed')}")
        return tools
    
    def load_tools(self) -> List[Dict[str, Any]]:
        """Load tools from file."""
        logger.info(f"Loading tools from {TOOL_CALLS_FILE}")
        if not TOOL_CALLS_FILE.exists():
            logger.info("No existing tool file found, starting fresh")
            return []
        
        with open(TOOL_CALLS_FILE, 'r', encoding='utf-8') as f:
            tools = json.load(f)
        logger.info(f"Loaded {len(tools)} existing tools")
        return tools
    
    def save_tools(self, tools: List[Dict[str, Any]]):
        """Save tools to file."""
        logger.info(f"Saving {len(tools)} tools to {TOOL_CALLS_FILE}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(TOOL_CALLS_FILE, 'w', encoding='utf-8') as f:
            json.dump(tools, f, indent=2)
        logger.info(f"Successfully saved {len(tools)} tools")
    
    def merge_tools(self, existing: List[Dict[str, Any]], new_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge new tools into existing, avoiding duplicates."""
        logger.info(f"Merging {len(new_tools)} new tools into {len(existing)} existing tools")
        existing_names = {t['tool_name'] for t in existing}
        
        merged = list(existing)
        added = 0
        skipped = 0
        for tool in new_tools:
            tool_name = tool.get('tool_name')
            if tool_name not in existing_names:
                merged.append(tool)
                existing_names.add(tool_name)
                added += 1
                logger.debug(f"  Added new tool: {tool_name}")
            else:
                skipped += 1
                logger.debug(f"  Skipped duplicate: {tool_name}")
        
        logger.info(f"Merge complete: added={added}, skipped={skipped}, total={len(merged)}")
        return merged
    
    def get_tools_summary(self, tools: List[Dict[str, Any]]) -> str:
        """Get a text summary of tools for LLM prompts."""
        lines = [f"Tool Set Summary ({len(tools)} tools):"]
        
        for tool in tools:
            lines.append(f"- {tool.get('tool_name', 'unnamed')}: {tool.get('tool_description', '')[:100]}")
        
        return "\n".join(lines)
    
    def format_tools_for_evaluation(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools as JSON string for LLM evaluation."""
        return json.dumps(tools, indent=2)

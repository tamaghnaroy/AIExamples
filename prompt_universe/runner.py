"""
Runner module - CLI orchestrator for prompt universe generation.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    from .config import (
        CATEGORIES, MIN_PROMPTS_PER_CELL, MIN_OVERALL_SCORE,
        INITIAL_PROMPT_COUNT, INITIAL_TOOL_COUNT,
        TOOL_VALIDATION_ITERATIONS, TOOL_VALIDATION_SAMPLE_SIZE,
        DATA_DIR, ITERATIONS_DIR, RANDOM_SEED
    )
    from .generator import PromptGenerator
    from .matrix import MatrixManager
    from .tools import ToolManager
    from .evaluator import Evaluator
    from .logging_config import setup_logging, get_logger
except ImportError:
    from config import (
        CATEGORIES, MIN_PROMPTS_PER_CELL, MIN_OVERALL_SCORE,
        INITIAL_PROMPT_COUNT, INITIAL_TOOL_COUNT,
        TOOL_VALIDATION_ITERATIONS, TOOL_VALIDATION_SAMPLE_SIZE,
        DATA_DIR, ITERATIONS_DIR, RANDOM_SEED
    )
    from generator import PromptGenerator
    from matrix import MatrixManager
    from tools import ToolManager
    from evaluator import Evaluator
    from logging_config import setup_logging, get_logger


class Runner:
    """Orchestrates the prompt universe generation and evaluation loop."""
    
    def __init__(self, seed: int = RANDOM_SEED, verbose: bool = True):
        setup_logging(verbose=verbose)
        self.logger = get_logger('runner')
        
        self.seed = seed
        self.start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("PROMPT UNIVERSE GENERATOR - INITIALIZATION")
        self.logger.info("=" * 60)
        self.logger.info(f"Seed: {seed}")
        self.logger.info(f"Target: {MIN_PROMPTS_PER_CELL} prompts per cell, score >= {MIN_OVERALL_SCORE}")
        self.logger.info(f"Initial targets: {INITIAL_PROMPT_COUNT} prompts, {INITIAL_TOOL_COUNT} tools")
        self.logger.info(f"Tool validation: {TOOL_VALIDATION_ITERATIONS} iterations, {TOOL_VALIDATION_SAMPLE_SIZE} samples each")
        
        self.generator = PromptGenerator(seed=seed)
        self.matrix_mgr = MatrixManager()
        self.tool_mgr = ToolManager(seed=seed)
        self.evaluator = Evaluator(seed=seed)
        self.current_iteration = 0
        
        self.logger.info("All components initialized successfully")
        
    def log(self, message: str):
        """Print timestamped log message."""
        self.logger.info(message)
    
    def _get_elapsed_time(self) -> str:
        """Get elapsed time since start."""
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _log_progress_summary(self, prompts: list, tools: list, matrix: dict):
        """Log a summary of current progress."""
        total_prompts = len(prompts)
        total_tools = len(tools)
        total_cells = len(self.matrix_mgr.categories) * 5
        
        filled_cells = 0
        min_cell = float('inf')
        max_cell = 0
        for diff in matrix:
            for cat, count in matrix[diff].items():
                if count >= MIN_PROMPTS_PER_CELL:
                    filled_cells += 1
                min_cell = min(min_cell, count)
                max_cell = max(max_cell, count)
        
        self.logger.info("-" * 40)
        self.logger.info(f"PROGRESS SUMMARY [Elapsed: {self._get_elapsed_time()}]")
        self.logger.info(f"  Total prompts: {total_prompts}")
        self.logger.info(f"  Total tools: {total_tools}")
        self.logger.info(f"  Categories: {len(self.matrix_mgr.categories)}")
        self.logger.info(f"  Cells filled (>={MIN_PROMPTS_PER_CELL}): {filled_cells}/{total_cells}")
        self.logger.info(f"  Cell range: min={min_cell}, max={max_cell}")
        self.logger.info("-" * 40)
    
    def run_iteration_0(self):
        """Run initial seed generation (iteration 0)."""
        self.logger.info("=" * 60)
        self.logger.info("ITERATION 0: SEED GENERATION")
        self.logger.info("=" * 60)
        
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ITERATIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Data directory: {DATA_DIR}")
        self.logger.debug(f"Iterations directory: {ITERATIONS_DIR}")
        
        self.logger.info(f"[STEP 1/3] Generating {INITIAL_PROMPT_COUNT} initial prompts...")
        iter0_start = time.time()
        prompts = self.generator.generate_initial_prompts(
            total_count=INITIAL_PROMPT_COUNT,
            iteration=0
        )
        self.generator.save_prompts(prompts, append=False)
        self.logger.info(f"[STEP 1/3] Complete: Generated {len(prompts)} prompts in {time.time()-iter0_start:.1f}s")
        
        self.logger.info(f"[STEP 2/3] Generating {INITIAL_TOOL_COUNT} initial tools...")
        tools_start = time.time()
        tools = self.tool_mgr.generate_initial_tools(count=INITIAL_TOOL_COUNT)
        self.tool_mgr.save_tools(tools)
        self.logger.info(f"[STEP 2/3] Complete: Generated {len(tools)} tools in {time.time()-tools_start:.1f}s")
        
        self.logger.info("[STEP 3/3] Computing and saving matrix...")
        matrix = self.matrix_mgr.compute_matrix(prompts)
        self.matrix_mgr.save_matrix(matrix)
        self.matrix_mgr.save_prompt_categories(prompts)
        self.logger.info("[STEP 3/3] Complete: Matrix saved")
        
        self._log_progress_summary(prompts, tools, matrix)
        self.logger.info("Iteration 0 complete. Proceeding to evaluation loop...")
        return prompts, tools, matrix
    
    def run_prompt_iteration(self, iteration: int, prompts: list, tools: list) -> tuple:
        """Run a single prompt enrichment iteration."""
        iter_start = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info(f"ITERATION {iteration}: PROMPT ENRICHMENT [Elapsed: {self._get_elapsed_time()}]")
        self.logger.info("=" * 60)
        
        self.logger.info("[STEP 1/6] Computing matrix and sampling prompts...")
        matrix = self.matrix_mgr.compute_matrix(prompts)
        sampled = self.matrix_mgr.sample_prompts_per_cell(prompts, max_per_cell=5)
        categories = sorted(self.matrix_mgr.categories)
        self.logger.debug(f"Matrix computed: {len(categories)} categories, {len(prompts)} prompts")
        
        self.logger.info("[STEP 2/6] Evaluating prompt universe...")
        prompt_eval = self.evaluator.score_prompt_universe(categories, matrix, sampled)
        overall_score = prompt_eval.get('overall_score', 0)
        self.logger.info(f"[STEP 2/6] Prompt universe score: {overall_score}/10")
        
        new_categories = prompt_eval.get('recommended_new_categories', [])
        missing_categories = prompt_eval.get('missing_categories', [])
        all_new = list(set(new_categories + missing_categories))
        if all_new:
            self.logger.info(f"[STEP 2/6] Adding {len(all_new)} new categories: {all_new}")
            self.matrix_mgr.add_categories(all_new)
            matrix = self.matrix_mgr.compute_matrix(prompts)
        
        self.logger.info("[STEP 3/6] Evaluating tool set...")
        matrix_summary = self.matrix_mgr.get_matrix_summary(matrix)
        tool_eval = self.evaluator.score_tool_set(tools, categories, matrix_summary)
        tool_score = tool_eval.get('tool_set_score', 0)
        self.logger.info(f"[STEP 3/6] Tool set score: {tool_score}/10")
        
        missing_tools = tool_eval.get('missing_tooling_for_categories', {})
        if missing_tools:
            self.logger.info(f"[STEP 4/6] Generating tools for {len(missing_tools)} categories with gaps...")
            new_tools = []
            for cat, tool_ideas in missing_tools.items():
                for idea in tool_ideas[:2]:
                    tool_name = idea.lower().replace(' ', '_').replace('-', '_')[:50]
                    new_tools.append({
                        "tool_name": tool_name,
                        "tool_description": idea,
                        "tool_inputs": {},
                        "tool_outputs": {"schema": "object", "fields": []},
                        "examples": []
                    })
                    self.logger.debug(f"  New tool idea: {tool_name}")
            tools = self.tool_mgr.merge_tools(tools, new_tools)
            self.tool_mgr.save_tools(tools)
            self.logger.info(f"[STEP 4/6] Added {len(new_tools)} tool definitions")
        else:
            self.logger.info("[STEP 4/6] No missing tools identified")
        
        self.logger.info("[STEP 5/6] Analyzing gaps and generating prompts...")
        gaps = self.matrix_mgr.find_gaps(matrix, MIN_PROMPTS_PER_CELL)
        total_needed = sum(sum(d.values()) for d in gaps.values())
        num_gap_cells = sum(len(d) for d in gaps.values())
        
        if total_needed > 0:
            batch_size = min(total_needed, 500)
            self.logger.info(f"[STEP 5/6] Gap analysis: {num_gap_cells} cells need {total_needed} prompts total")
            self.logger.info(f"[STEP 5/6] Generating batch of up to {batch_size} prompts...")
            
            scaled_gaps = {}
            scale_factor = batch_size / total_needed if total_needed > batch_size else 1.0
            for cat, diffs in gaps.items():
                scaled_gaps[cat] = {}
                for diff, count in diffs.items():
                    scaled_count = max(1, int(count * scale_factor))
                    scaled_gaps[cat][diff] = scaled_count
            
            gen_start = time.time()
            new_prompts = self.generator.generate_targeted_prompts(
                gaps=scaled_gaps,
                iteration=iteration,
                existing_prompts=prompts
            )
            gen_time = time.time() - gen_start
            
            if new_prompts:
                self.generator.save_prompts(new_prompts, append=True)
                prompts.extend(new_prompts)
                self.logger.info(f"[STEP 5/6] Generated {len(new_prompts)} new prompts in {gen_time:.1f}s")
        else:
            self.logger.info("[STEP 5/6] No gaps to fill - matrix is complete!")
        
        self.logger.info("[STEP 6/6] Saving iteration results...")
        matrix = self.matrix_mgr.compute_matrix(prompts)
        self.matrix_mgr.save_matrix(matrix)
        self.matrix_mgr.save_prompt_categories(prompts)
        
        stats = {
            "iteration": iteration,
            "total_prompts": len(prompts),
            "total_tools": len(tools),
            "overall_score": overall_score,
            "tool_score": tool_score,
            "gaps_remaining": total_needed,
            "matrix_complete": self.matrix_mgr.is_complete(matrix),
            "iteration_time_seconds": time.time() - iter_start
        }
        self.evaluator.save_iteration_results(iteration, prompt_eval, tool_eval, matrix, stats)
        
        self._log_progress_summary(prompts, tools, matrix)
        self.logger.info(f"Iteration {iteration} complete in {time.time()-iter_start:.1f}s")
        
        return prompts, tools, matrix, overall_score
    
    def run_tool_validation_loop(self, prompts: list, tools: list) -> list:
        """Run tool validation loop after prompt generation stops."""
        self.logger.info("=" * 60)
        self.logger.info(f"TOOL VALIDATION LOOP [Elapsed: {self._get_elapsed_time()}]")
        self.logger.info("=" * 60)
        self.logger.info(f"Running {TOOL_VALIDATION_ITERATIONS} validation iterations with {TOOL_VALIDATION_SAMPLE_SIZE} samples each")
        
        categories = sorted(self.matrix_mgr.categories)
        all_missing_tools = []
        
        for i in range(TOOL_VALIDATION_ITERATIONS):
            iter_start = time.time()
            self.logger.info(f"[VALIDATION {i+1}/{TOOL_VALIDATION_ITERATIONS}] Starting...")
            
            sampled, redistribution = self.evaluator.sample_prompts_uniformly(
                prompts, categories, TOOL_VALIDATION_SAMPLE_SIZE
            )
            
            if redistribution:
                self.logger.debug(f"[VALIDATION {i+1}] Redistribution needed for {len(redistribution)} cells")
            
            result = self.evaluator.validate_tools_against_prompts(sampled, tools)
            cannot_answer = result.get('cannot_answer', [])
            
            if cannot_answer:
                self.logger.info(f"[VALIDATION {i+1}] Found {len(cannot_answer)} prompts that cannot be answered")
                for item in cannot_answer:
                    suggested = item.get('suggested_tool', {})
                    if suggested and suggested.get('tool_name'):
                        all_missing_tools.append(suggested)
                        self.logger.debug(f"  Suggested tool: {suggested.get('tool_name')}")
            else:
                self.logger.info(f"[VALIDATION {i+1}] All {len(sampled)} sampled prompts can be answered")
            
            self.logger.info(f"[VALIDATION {i+1}] Complete in {time.time()-iter_start:.1f}s")
        
        if all_missing_tools:
            seen_names = set()
            unique_tools = []
            for tool in all_missing_tools:
                name = tool.get('tool_name', '')
                if name and name not in seen_names:
                    seen_names.add(name)
                    unique_tools.append(tool)
            
            self.logger.info(f"Adding {len(unique_tools)} unique new tools from validation (from {len(all_missing_tools)} suggestions)")
            tools = self.tool_mgr.merge_tools(tools, unique_tools)
            self.tool_mgr.save_tools(tools)
        else:
            self.logger.info("Tool validation complete - no missing tools found across all iterations")
        
        self.logger.info(f"Final tool count: {len(tools)}")
        return tools
    
    def check_prompt_stopping_criteria(self, matrix: dict, overall_score: float) -> bool:
        """Check if prompt generation should stop."""
        matrix_complete = self.matrix_mgr.is_complete(matrix, MIN_PROMPTS_PER_CELL)
        score_met = overall_score >= MIN_OVERALL_SCORE
        
        self.logger.info("-" * 40)
        self.logger.info("STOPPING CRITERIA CHECK")
        self.logger.info(f"  Matrix complete (>={MIN_PROMPTS_PER_CELL}/cell): {matrix_complete}")
        self.logger.info(f"  Score met (>={MIN_OVERALL_SCORE}): {score_met} (current: {overall_score})")
        self.logger.info(f"  Should stop: {matrix_complete and score_met}")
        self.logger.info("-" * 40)
        
        return matrix_complete and score_met
    
    def run(self, max_iterations: int = 100):
        """Run the full generation and evaluation loop."""
        self.logger.info("=" * 60)
        self.logger.info("STARTING PROMPT UNIVERSE GENERATION")
        self.logger.info("=" * 60)
        self.logger.info(f"Max iterations: {max_iterations}")
        
        self.logger.info("Checking for existing data...")
        prompts = self.generator.load_existing_prompts()
        tools = self.tool_mgr.load_tools()
        
        if not prompts or not tools:
            self.logger.info("No existing data found. Starting fresh with iteration 0...")
            prompts, tools, matrix = self.run_iteration_0()
            self.current_iteration = 1
        else:
            self.logger.info(f"Resuming from existing data: {len(prompts)} prompts, {len(tools)} tools")
            matrix = self.matrix_mgr.compute_matrix(prompts)
            self._log_progress_summary(prompts, tools, matrix)
            self.current_iteration = 1
        
        overall_score = 0
        
        self.logger.info("=" * 60)
        self.logger.info("ENTERING MAIN ITERATION LOOP")
        self.logger.info("=" * 60)
        
        while self.current_iteration <= max_iterations:
            self.logger.info(f">>> Starting iteration {self.current_iteration}/{max_iterations}")
            
            prompts, tools, matrix, overall_score = self.run_prompt_iteration(
                self.current_iteration, prompts, tools
            )
            
            if self.check_prompt_stopping_criteria(matrix, overall_score):
                self.logger.info("*** PROMPT STOPPING CRITERIA MET! ***")
                break
            
            self.current_iteration += 1
            self.logger.info(f"<<< Proceeding to iteration {self.current_iteration}")
        
        if self.current_iteration > max_iterations:
            self.logger.warning(f"*** REACHED MAX ITERATIONS ({max_iterations}) WITHOUT MEETING CRITERIA ***")
        
        tools = self.run_tool_validation_loop(prompts, tools)
        
        total_time = time.time() - self.start_time
        hours, remainder = divmod(int(total_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.logger.info("=" * 60)
        self.logger.info("GENERATION COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        self.logger.info(f"Final prompts: {len(prompts)}")
        self.logger.info(f"Final tools: {len(tools)}")
        self.logger.info(f"Final score: {overall_score}")
        self.logger.info(f"Iterations completed: {self.current_iteration}")
        self.logger.info("=" * 60)
        
        return prompts, tools


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prompt Universe Generator for FX/Macro Hedge Fund LLM Agent"
    )
    parser.add_argument(
        '--run',
        action='store_true',
        help='Run the generation loop'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=100,
        help='Maximum number of iterations (default: 100)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {RANDOM_SEED})'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        default=True,
        help='Enable verbose (DEBUG) logging (default: True)'
    )
    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Reduce logging to INFO level only'
    )
    
    args = parser.parse_args()
    
    if args.run:
        verbose = not args.quiet
        runner = Runner(seed=args.seed, verbose=verbose)
        runner.run(max_iterations=args.max_iterations)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

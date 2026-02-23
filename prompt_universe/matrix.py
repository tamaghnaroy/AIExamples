"""
Matrix computation module for prompt coverage analysis.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict

try:
    from .config import (
        CATEGORIES, DIFFICULTY_SCALE,
        PROMPT_FILE, PROMPT_CATEGORY_FILE, PROMPT_MATRIX_FILE,
        DATA_DIR, MIN_PROMPTS_PER_CELL
    )
except ImportError:
    from config import (
        CATEGORIES, DIFFICULTY_SCALE,
        PROMPT_FILE, PROMPT_CATEGORY_FILE, PROMPT_MATRIX_FILE,
        DATA_DIR, MIN_PROMPTS_PER_CELL
    )


class MatrixManager:
    """Manages prompt matrix computation and category tracking."""
    
    def __init__(self):
        self.categories = set(CATEGORIES)
        self.difficulties = list(range(1, 6))
        
    def add_categories(self, new_categories: List[str]):
        """Add new categories to the tracking set."""
        for cat in new_categories:
            if cat and cat not in self.categories:
                self.categories.add(cat)
    
    def compute_matrix(self, prompts: List[Dict[str, Any]]) -> Dict[int, Dict[str, int]]:
        """Compute the prompt matrix from prompts list."""
        matrix = {d: {cat: 0 for cat in sorted(self.categories)} for d in self.difficulties}
        
        for prompt in prompts:
            difficulty = prompt.get('difficulty', 1)
            category = prompt.get('category', '')
            
            if category not in self.categories:
                self.categories.add(category)
                for d in self.difficulties:
                    matrix[d][category] = 0
            
            if difficulty in matrix and category in matrix[difficulty]:
                matrix[difficulty][category] += 1
        
        return matrix
    
    def find_gaps(self, matrix: Dict[int, Dict[str, int]], min_count: int = MIN_PROMPTS_PER_CELL) -> Dict[str, Dict[int, int]]:
        """Find cells that need more prompts."""
        gaps = defaultdict(dict)
        
        for difficulty, categories in matrix.items():
            for category, count in categories.items():
                if count < min_count:
                    needed = min_count - count
                    gaps[category][difficulty] = needed
        
        return dict(gaps)
    
    def is_complete(self, matrix: Dict[int, Dict[str, int]], min_count: int = MIN_PROMPTS_PER_CELL) -> bool:
        """Check if all cells meet minimum count."""
        for difficulty, categories in matrix.items():
            for category, count in categories.items():
                if count < min_count:
                    return False
        return True
    
    def save_matrix(self, matrix: Dict[int, Dict[str, int]]):
        """Save matrix to CSV file."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        categories = sorted(self.categories)
        
        with open(PROMPT_MATRIX_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['difficulty'] + categories)
            
            for difficulty in self.difficulties:
                row = [difficulty] + [matrix[difficulty].get(cat, 0) for cat in categories]
                writer.writerow(row)
    
    def load_matrix(self) -> Dict[int, Dict[str, int]]:
        """Load matrix from CSV file."""
        if not PROMPT_MATRIX_FILE.exists():
            return {d: {cat: 0 for cat in self.categories} for d in self.difficulties}
        
        matrix = {}
        with open(PROMPT_MATRIX_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                difficulty = int(row['difficulty'])
                matrix[difficulty] = {}
                for key, value in row.items():
                    if key != 'difficulty':
                        matrix[difficulty][key] = int(value)
                        self.categories.add(key)
        
        return matrix
    
    def save_prompt_categories(self, prompts: List[Dict[str, Any]]):
        """Save prompt category details to CSV."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(PROMPT_CATEGORY_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'prompt_id', 'category', 'difficulty', 'persona',
                'difficulty_rationale', 'category_rationale'
            ])
            
            for prompt in prompts:
                writer.writerow([
                    prompt.get('prompt_id', ''),
                    prompt.get('category', ''),
                    prompt.get('difficulty', ''),
                    prompt.get('persona', ''),
                    prompt.get('difficulty_rationale', ''),
                    prompt.get('category_rationale', '')
                ])
    
    def get_matrix_summary(self, matrix: Dict[int, Dict[str, int]]) -> str:
        """Get a text summary of the matrix for LLM prompts."""
        lines = ["Prompt Matrix Summary:"]
        lines.append(f"Categories: {len(self.categories)}")
        lines.append(f"Difficulty levels: 1-5")
        lines.append("")
        
        total = 0
        for difficulty in self.difficulties:
            row_total = sum(matrix[difficulty].values())
            total += row_total
            lines.append(f"Difficulty {difficulty}: {row_total} prompts")
        
        lines.append(f"\nTotal prompts: {total}")
        
        gaps = self.find_gaps(matrix)
        if gaps:
            lines.append(f"\nGaps (cells below {MIN_PROMPTS_PER_CELL}):")
            for cat, diffs in list(gaps.items())[:10]:
                lines.append(f"  {cat}: {diffs}")
            if len(gaps) > 10:
                lines.append(f"  ... and {len(gaps) - 10} more categories with gaps")
        
        return "\n".join(lines)
    
    def sample_prompts_per_cell(
        self,
        prompts: List[Dict[str, Any]],
        max_per_cell: int = 5
    ) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
        """Sample prompts from each cell for evaluation."""
        
        grouped = defaultdict(lambda: defaultdict(list))
        for prompt in prompts:
            cat = prompt.get('category', '')
            diff = prompt.get('difficulty', 1)
            grouped[cat][diff].append(prompt)
        
        sampled = {}
        for cat, diffs in grouped.items():
            sampled[cat] = {}
            for diff, prompt_list in diffs.items():
                import random
                sampled[cat][diff] = random.sample(prompt_list, min(max_per_cell, len(prompt_list)))
        
        return sampled

"""
Standalone runner script for prompt_universe module.
Run this directly: python run_prompt_universe.py --run
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from runner import main

if __name__ == '__main__':
    main()

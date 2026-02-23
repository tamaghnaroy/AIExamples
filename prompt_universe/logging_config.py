"""
Centralized logging configuration for prompt_universe module.
Provides detailed logging to both console and file for progress tracking.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

MODULE_DIR = Path(__file__).parent
LOG_DIR = MODULE_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"prompt_universe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def setup_logging(verbose: bool = True) -> logging.Logger:
    """
    Setup logging with both console and file handlers.
    
    Args:
        verbose: If True, show DEBUG level on console. Otherwise INFO.
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger('prompt_universe')
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {LOG_FILE}")
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a child logger for a specific module."""
    base_logger = logging.getLogger('prompt_universe')
    if name:
        return base_logger.getChild(name)
    return base_logger

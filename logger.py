"""
Trading Bot Logger Module
Provides structured logging with file and console output.
"""

import logging
import os
from datetime import datetime
from config import LOG_FILE, LOG_LEVEL

def setup_logger(name: str) -> logging.Logger:
    """Create and configure a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger("TradingBot")
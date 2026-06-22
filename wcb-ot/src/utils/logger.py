"""
Logger utility for WCB-OT project.
Uses loguru for colorized, rotatable, and structured logging.
"""

import sys
from pathlib import Path
from datetime import datetime
from loguru import logger
from typing import Optional

def get_logger(name: str, log_file: Optional[Path] = None) -> "logger":
    """
    Configures and returns a loguru logger instance.

    Args:
        name (str): The name of the module or process for the log file prefix.
        log_file (Optional[Path]): Explicit path to the log file. 
            If None, defaults to results/logs/{name}_{timestamp}.log.

    Returns:
        loguru.Logger: A configured logger instance.
    """
    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>',
        level="INFO",
        colorize=True
    )

    # File handler naming
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("results/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}_{timestamp}.log"

    # File handler
    logger.add(
        log_file,
        format='{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}',
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )

    return logger.bind(name=name)

if __name__ == "__main__":
    test_log = get_logger("test_module")
    test_log.info("Logger initialized successfully.")

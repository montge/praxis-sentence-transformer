# src/logger/logger.py

import os
import logging
import time
from functools import wraps
from typing import Optional, Callable
from datetime import datetime

# ANSI color codes
COLORS = {
    'DEBUG': '\033[36m',     # Cyan
    'INFO': '\033[32m',      # Green
    'WARNING': '\033[33m',   # Yellow
    'ERROR': '\033[31m',     # Red
    'CRITICAL': '\033[41m',  # Red background
    'RESET': '\033[0m'       # Reset color
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels"""
    
    def format(self, record):
        # Save original levelname
        orig_levelname = record.levelname
        # Add color to levelname
        record.levelname = f"{COLORS.get(record.levelname, '')}{record.levelname}{COLORS['RESET']}"
        
        # Add color to module/logger name
        record.name = f"\033[35m{record.name}{COLORS['RESET']}"  # Purple
        
        # Format the message
        result = super().format(record)
        
        # Restore original levelname
        record.levelname = orig_levelname
        return result

def setup_logging(logger_name: str, default_level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration with colored output and environment variable override
    
    Parameters:
        logger_name (str): Name of the logger
        default_level (int): Default logging level if not specified in environment
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get logging level from environment variable
    log_level = os.getenv('LOGGING_LEVEL', 'INFO').upper()
    level = getattr(logging, log_level, default_level)
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Create console handler if none exists
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Create colored formatter
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        logger.debug(f"Logger {logger_name} initialized with level {log_level}")
    
    return logger

def log_exception(logger: logging.Logger, message: str, exception: Exception) -> None:
    """
    Handle and log an exception with colored output
    
    Args:
        logger: Logger instance
        message: Error message
        exception: Exception that was caught
    """
    error_msg = f"{COLORS['ERROR']}{message}: {str(exception)}{COLORS['RESET']}"
    logger.error(error_msg, exc_info=True)

def handle_exception(func):
    """
    Decorator to handle exceptions and log them appropriately with colors
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"{COLORS['ERROR']}Error in {func.__name__}: {str(e)}{COLORS['RESET']}"
            logger.error(error_msg)
            logger.debug(f"Function arguments: args={args}, kwargs={kwargs}")
            logger.exception(f"{COLORS['ERROR']}Detailed error trace:{COLORS['RESET']}")
            raise
    return wrapper

class DebugTimer:
    """Timer utility for debugging and performance monitoring with colored output"""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize timer
        
        Args:
            name: Timer name for identification
            logger: Optional logger instance
        """
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.checkpoints = []
        self._duration = 0  # Add duration storage
        
    def start(self) -> None:
        """Start the timer"""
        self.start_time = time.time()
        self.checkpoints = []
        self.logger.debug(f"{COLORS['DEBUG']}{self.name} timer started{COLORS['RESET']}")
        
    def checkpoint(self, description: str) -> None:
        """
        Record a checkpoint with description
        
        Args:
            description: Description of the checkpoint
        """
        if self.start_time is None:
            self.start()
            
        current_time = time.time()
        elapsed = current_time - self.start_time
        checkpoint_msg = f"{COLORS['DEBUG']}{self.name} - {description}: {elapsed:.2f}s{COLORS['RESET']}"
        self.checkpoints.append((description, elapsed))
        self.logger.debug(checkpoint_msg)
        
    def end(self) -> None:
        """End timing and log final results"""
        if self.start_time is None:
            self.logger.warning(f"{COLORS['WARNING']}Timer {self.name} ended without being started{COLORS['RESET']}")
            return
            
        self._duration = time.time() - self.start_time  # Store duration
        end_msg = f"{COLORS['INFO']}{self.name} completed in {self._duration:.2f}s{COLORS['RESET']}"
        self.logger.info(end_msg)
        self.start_time = None

    @property
    def duration(self) -> float:
        """Get the total duration of the timer
        
        Returns:
            float: Duration in seconds
        """
        return self._duration
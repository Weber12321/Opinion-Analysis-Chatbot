# app/utils/logger.py
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(name=None):
    """
    Set up a logger with both console and file handlers
    
    Args:
        name: The name of the logger (default: None)
        
    Returns:
        A configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name or 'opinion_analysis_bot')
    
    # Set level
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logger.setLevel(getattr(logging, log_level))
    
    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'opinion_analysis_bot.log'),
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Define log levels as constants
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# Default log format with timestamps, level, and module information
DEFAULT_LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'

# Global variable to track if logger has been configured
_logger_configured = False

def configure_logger(
    log_level=logging.INFO,
    log_format=DEFAULT_LOG_FORMAT,
    log_file=None,
    console_output=True,
    log_dir='logs',
    app_name='tft_pipeline'
):
    """
    Configure the root logger with handlers based on parameters.
    
    Args:
        log_level: Minimum log level to record (default: INFO)
        log_format: Format string for log messages
        log_file: Optional specific log file path 
        console_output: Whether to output logs to console
        log_dir: Directory to store log files if log_file isn't specified
        app_name: Name of the application for default log file naming
    
    Returns:
        The configured root logger
    """
    global _logger_configured
    
    if _logger_configured:
        return logging.getLogger()
    
    # Reset any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure root logger
    root_logger.setLevel(log_level)
    formatter = logging.Formatter(log_format)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if specified or create a default one
    if log_file or log_dir:
        if log_file is None:
            # Create log directory if it doesn't exist
            Path(log_dir).mkdir(exist_ok=True)
            
            # Create log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"{app_name}_{timestamp}.log")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    _logger_configured = True
    
    # Log startup message
    root_logger.info(f"Logging configured. Level: {logging.getLevelName(log_level)}, File: {log_file if log_file else 'None'}")
    return root_logger

def get_logger(name):
    """
    Get a logger for a specific module.
    
    Args:
        name: Name of the module (typically __name__)
    
    Returns:
        A configured logger instance
    """
    # If logger hasn't been configured yet, configure with defaults
    if not _logger_configured:
        configure_logger()
    
    return logging.getLogger(name)
"""
Logging configuration for the RAG system.
Provides structured logging with file and console output.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Dict, Any
import structlog


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging configuration based on the provided config.
    
    Args:
        config: Logging configuration dictionary
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(config.get("logs_dir", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure standard logging
    log_level = getattr(logging, config.get("level", "INFO").upper())
    log_format = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Create formatters
    formatter = logging.Formatter(log_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    if config.get("enable_console", True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if config.get("enable_file", True):
        log_file = config.get("file_path", "./logs/rag_system.log")
        max_bytes = _parse_size(config.get("max_file_size", "10MB"))
        backup_count = config.get("backup_count", 5)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def _parse_size(size_str: str) -> int:
    """
    Parse size string like '10MB' to bytes.
    
    Args:
        size_str: Size string (e.g., '10MB', '1GB')
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)
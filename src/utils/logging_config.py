"""
Logging configuration for the RAG system.
Provides structured logging with file and console output.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Dict, Any, List
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


def setup_document_processing_logger(log_dir: str = "./logs") -> logging.Logger:
    """
    Set up a dedicated logger for document processing operations.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Configured logger for document processing
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create dedicated logger
    logger = logging.getLogger('document_processing')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler for document processing logs
    log_file = log_path / "document_processing.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10,
        encoding='utf-8'
    )
    
    # Detailed formatter for document processing
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def log_document_processing_event(
    logger: logging.Logger,
    event_type: str,
    document_id: str,
    details: Dict[str, Any],
    level: str = "INFO"
) -> None:
    """
    Log a document processing event with structured data.
    
    Args:
        logger: Logger instance
        event_type: Type of event (e.g., 'processing_start', 'processing_complete', 'error')
        document_id: ID of the document being processed
        details: Additional event details
        level: Log level
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    message = f"[{event_type}] Document: {document_id}"
    if details:
        detail_str = ", ".join([f"{k}={v}" for k, v in details.items()])
        message += f" | {detail_str}"
    
    logger.log(log_level, message)


def log_batch_processing_summary(
    logger: logging.Logger,
    total_documents: int,
    successful: int,
    failed: int,
    processing_time: float,
    errors: List[str]
) -> None:
    """
    Log a summary of batch processing results.
    
    Args:
        logger: Logger instance
        total_documents: Total number of documents processed
        successful: Number of successfully processed documents
        failed: Number of failed documents
        processing_time: Total processing time in seconds
        errors: List of error messages
    """
    success_rate = (successful / total_documents * 100) if total_documents > 0 else 0
    
    summary = (
        f"Batch Processing Summary: {successful}/{total_documents} documents processed "
        f"({success_rate:.1f}% success rate) in {processing_time:.2f}s"
    )
    
    if failed > 0:
        summary += f" | {failed} failures"
        logger.warning(summary)
        
        # Log individual errors
        for i, error in enumerate(errors[:10]):  # Limit to first 10 errors
            logger.error(f"Error {i+1}: {error}")
        
        if len(errors) > 10:
            logger.error(f"... and {len(errors) - 10} more errors")
    else:
        logger.info(summary)
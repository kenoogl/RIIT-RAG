"""
Custom exceptions for the RAG system.

This module defines specific exception classes for different types of errors
that can occur in the system, enabling better error handling and debugging.
"""


class RAGSystemError(Exception):
    """Base exception class for RAG system errors."""
    pass


class CrawlingError(RAGSystemError):
    """Exception raised when web crawling fails."""
    pass


class ExtractionError(RAGSystemError):
    """Exception raised when text extraction fails."""
    pass


class ProcessingError(RAGSystemError):
    """Exception raised when document processing fails."""
    pass


class EmbeddingError(RAGSystemError):
    """Exception raised when embedding generation fails."""
    pass


class IndexError(RAGSystemError):
    """Exception raised when index operations fail."""
    pass


class StorageError(RAGSystemError):
    """Exception raised when vector storage operations fail."""
    pass


class SearchError(RAGSystemError):
    """Exception raised when search operations fail."""
    pass


class DeletionError(RAGSystemError):
    """Exception raised when deletion operations fail."""
    pass


class GenerationError(RAGSystemError):
    """Exception raised when answer generation fails."""
    pass


class ModelLoadError(RAGSystemError):
    """Exception raised when model loading fails."""
    pass


class EncodingError(RAGSystemError):
    """Exception raised when text encoding fails."""
    pass


class ConfigurationError(RAGSystemError):
    """Exception raised when configuration operations fail."""
    pass
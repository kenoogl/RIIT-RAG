# Data models for the RAG system

from .core import Document, Chunk, SearchResult, Answer, Serializable
from .interfaces import (
    WebCrawlerInterface,
    DocumentProcessorInterface,
    VectorDatabaseInterface,
    SearchEngineInterface,
    GenerativeModelInterface,
    EmbeddingModelInterface,
    LoggingInterface,
    ConfigurationInterface
)
from .exceptions import (
    RAGSystemError,
    CrawlingError,
    ExtractionError,
    ProcessingError,
    EmbeddingError,
    IndexError,
    StorageError,
    SearchError,
    DeletionError,
    GenerationError,
    ModelLoadError,
    EncodingError,
    ConfigurationError
)

__all__ = [
    # Core data models
    'Document', 'Chunk', 'SearchResult', 'Answer', 'Serializable',
    # Interfaces
    'WebCrawlerInterface', 'DocumentProcessorInterface', 'VectorDatabaseInterface',
    'SearchEngineInterface', 'GenerativeModelInterface', 'EmbeddingModelInterface',
    'LoggingInterface', 'ConfigurationInterface',
    # Exceptions
    'RAGSystemError', 'CrawlingError', 'ExtractionError', 'ProcessingError',
    'EmbeddingError', 'IndexError', 'StorageError', 'SearchError', 'DeletionError',
    'GenerationError', 'ModelLoadError', 'EncodingError', 'ConfigurationError'
]
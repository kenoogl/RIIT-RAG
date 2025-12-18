"""
Abstract interfaces for RAG system components.

This module defines the abstract base classes that establish the contracts
for each major component in the system. These interfaces ensure loose coupling
and enable easy testing and extensibility.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from .core import Document, Chunk, SearchResult, Answer


class WebCrawlerInterface(ABC):
    """Abstract interface for web crawling functionality."""
    
    @abstractmethod
    def crawl_website(self, url: str) -> List[Document]:
        """
        Crawl a website and extract documents.
        
        Args:
            url: The URL to crawl
            
        Returns:
            List of extracted documents
            
        Raises:
            CrawlingError: If crawling fails
        """
        pass
    
    @abstractmethod
    def extract_text_from_html(self, html: str) -> str:
        """
        Extract text content from HTML.
        
        Args:
            html: Raw HTML content
            
        Returns:
            Extracted text content
            
        Raises:
            ExtractionError: If text extraction fails
        """
        pass
    
    @abstractmethod
    def filter_japanese_content(self, text: str) -> str:
        """
        Filter and extract Japanese content from text.
        
        Args:
            text: Input text content
            
        Returns:
            Filtered Japanese content
        """
        pass


class DocumentProcessorInterface(ABC):
    """Abstract interface for document processing functionality."""
    
    @abstractmethod
    def process_document(self, document: Document) -> List[Chunk]:
        """
        Process a document into chunks.
        
        Args:
            document: Document to process
            
        Returns:
            List of processed chunks
            
        Raises:
            ProcessingError: If document processing fails
        """
        pass
    
    @abstractmethod
    def generate_embeddings(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Generate embedding vectors for chunks.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass
    
    @abstractmethod
    def update_index(self, chunks: List[Chunk], vectors: List[List[float]]) -> bool:
        """
        Update the search index with new chunks and vectors.
        
        Args:
            chunks: List of chunks to index
            vectors: Corresponding embedding vectors
            
        Returns:
            True if update successful, False otherwise
            
        Raises:
            IndexError: If index update fails
        """
        pass


class VectorDatabaseInterface(ABC):
    """Abstract interface for vector database functionality."""
    
    @abstractmethod
    def store_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> bool:
        """
        Store vectors with associated metadata.
        
        Args:
            vectors: List of embedding vectors
            metadata: Associated metadata for each vector
            
        Returns:
            True if storage successful, False otherwise
            
        Raises:
            StorageError: If vector storage fails
        """
        pass
    
    @abstractmethod
    def search_similar(self, query_vector: List[float], top_k: int) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of search results ordered by similarity
            
        Raises:
            SearchError: If similarity search fails
        """
        pass
    
    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> bool:
        """
        Delete all vectors associated with a document.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if deletion successful, False otherwise
            
        Raises:
            DeletionError: If vector deletion fails
        """
        pass


class SearchEngineInterface(ABC):
    """Abstract interface for search engine functionality."""
    
    @abstractmethod
    def search(self, query: str) -> List[SearchResult]:
        """
        Search for relevant documents based on query.
        
        Args:
            query: Search query string
            
        Returns:
            List of search results
            
        Raises:
            SearchError: If search fails
        """
        pass
    
    @abstractmethod
    def rank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Rank search results by relevance.
        
        Args:
            results: List of search results to rank
            
        Returns:
            Ranked list of search results
        """
        pass


class GenerativeModelInterface(ABC):
    """Abstract interface for generative AI model functionality."""
    
    @abstractmethod
    def generate_answer(self, query: str, context: List[str]) -> Answer:
        """
        Generate an answer based on query and context.
        
        Args:
            query: User question
            context: List of relevant context strings
            
        Returns:
            Generated answer with metadata
            
        Raises:
            GenerationError: If answer generation fails
        """
        pass
    
    @abstractmethod
    def load_local_model(self, model_path: str) -> bool:
        """
        Load a local generative model.
        
        Args:
            model_path: Path to the model files
            
        Returns:
            True if model loaded successfully, False otherwise
            
        Raises:
            ModelLoadError: If model loading fails
        """
        pass


class EmbeddingModelInterface(ABC):
    """Abstract interface for embedding model functionality."""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode texts into embedding vectors.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of embedding vectors
            
        Raises:
            EncodingError: If encoding fails
        """
        pass
    
    @abstractmethod
    def load_model(self, model_name: str) -> bool:
        """
        Load an embedding model.
        
        Args:
            model_name: Name or path of the model
            
        Returns:
            True if model loaded successfully, False otherwise
            
        Raises:
            ModelLoadError: If model loading fails
        """
        pass


class LoggingInterface(ABC):
    """Abstract interface for logging functionality."""
    
    @abstractmethod
    def log_search(self, query: str, results: List[SearchResult], processing_time: float) -> None:
        """
        Log search operation details.
        
        Args:
            query: Search query
            results: Search results
            processing_time: Time taken for search
        """
        pass
    
    @abstractmethod
    def log_document_processing(self, document_id: str, status: str, error: Optional[str] = None) -> None:
        """
        Log document processing operation.
        
        Args:
            document_id: ID of processed document
            status: Processing status (success/error)
            error: Error message if processing failed
        """
        pass
    
    @abstractmethod
    def log_answer_generation(self, query: str, answer: Answer, chunks_used: List[str]) -> None:
        """
        Log answer generation details.
        
        Args:
            query: User query
            answer: Generated answer
            chunks_used: List of chunk IDs used for generation
        """
        pass


class ConfigurationInterface(ABC):
    """Abstract interface for system configuration."""
    
    @abstractmethod
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        pass
    
    @abstractmethod
    def set_config(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        pass
    
    @abstractmethod
    def load_config(self, config_path: str) -> bool:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if configuration loaded successfully, False otherwise
        """
        pass
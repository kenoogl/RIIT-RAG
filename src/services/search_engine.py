"""
Search engine implementation for the RAG system.

This module provides search functionality that combines vector similarity search
with ranking algorithms to find the most relevant documents for user queries.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import time

from ..models.interfaces import SearchEngineInterface, VectorDatabaseInterface, EmbeddingModelInterface
from ..models.core import SearchResult, Document, Chunk
from ..models.exceptions import SearchError
from .vector_database import LocalVectorDatabase


class VectorSearchEngine(SearchEngineInterface):
    """
    Vector-based search engine implementation.
    
    This class provides search functionality using embedding vectors and cosine similarity.
    It integrates with the vector database and embedding model to perform semantic search.
    """
    
    def __init__(
        self, 
        vector_db: VectorDatabaseInterface,
        embedding_model: Optional[EmbeddingModelInterface] = None,
        default_top_k: int = 10,
        min_similarity_threshold: float = 0.1
    ):
        """
        Initialize the vector search engine.
        
        Args:
            vector_db: Vector database instance
            embedding_model: Embedding model for query encoding
            default_top_k: Default number of results to return
            min_similarity_threshold: Minimum similarity score for results
        """
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.default_top_k = default_top_k
        self.min_similarity_threshold = min_similarity_threshold
        self.logger = logging.getLogger(__name__)
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Search for relevant documents based on query.
        
        Args:
            query: Search query string
            top_k: Number of top results to return (uses default if None)
            
        Returns:
            List of search results ordered by relevance
            
        Raises:
            SearchError: If search fails
        """
        try:
            start_time = time.time()
            
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            if top_k is None:
                top_k = self.default_top_k
            
            # Encode query to vector if embedding model is available
            if self.embedding_model is None:
                raise SearchError("Embedding model not available for query encoding")
            
            query_vector = self.embedding_model.encode([query.strip()])[0]
            
            # Search similar vectors
            results = self.vector_db.search_similar(query_vector, top_k * 2)  # Get more results for filtering
            
            # Filter by minimum similarity threshold
            filtered_results = [
                result for result in results 
                if result.score >= self.min_similarity_threshold
            ]
            
            # Rank and limit results
            ranked_results = self.rank_results(filtered_results)[:top_k]
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"Search completed: query='{query}', results={len(ranked_results)}, "
                f"time={processing_time:.3f}s"
            )
            
            return ranked_results
            
        except Exception as e:
            self.logger.error(f"Search failed for query '{query}': {e}")
            raise SearchError(f"Search failed: {e}")
    
    def rank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Rank search results by relevance using multiple factors.
        
        Args:
            results: List of search results to rank
            
        Returns:
            Ranked list of search results
        """
        if not results:
            return results
        
        try:
            # Enhanced ranking that considers multiple factors
            for result in results:
                # Base score is the similarity score
                base_score = result.score
                
                # Boost score based on content length (longer chunks might be more informative)
                content_length_factor = min(len(result.chunk.content) / 1000.0, 1.0)  # Normalize to [0, 1]
                
                # Boost score based on chunk position (earlier chunks might be more important)
                position_factor = 1.0 / (1.0 + result.chunk.position * 0.1)  # Diminishing returns
                
                # Calculate final score
                final_score = base_score * (1.0 + content_length_factor * 0.1 + position_factor * 0.05)
                
                # Update the result with the new score
                result.score = min(final_score, 1.0)  # Cap at 1.0
            
            # Sort by final score in descending order
            ranked_results = sorted(results, key=lambda x: x.score, reverse=True)
            
            self.logger.debug(f"Ranked {len(results)} search results")
            return ranked_results
            
        except Exception as e:
            self.logger.error(f"Failed to rank search results: {e}")
            # Return original results if ranking fails
            return sorted(results, key=lambda x: x.score, reverse=True)
    
    def search_with_filters(
        self, 
        query: str, 
        document_ids: Optional[List[str]] = None,
        language: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Search with additional filters.
        
        Args:
            query: Search query string
            document_ids: List of document IDs to search within (None for all)
            language: Language filter (None for all languages)
            top_k: Number of top results to return
            
        Returns:
            List of filtered and ranked search results
        """
        try:
            # Perform basic search
            results = self.search(query, top_k * 3 if top_k else self.default_top_k * 3)
            
            # Apply filters
            filtered_results = results
            
            if document_ids:
                filtered_results = [
                    result for result in filtered_results
                    if result.document.id in document_ids
                ]
            
            if language:
                filtered_results = [
                    result for result in filtered_results
                    if result.document.language == language
                ]
            
            # Re-rank and limit results
            final_results = self.rank_results(filtered_results)
            if top_k:
                final_results = final_results[:top_k]
            
            self.logger.info(
                f"Filtered search completed: query='{query}', "
                f"filters=(docs={document_ids}, lang={language}), "
                f"results={len(final_results)}"
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Filtered search failed: {e}")
            raise SearchError(f"Filtered search failed: {e}")
    
    def get_similar_chunks(self, chunk_id: str, top_k: int = 5) -> List[SearchResult]:
        """
        Find chunks similar to a given chunk.
        
        Args:
            chunk_id: ID of the reference chunk
            top_k: Number of similar chunks to return
            
        Returns:
            List of similar chunks
        """
        try:
            # This would require storing chunk embeddings and finding the reference chunk
            # For now, we'll implement a basic version that searches using the chunk content
            # In a full implementation, we'd store chunk vectors separately for this use case
            
            self.logger.warning(f"get_similar_chunks not fully implemented for chunk_id: {chunk_id}")
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to find similar chunks for {chunk_id}: {e}")
            raise SearchError(f"Failed to find similar chunks: {e}")
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        Get search engine statistics.
        
        Returns:
            Dictionary containing search statistics
        """
        try:
            db_stats = {}
            if hasattr(self.vector_db, 'get_stats'):
                db_stats = self.vector_db.get_stats()
            
            return {
                'vector_database': db_stats,
                'default_top_k': self.default_top_k,
                'min_similarity_threshold': self.min_similarity_threshold,
                'embedding_model_available': self.embedding_model is not None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get search stats: {e}")
            return {'error': str(e)}
    
    def set_embedding_model(self, embedding_model: EmbeddingModelInterface) -> None:
        """
        Set or update the embedding model.
        
        Args:
            embedding_model: New embedding model instance
        """
        self.embedding_model = embedding_model
        self.logger.info("Embedding model updated")
    
    def update_similarity_threshold(self, threshold: float) -> None:
        """
        Update the minimum similarity threshold.
        
        Args:
            threshold: New minimum similarity threshold (0.0 to 1.0)
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        
        self.min_similarity_threshold = threshold
        self.logger.info(f"Similarity threshold updated to {threshold}")


class HybridSearchEngine(VectorSearchEngine):
    """
    Hybrid search engine that combines vector search with keyword search.
    
    This extends the vector search engine to include traditional keyword-based
    search capabilities for better coverage of different query types.
    """
    
    def __init__(
        self, 
        vector_db: VectorDatabaseInterface,
        embedding_model: Optional[EmbeddingModelInterface] = None,
        default_top_k: int = 10,
        min_similarity_threshold: float = 0.1,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7
    ):
        """
        Initialize the hybrid search engine.
        
        Args:
            vector_db: Vector database instance
            embedding_model: Embedding model for query encoding
            default_top_k: Default number of results to return
            min_similarity_threshold: Minimum similarity score for results
            keyword_weight: Weight for keyword search scores
            vector_weight: Weight for vector search scores
        """
        super().__init__(vector_db, embedding_model, default_top_k, min_similarity_threshold)
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        
        # Ensure weights sum to 1.0
        total_weight = keyword_weight + vector_weight
        self.keyword_weight = keyword_weight / total_weight
        self.vector_weight = vector_weight / total_weight
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and keyword search.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of search results combining both search methods
        """
        try:
            if top_k is None:
                top_k = self.default_top_k
            
            # Perform vector search
            vector_results = super().search(query, top_k * 2)
            
            # Perform keyword search
            keyword_results = self._keyword_search(query, top_k * 2)
            
            # Combine and re-rank results
            combined_results = self._combine_results(vector_results, keyword_results)
            
            # Final ranking and limiting
            final_results = self.rank_results(combined_results)[:top_k]
            
            self.logger.info(
                f"Hybrid search completed: query='{query}', "
                f"vector_results={len(vector_results)}, "
                f"keyword_results={len(keyword_results)}, "
                f"final_results={len(final_results)}"
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            # Fallback to vector search only
            return super().search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int) -> List[SearchResult]:
        """
        Perform keyword-based search.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of keyword search results
        """
        try:
            # Simple keyword search implementation
            # In a full implementation, this would use a proper text search index
            
            # For now, we'll implement a basic TF-IDF-like scoring
            # This is a simplified version for demonstration
            
            query_terms = query.lower().split()
            if not query_terms:
                return []
            
            # This is a placeholder implementation
            # In practice, you'd want to implement proper keyword indexing
            self.logger.debug(f"Keyword search not fully implemented for query: {query}")
            return []
            
        except Exception as e:
            self.logger.error(f"Keyword search failed: {e}")
            return []
    
    def _combine_results(
        self, 
        vector_results: List[SearchResult], 
        keyword_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Combine vector and keyword search results.
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            
        Returns:
            Combined list of search results with adjusted scores
        """
        try:
            # Create a dictionary to merge results by chunk ID
            combined = {}
            
            # Add vector results
            for result in vector_results:
                chunk_id = result.chunk.id
                combined[chunk_id] = result
                # Adjust score with vector weight
                combined[chunk_id].score *= self.vector_weight
            
            # Add keyword results
            for result in keyword_results:
                chunk_id = result.chunk.id
                if chunk_id in combined:
                    # Combine scores
                    combined[chunk_id].score += result.score * self.keyword_weight
                else:
                    # New result from keyword search
                    result.score *= self.keyword_weight
                    combined[chunk_id] = result
            
            return list(combined.values())
            
        except Exception as e:
            self.logger.error(f"Failed to combine search results: {e}")
            # Return vector results as fallback
            return vector_results
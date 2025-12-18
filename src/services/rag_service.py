"""
RAG (Retrieval-Augmented Generation) service implementation.

This module provides the main RAG functionality that combines document retrieval
with answer generation to provide comprehensive responses to user queries.
"""

import logging
import time
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..models.interfaces import SearchEngineInterface, GenerativeModelInterface, LoggingInterface
from ..models.core import Answer, SearchResult, Document, Chunk
from ..models.exceptions import GenerationError, SearchError
from ..utils.config import get_config


logger = logging.getLogger(__name__)


class RAGService:
    """
    Main RAG service that combines retrieval and generation.
    
    This service orchestrates the retrieval of relevant documents and the generation
    of answers based on the retrieved context. It provides the core functionality
    for the question-answering system.
    """
    
    def __init__(
        self,
        search_engine: SearchEngineInterface,
        generative_model: GenerativeModelInterface,
        config_path: Optional[str] = None,
        logging_service: Optional[LoggingInterface] = None
    ):
        """
        Initialize the RAG service.
        
        Args:
            search_engine: Search engine for document retrieval
            generative_model: Generative model for answer generation
            config_path: Path to configuration file
            logging_service: Optional logging service for detailed tracking
        """
        self.search_engine = search_engine
        self.generative_model = generative_model
        self.logging_service = logging_service
        self.config = get_config(config_path)
        
        # Load configuration
        self._load_config()
        
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> None:
        """Load RAG-specific configuration."""
        search_config = self.config.get("search", {})
        
        self.max_context_chunks = search_config.get("max_results", 5)
        self.min_similarity_score = search_config.get("min_similarity_score", 0.5)
        self.enable_reranking = search_config.get("enable_reranking", True)
        self.max_context_length = search_config.get("max_context_length", 2000)
        
        # Generation-specific settings
        generation_config = self.config.get("generation", {})
        self.include_sources = generation_config.get("include_sources", True)
        self.source_citation_format = generation_config.get("source_citation_format", "numbered")
    
    def generate_answer(
        self, 
        query: str, 
        max_results: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> Answer:
        """
        Generate an answer for the given query using RAG.
        
        Args:
            query: User question
            max_results: Maximum number of context chunks to use
            min_score: Minimum similarity score for context chunks
            
        Returns:
            Generated answer with metadata
            
        Raises:
            GenerationError: If answer generation fails
            SearchError: If document retrieval fails
        """
        try:
            start_time = time.time()
            
            self.logger.info(f"Processing RAG query: {query[:100]}...")
            
            # Step 1: Retrieve relevant documents
            search_results = self._retrieve_context(query, max_results, min_score)
            
            if not search_results:
                return self._generate_no_context_answer(query, time.time() - start_time)
            
            # Step 2: Prepare context for generation
            context_strings = self._prepare_context(search_results)
            
            # Step 3: Generate answer using the generative model
            answer = self.generative_model.generate_answer(query, context_strings)
            
            # Step 4: Enhance answer with RAG-specific metadata
            enhanced_answer = self._enhance_answer(answer, search_results, query)
            
            # Step 5: Log the interaction
            self._log_rag_interaction(query, enhanced_answer, search_results)
            
            total_time = time.time() - start_time
            enhanced_answer.processing_time = total_time
            
            self.logger.info(f"RAG answer generated successfully in {total_time:.2f}s")
            
            return enhanced_answer
            
        except SearchError as e:
            error_msg = f"Failed to retrieve context for query: {str(e)}"
            self.logger.error(error_msg)
            raise GenerationError(error_msg) from e
        
        except Exception as e:
            error_msg = f"RAG answer generation failed: {str(e)}"
            self.logger.error(error_msg)
            raise GenerationError(error_msg) from e
    
    def _retrieve_context(
        self, 
        query: str, 
        max_results: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Retrieve relevant context for the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            min_score: Minimum similarity score
            
        Returns:
            List of relevant search results
        """
        if max_results is None:
            max_results = self.max_context_chunks
        
        if min_score is None:
            min_score = self.min_similarity_score
        
        # Perform search
        search_results = self.search_engine.search(query, top_k=max_results * 2)
        
        # Filter by minimum score
        filtered_results = [
            result for result in search_results
            if result.score >= min_score
        ]
        
        # Limit to max results
        return filtered_results[:max_results]
    
    def _prepare_context(self, search_results: List[SearchResult]) -> List[str]:
        """
        Prepare context strings from search results.
        
        Args:
            search_results: List of search results
            
        Returns:
            List of context strings
        """
        context_strings = []
        total_length = 0
        
        for i, result in enumerate(search_results):
            # Create context string with metadata
            context = self._format_context_chunk(result, i + 1)
            
            # Check if adding this context would exceed the limit
            if total_length + len(context) > self.max_context_length:
                break
            
            context_strings.append(context)
            total_length += len(context)
        
        return context_strings
    
    def _format_context_chunk(self, result: SearchResult, index: int) -> str:
        """
        Format a search result into a context string.
        
        Args:
            result: Search result to format
            index: Index number for the context
            
        Returns:
            Formatted context string
        """
        # Basic context formatting
        context = f"[文書{index}] {result.chunk.content}"
        
        # Add document title if available
        if hasattr(result, 'document') and result.document and result.document.title:
            context = f"[文書{index}: {result.document.title}] {result.chunk.content}"
        
        # Add URL if available
        if hasattr(result, 'document') and result.document and result.document.url:
            context += f"\n出典: {result.document.url}"
        
        return context
    
    def _enhance_answer(
        self, 
        answer: Answer, 
        search_results: List[SearchResult], 
        query: str
    ) -> Answer:
        """
        Enhance the generated answer with RAG-specific metadata.
        
        Args:
            answer: Original generated answer
            search_results: Search results used for context
            query: Original query
            
        Returns:
            Enhanced answer with additional metadata
        """
        # Create enhanced sources list
        enhanced_sources = self._create_enhanced_sources(search_results)
        
        # Calculate enhanced confidence based on search results
        enhanced_confidence = self._calculate_enhanced_confidence(
            answer, search_results, query
        )
        
        # Create enhanced answer
        enhanced_answer = Answer(
            text=answer.text,
            sources=enhanced_sources,
            confidence=enhanced_confidence,
            processing_time=answer.processing_time
        )
        
        return enhanced_answer
    
    def _create_enhanced_sources(self, search_results: List[SearchResult]) -> List[str]:
        """
        Create enhanced source information from search results.
        
        Args:
            search_results: Search results to extract sources from
            
        Returns:
            List of enhanced source strings
        """
        sources = []
        
        for i, result in enumerate(search_results):
            if self.source_citation_format == "numbered":
                source = f"[{i+1}]"
            else:
                source = f"文書{i+1}"
            
            # Add document title if available
            if hasattr(result, 'document') and result.document:
                if result.document.title:
                    source += f" {result.document.title}"
                
                # Add URL if available
                if result.document.url:
                    source += f" ({result.document.url})"
            
            # Add similarity score
            source += f" (関連度: {result.score:.2f})"
            
            sources.append(source)
        
        return sources
    
    def _calculate_enhanced_confidence(
        self, 
        answer: Answer, 
        search_results: List[SearchResult], 
        query: str
    ) -> float:
        """
        Calculate enhanced confidence score based on search results.
        
        Args:
            answer: Generated answer
            search_results: Search results used
            query: Original query
            
        Returns:
            Enhanced confidence score
        """
        base_confidence = answer.confidence
        
        # Factor in search result quality
        if search_results:
            avg_similarity = sum(result.score for result in search_results) / len(search_results)
            max_similarity = max(result.score for result in search_results)
            
            # Boost confidence based on search quality
            search_quality_boost = (avg_similarity + max_similarity) / 2 * 0.2
            
            # Boost confidence based on number of relevant results
            result_count_boost = min(len(search_results) / self.max_context_chunks, 1.0) * 0.1
            
            enhanced_confidence = base_confidence + search_quality_boost + result_count_boost
        else:
            # Reduce confidence if no relevant context found
            enhanced_confidence = base_confidence * 0.5
        
        # Ensure confidence is within bounds
        return max(0.0, min(1.0, enhanced_confidence))
    
    def _generate_no_context_answer(self, query: str, processing_time: float) -> Answer:
        """
        Generate an answer when no relevant context is found.
        
        Args:
            query: Original query
            processing_time: Time spent on processing
            
        Returns:
            Answer indicating no relevant information was found
        """
        no_context_text = (
            "申し訳ございませんが、お尋ねの内容に関する適切な情報を見つけることができませんでした。"
            "より具体的な質問をしていただくか、別の表現で質問を再度お試しください。"
        )
        
        return Answer(
            text=no_context_text,
            sources=[],
            confidence=0.1,
            processing_time=processing_time
        )
    
    def _log_rag_interaction(
        self, 
        query: str, 
        answer: Answer, 
        search_results: List[SearchResult]
    ) -> None:
        """
        Log the RAG interaction for monitoring and analysis.
        
        Args:
            query: User query
            answer: Generated answer
            search_results: Search results used
        """
        if self.logging_service:
            # Use dedicated logging service if available
            chunk_ids = [result.chunk.id for result in search_results]
            self.logging_service.log_answer_generation(query, answer, chunk_ids)
        else:
            # Basic logging
            self.logger.info(
                f"RAG interaction: query_length={len(query)}, "
                f"context_chunks={len(search_results)}, "
                f"answer_length={len(answer.text)}, "
                f"confidence={answer.confidence:.3f}, "
                f"processing_time={answer.processing_time:.3f}s"
            )
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get the current status of the RAG service.
        
        Returns:
            Dictionary containing service status information
        """
        return {
            "search_engine_available": self.search_engine is not None,
            "generative_model_available": self.generative_model is not None,
            "max_context_chunks": self.max_context_chunks,
            "min_similarity_score": self.min_similarity_score,
            "enable_reranking": self.enable_reranking,
            "max_context_length": self.max_context_length,
            "include_sources": self.include_sources,
            "source_citation_format": self.source_citation_format
        }


def create_rag_service(
    search_engine: SearchEngineInterface,
    generative_model: GenerativeModelInterface,
    config_path: Optional[str] = None,
    logging_service: Optional[LoggingInterface] = None
) -> RAGService:
    """
    Factory function to create a RAG service.
    
    Args:
        search_engine: Search engine instance
        generative_model: Generative model instance
        config_path: Path to configuration file
        logging_service: Optional logging service
        
    Returns:
        Configured RAG service instance
    """
    return RAGService(search_engine, generative_model, config_path, logging_service)
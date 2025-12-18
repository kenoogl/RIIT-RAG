"""
Enhanced search engine with advanced query processing and logging.

This module extends the basic search engine with sophisticated query processing,
Japanese language support, and comprehensive logging for the RAG system.
"""

import logging
import time
import re
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict

from ..models.interfaces import SearchEngineInterface, VectorDatabaseInterface, EmbeddingModelInterface, LoggingInterface
from ..models.core import SearchResult, Document, Chunk
from ..models.exceptions import SearchError
from ..utils.config import get_config
from .search_engine import VectorSearchEngine


class EnhancedSearchEngine(VectorSearchEngine):
    """
    Enhanced search engine with advanced query processing and logging.
    
    This class extends the basic vector search engine with:
    - Advanced Japanese query processing
    - Query expansion and normalization
    - Comprehensive search logging and metrics
    - Search result recording for analysis
    """
    
    def __init__(
        self,
        vector_db: VectorDatabaseInterface,
        embedding_model: Optional[EmbeddingModelInterface] = None,
        logging_service: Optional[LoggingInterface] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the enhanced search engine.
        
        Args:
            vector_db: Vector database instance
            embedding_model: Embedding model for query encoding
            logging_service: Optional logging service for detailed tracking
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = get_config(config_path)
        search_config = self.config.get("search", {})
        
        # Initialize parent class
        super().__init__(
            vector_db=vector_db,
            embedding_model=embedding_model,
            default_top_k=search_config.get("max_results", 10),
            min_similarity_threshold=search_config.get("min_similarity_score", 0.1)
        )
        
        self.logging_service = logging_service
        
        # Enhanced search configuration
        self.enable_query_expansion = search_config.get("enable_query_expansion", True)
        self.enable_query_normalization = search_config.get("enable_query_normalization", True)
        self.max_query_length = search_config.get("max_query_length", 500)
        self.enable_search_analytics = search_config.get("enable_search_analytics", True)
        
        # Search analytics storage
        self.search_history = []
        self.search_metrics = defaultdict(int)
        
        self.logger = logging.getLogger(__name__)
        
        # Japanese text processing patterns
        self._init_japanese_patterns()
    
    def _init_japanese_patterns(self) -> None:
        """Initialize Japanese text processing patterns."""
        # Common Japanese query patterns for supercomputer domain
        self.supercomputer_synonyms = {
            "スパコン": ["スーパーコンピュータ", "スーパーコンピューター", "HPC"],
            "ジョブ": ["バッチジョブ", "計算ジョブ", "タスク"],
            "アカウント": ["ユーザー", "利用者", "ログイン"],
            "ストレージ": ["保存", "ファイル", "データ"],
            "キュー": ["待ち行列", "スケジューラ", "SLURM"],
            "ノード": ["計算ノード", "サーバー", "マシン"]
        }
        
        # Japanese text normalization patterns
        self.normalization_patterns = [
            (r'[０-９]', lambda m: chr(ord(m.group()) - ord('０') + ord('0'))),  # Full-width to half-width numbers
            (r'[Ａ-Ｚａ-ｚ]', lambda m: chr(ord(m.group()) - ord('Ａ') + ord('A')) if m.group() <= 'Ｚ' else chr(ord(m.group()) - ord('ａ') + ord('a'))),  # Full-width to half-width letters
            (r'\s+', ' '),  # Multiple spaces to single space
        ]
        
        # Question word patterns for Japanese
        self.question_patterns = [
            r'.*どう.*',  # How
            r'.*なに.*', r'.*何.*',  # What
            r'.*いつ.*',  # When
            r'.*どこ.*',  # Where
            r'.*だれ.*', r'.*誰.*',  # Who
            r'.*なぜ.*', r'.*どうして.*',  # Why
            r'.*方法.*', r'.*やり方.*',  # Method/way
            r'.*手順.*', r'.*ステップ.*',  # Steps/procedure
        ]
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Enhanced search with advanced query processing and logging.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of search results with enhanced processing
        """
        try:
            start_time = time.time()
            
            # Record search attempt
            self.search_metrics["total_searches"] += 1
            
            # Process and validate query
            processed_query = self._process_query(query)
            if not processed_query:
                self.search_metrics["empty_queries"] += 1
                return []
            
            # Perform enhanced search
            results = self._enhanced_search(processed_query, top_k)
            
            # Record search completion
            processing_time = time.time() - start_time
            self._record_search_result(query, processed_query, results, processing_time)
            
            self.logger.info(
                f"Enhanced search completed: original='{query[:50]}...', "
                f"processed='{processed_query[:50]}...', "
                f"results={len(results)}, time={processing_time:.3f}s"
            )
            
            return results
            
        except Exception as e:
            self.search_metrics["failed_searches"] += 1
            self.logger.error(f"Enhanced search failed for query '{query}': {e}")
            raise SearchError(f"Enhanced search failed: {e}")
    
    def _process_query(self, query: str) -> str:
        """
        Process and enhance the search query.
        
        Args:
            query: Raw search query
            
        Returns:
            Processed and enhanced query
        """
        if not query or not query.strip():
            return ""
        
        processed = query.strip()
        
        # Limit query length
        if len(processed) > self.max_query_length:
            processed = processed[:self.max_query_length]
            self.logger.warning(f"Query truncated to {self.max_query_length} characters")
        
        # Normalize Japanese text
        if self.enable_query_normalization:
            processed = self._normalize_japanese_text(processed)
        
        # Expand query with synonyms
        if self.enable_query_expansion:
            processed = self._expand_query(processed)
        
        return processed
    
    def _normalize_japanese_text(self, text: str) -> str:
        """
        Normalize Japanese text for better search.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        normalized = text
        
        # Apply normalization patterns
        for pattern, replacement in self.normalization_patterns:
            if callable(replacement):
                normalized = re.sub(pattern, replacement, normalized)
            else:
                normalized = re.sub(pattern, replacement, normalized)
        
        return normalized.strip()
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with domain-specific synonyms.
        
        Args:
            query: Input query
            
        Returns:
            Expanded query
        """
        expanded_terms = []
        query_lower = query.lower()
        
        # Add original query
        expanded_terms.append(query)
        
        # Add synonyms for supercomputer domain terms
        for term, synonyms in self.supercomputer_synonyms.items():
            if term in query:
                # Add the most relevant synonym
                if synonyms:
                    expanded_terms.append(query.replace(term, synonyms[0]))
                    break  # Only add one expansion to avoid over-expansion
        
        # Join expanded terms
        return " ".join(expanded_terms)
    
    def _enhanced_search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Perform enhanced search with multiple strategies.
        
        Args:
            query: Processed query
            top_k: Number of results to return
            
        Returns:
            Enhanced search results
        """
        if top_k is None:
            top_k = self.default_top_k
        
        # Perform base vector search
        results = super().search(query, top_k * 2)  # Get more results for re-ranking
        
        # Apply enhanced ranking
        enhanced_results = self._enhanced_ranking(query, results)
        
        # Apply query-specific filtering
        filtered_results = self._apply_query_filters(query, enhanced_results)
        
        return filtered_results[:top_k]
    
    def _enhanced_ranking(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Apply enhanced ranking based on query characteristics.
        
        Args:
            query: Search query
            results: Initial search results
            
        Returns:
            Re-ranked search results
        """
        if not results:
            return results
        
        # Detect query type
        query_type = self._detect_query_type(query)
        
        # Apply query-type specific ranking
        for result in results:
            base_score = result.score
            
            # Boost based on query type
            if query_type == "how_to" and any(word in result.chunk.content.lower() for word in ["方法", "手順", "やり方"]):
                result.score *= 1.2
            elif query_type == "what_is" and any(word in result.chunk.content.lower() for word in ["とは", "について", "説明"]):
                result.score *= 1.2
            elif query_type == "troubleshooting" and any(word in result.chunk.content.lower() for word in ["エラー", "問題", "解決"]):
                result.score *= 1.2
            
            # Boost for document title relevance
            if hasattr(result, 'document') and result.document and result.document.title:
                title_lower = result.document.title.lower()
                query_lower = query.lower()
                if any(word in title_lower for word in query_lower.split()):
                    result.score *= 1.1
            
            # Boost for chunk position (earlier chunks might be more important)
            position_boost = 1.0 / (1.0 + result.chunk.position * 0.05)
            result.score *= position_boost
            
            # Ensure score doesn't exceed 1.0
            result.score = min(result.score, 1.0)
        
        # Sort by enhanced score
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def _detect_query_type(self, query: str) -> str:
        """
        Detect the type of query for specialized processing.
        
        Args:
            query: Search query
            
        Returns:
            Query type string
        """
        query_lower = query.lower()
        
        # Troubleshooting questions (check first as they're specific)
        if any(word in query_lower for word in ["エラー", "問題", "うまくいかない", "できない"]):
            return "troubleshooting"
        
        # Account/access questions (check before how-to as "申請方法" contains "方法")
        if any(word in query_lower for word in ["アカウント", "ログイン", "申請", "登録"]):
            # But if it also contains how-to words, it's still account-related
            if not any(pattern in query_lower for pattern in ["どう", "やり方", "教えて"]):
                return "account"
        
        # Job/batch questions (check before how-to)
        if any(word in query_lower for word in ["ジョブ", "バッチ", "投入"]):
            if not any(pattern in query_lower for pattern in ["どう", "方法", "やり方", "教えて"]):
                return "job_batch"
        
        # How-to questions
        if any(pattern in query_lower for pattern in ["どう", "方法", "やり方", "教えて", "使い方"]):
            return "how_to"
        
        # What-is questions
        if any(pattern in query_lower for pattern in ["とは", "なに", "何", "何ですか"]):
            return "what_is"
        
        return "general"
    
    def _apply_query_filters(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Apply query-specific filters to results.
        
        Args:
            query: Search query
            results: Search results to filter
            
        Returns:
            Filtered search results
        """
        if not results:
            return results
        
        # Filter by minimum similarity threshold
        filtered = [r for r in results if r.score >= self.min_similarity_threshold]
        
        # Apply language filtering (Japanese content only)
        japanese_filtered = []
        for result in filtered:
            if hasattr(result, 'document') and result.document:
                if result.document.language == 'ja':
                    japanese_filtered.append(result)
            else:
                # If no language info, include by default
                japanese_filtered.append(result)
        
        return japanese_filtered
    
    def _record_search_result(
        self, 
        original_query: str, 
        processed_query: str, 
        results: List[SearchResult], 
        processing_time: float
    ) -> None:
        """
        Record search results for analytics and logging.
        
        Args:
            original_query: Original user query
            processed_query: Processed query
            results: Search results
            processing_time: Time taken for search
        """
        # Record in search history
        if self.enable_search_analytics:
            search_record = {
                "timestamp": datetime.now(),
                "original_query": original_query,
                "processed_query": processed_query,
                "num_results": len(results),
                "processing_time": processing_time,
                "top_score": results[0].score if results else 0.0,
                "query_type": self._detect_query_type(original_query)
            }
            
            self.search_history.append(search_record)
            
            # Keep only recent history (last 1000 searches)
            if len(self.search_history) > 1000:
                self.search_history = self.search_history[-1000:]
        
        # Update metrics
        self.search_metrics["successful_searches"] += 1
        if results:
            self.search_metrics["searches_with_results"] += 1
        else:
            self.search_metrics["searches_without_results"] += 1
        
        # Use logging service if available
        if self.logging_service:
            self.logging_service.log_search(original_query, results, processing_time)
        
        # Log search details
        self.logger.debug(
            f"Search recorded: query='{original_query}', "
            f"results={len(results)}, time={processing_time:.3f}s"
        )
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """
        Get search analytics and metrics.
        
        Returns:
            Dictionary containing search analytics
        """
        if not self.enable_search_analytics:
            return {"analytics_disabled": True}
        
        # Calculate analytics
        total_searches = self.search_metrics["total_searches"]
        if total_searches == 0:
            return {"no_searches": True}
        
        # Query type distribution and popular terms
        query_types = defaultdict(int)
        popular_terms = defaultdict(int)
        avg_processing_time = 0.0
        avg_results_count = 0.0
        
        if self.search_history:
            for record in self.search_history:
                query_types[record["query_type"]] += 1
                avg_processing_time += record["processing_time"]
                avg_results_count += record["num_results"]
                
                # Extract terms from query for popular terms tracking
                query = record.get("query", "")
                # Split on common Japanese and English delimiters
                import re
                words = re.findall(r'[\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+', query)
                for word in words:
                    if len(word) > 1:  # Only count meaningful words
                        popular_terms[word] += 1
            
            avg_processing_time /= len(self.search_history)
            avg_results_count /= len(self.search_history)
        
        # Sort popular terms by frequency
        sorted_popular_terms = dict(sorted(
            popular_terms.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])
        
        return {
            "total_searches": total_searches,
            "successful_searches": self.search_metrics["successful_searches"],
            "failed_searches": self.search_metrics["failed_searches"],
            "searches_with_results": self.search_metrics["searches_with_results"],
            "searches_without_results": self.search_metrics["searches_without_results"],
            "success_rate": self.search_metrics["successful_searches"] / total_searches,
            "result_rate": self.search_metrics["searches_with_results"] / total_searches,
            "avg_processing_time": avg_processing_time,
            "avg_results_count": avg_results_count,
            "query_type_distribution": dict(query_types),
            "popular_terms": sorted_popular_terms,
            "recent_searches": len(self.search_history)
        }
    
    def get_recent_searches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent search history.
        
        Args:
            limit: Maximum number of recent searches to return
            
        Returns:
            List of recent search records
        """
        if not self.enable_search_analytics:
            return []
        
        return self.search_history[-limit:] if self.search_history else []
    
    def clear_search_history(self) -> None:
        """Clear search history and reset metrics."""
        self.search_history.clear()
        self.search_metrics.clear()
        self.logger.info("Search history and metrics cleared")


def create_enhanced_search_engine(
    vector_db: VectorDatabaseInterface,
    embedding_model: Optional[EmbeddingModelInterface] = None,
    logging_service: Optional[LoggingInterface] = None,
    config_path: Optional[str] = None
) -> EnhancedSearchEngine:
    """
    Factory function to create an enhanced search engine.
    
    Args:
        vector_db: Vector database instance
        embedding_model: Embedding model instance
        logging_service: Optional logging service
        config_path: Path to configuration file
        
    Returns:
        Configured enhanced search engine instance
    """
    return EnhancedSearchEngine(
        vector_db=vector_db,
        embedding_model=embedding_model,
        logging_service=logging_service,
        config_path=config_path
    )
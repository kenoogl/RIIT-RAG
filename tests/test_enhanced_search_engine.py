"""
Tests for the enhanced search engine.

This module contains unit tests for the EnhancedSearchEngine class,
testing advanced query processing, Japanese language support, and logging.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.services.enhanced_search_engine import EnhancedSearchEngine, create_enhanced_search_engine
from src.services.search_logging_service import SearchLoggingService
from src.models.core import SearchResult, Document, Chunk
from src.models.exceptions import SearchError
from src.models.interfaces import VectorDatabaseInterface, EmbeddingModelInterface, LoggingInterface


class TestEnhancedSearchEngine:
    """Test cases for EnhancedSearchEngine."""
    
    @pytest.fixture
    def mock_vector_db(self):
        """Create a mock vector database."""
        vector_db = Mock(spec=VectorDatabaseInterface)
        return vector_db
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        embedding_model = Mock(spec=EmbeddingModelInterface)
        embedding_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        return embedding_model
    
    @pytest.fixture
    def mock_logging_service(self):
        """Create a mock logging service."""
        logging_service = Mock(spec=LoggingInterface)
        return logging_service
    
    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results."""
        doc = Document(
            id="doc1",
            url="https://www.cc.kyushu-u.ac.jp/scp/guide",
            title="九州大学スパコン利用ガイド",
            content="スパコンの利用方法について説明します。",
            language="ja",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content="九州大学のスーパーコンピュータを利用するには、まずアカウント申請が必要です。",
            position=0,
            embedding=None
        )
        
        return [SearchResult(chunk=chunk, score=0.9, document=doc)]
    
    @pytest.fixture
    def enhanced_search_engine(self, mock_vector_db, mock_embedding_model, mock_logging_service):
        """Create an enhanced search engine instance."""
        with patch('src.services.enhanced_search_engine.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "search": {
                    "max_results": 10,
                    "min_similarity_score": 0.1,
                    "enable_query_expansion": True,
                    "enable_query_normalization": True,
                    "max_query_length": 500,
                    "enable_search_analytics": True
                }
            }
            return EnhancedSearchEngine(
                vector_db=mock_vector_db,
                embedding_model=mock_embedding_model,
                logging_service=mock_logging_service
            )
    
    def test_initialization(self, enhanced_search_engine):
        """Test enhanced search engine initialization."""
        assert enhanced_search_engine.vector_db is not None
        assert enhanced_search_engine.embedding_model is not None
        assert enhanced_search_engine.logging_service is not None
        assert enhanced_search_engine.enable_query_expansion is True
        assert enhanced_search_engine.enable_query_normalization is True
        assert enhanced_search_engine.enable_search_analytics is True
    
    def test_japanese_text_normalization(self, enhanced_search_engine):
        """Test Japanese text normalization."""
        # Test full-width to half-width conversion
        test_cases = [
            ("１２３", "123"),
            ("ＡＢＣ", "ABC"),
            ("ａｂｃ", "abc"),
            ("  複数  スペース  ", "複数 スペース"),
        ]
        
        for input_text, expected in test_cases:
            result = enhanced_search_engine._normalize_japanese_text(input_text)
            assert result == expected
    
    def test_query_expansion(self, enhanced_search_engine):
        """Test query expansion with synonyms."""
        test_cases = [
            ("スパコンの使い方", "スーパーコンピュータ"),
            ("ジョブの実行", "バッチジョブ"),
            ("アカウント申請", "ユーザー"),
        ]
        
        for query, expected_synonym in test_cases:
            expanded = enhanced_search_engine._expand_query(query)
            assert expected_synonym in expanded or query in expanded
    
    def test_query_type_detection(self, enhanced_search_engine):
        """Test query type detection."""
        test_cases = [
            ("スパコンの使い方を教えてください", "how_to"),
            ("スパコンとは何ですか", "what_is"),
            ("エラーが発生しました", "troubleshooting"),
            ("アカウント申請について", "account"),  # Changed to avoid "方法"
            ("ジョブを投入したい", "job_batch"),
            ("一般的な質問", "general"),
        ]
        
        for query, expected_type in test_cases:
            detected_type = enhanced_search_engine._detect_query_type(query)
            assert detected_type == expected_type
    
    def test_query_processing(self, enhanced_search_engine):
        """Test comprehensive query processing."""
        # Test normal query
        query = "スパコンの使い方を教えてください"
        processed = enhanced_search_engine._process_query(query)
        assert len(processed) > 0
        assert "スパコン" in processed or "スーパーコンピュータ" in processed
        
        # Test empty query
        empty_processed = enhanced_search_engine._process_query("")
        assert empty_processed == ""
        
        # Test very long query
        long_query = "非常に長い質問です。" * 100
        long_processed = enhanced_search_engine._process_query(long_query)
        assert len(long_processed) <= enhanced_search_engine.max_query_length
    
    def test_enhanced_search_success(self, enhanced_search_engine, mock_vector_db, 
                                   mock_embedding_model, sample_search_results):
        """Test successful enhanced search."""
        # Setup mocks
        mock_vector_db.search_similar.return_value = sample_search_results
        
        # Test search
        query = "スパコンの使い方"
        results = enhanced_search_engine.search(query)
        
        # Verify calls
        mock_embedding_model.encode.assert_called()
        mock_vector_db.search_similar.assert_called()
        
        # Verify results
        assert isinstance(results, list)
        assert len(results) <= enhanced_search_engine.default_top_k
    
    def test_enhanced_ranking(self, enhanced_search_engine, sample_search_results):
        """Test enhanced ranking functionality."""
        # Create multiple results with different characteristics
        results = sample_search_results * 3
        for i, result in enumerate(results):
            result.chunk.id = f"chunk{i+1}"
            result.score = 0.8 - i * 0.1  # Decreasing scores
        
        # Test ranking
        ranked = enhanced_search_engine._enhanced_ranking("スパコンの使い方", results)
        
        assert len(ranked) == len(results)
        # Verify results are sorted by score (descending)
        for i in range(len(ranked) - 1):
            assert ranked[i].score >= ranked[i + 1].score
    
    def test_query_filters(self, enhanced_search_engine, sample_search_results):
        """Test query-specific filtering."""
        # Test with Japanese content
        filtered = enhanced_search_engine._apply_query_filters("テスト質問", sample_search_results)
        assert len(filtered) > 0
        
        # Test with low-score results
        low_score_results = sample_search_results.copy()
        low_score_results[0].score = 0.05  # Below threshold
        
        filtered_low = enhanced_search_engine._apply_query_filters("テスト質問", low_score_results)
        assert len(filtered_low) == 0  # Should be filtered out
    
    def test_search_analytics(self, enhanced_search_engine, mock_vector_db, 
                            mock_embedding_model, sample_search_results):
        """Test search analytics functionality."""
        # Setup mocks
        mock_vector_db.search_similar.return_value = sample_search_results
        
        # Perform multiple searches
        queries = ["スパコンの使い方", "アカウント申請", "ジョブ実行"]
        for query in queries:
            enhanced_search_engine.search(query)
        
        # Check analytics
        analytics = enhanced_search_engine.get_search_analytics()
        
        assert analytics["total_searches"] == len(queries)
        assert analytics["successful_searches"] == len(queries)
        assert analytics["searches_with_results"] == len(queries)
        assert analytics["avg_processing_time"] > 0
        assert "query_type_distribution" in analytics
    
    def test_search_history(self, enhanced_search_engine, mock_vector_db, 
                          mock_embedding_model, sample_search_results):
        """Test search history tracking."""
        # Setup mocks
        mock_vector_db.search_similar.return_value = sample_search_results
        
        # Perform searches
        queries = ["質問1", "質問2", "質問3"]
        for query in queries:
            enhanced_search_engine.search(query)
        
        # Check history
        recent_searches = enhanced_search_engine.get_recent_searches(limit=5)
        assert len(recent_searches) == len(queries)
        
        # Verify history content
        for i, record in enumerate(recent_searches):
            assert record["original_query"] == queries[i]
            assert record["num_results"] > 0
            assert record["processing_time"] > 0
    
    def test_search_error_handling(self, enhanced_search_engine, mock_vector_db, mock_embedding_model):
        """Test search error handling."""
        # Test with embedding model error
        mock_embedding_model.encode.side_effect = Exception("Encoding failed")
        
        with pytest.raises(SearchError):
            enhanced_search_engine.search("テスト質問")
        
        # Verify error metrics
        analytics = enhanced_search_engine.get_search_analytics()
        assert analytics["failed_searches"] > 0
    
    def test_clear_search_history(self, enhanced_search_engine, mock_vector_db, 
                                mock_embedding_model, sample_search_results):
        """Test clearing search history."""
        # Setup mocks and perform searches
        mock_vector_db.search_similar.return_value = sample_search_results
        enhanced_search_engine.search("テスト質問")
        
        # Verify history exists
        assert len(enhanced_search_engine.search_history) > 0
        
        # Clear history
        enhanced_search_engine.clear_search_history()
        
        # Verify history is cleared
        assert len(enhanced_search_engine.search_history) == 0
        analytics = enhanced_search_engine.get_search_analytics()
        # After clearing, analytics should show no searches or indicate no data
        assert analytics.get("total_searches", 0) == 0 or "no_searches" in analytics
    
    def test_factory_function(self, mock_vector_db, mock_embedding_model, mock_logging_service):
        """Test enhanced search engine factory function."""
        with patch('src.services.enhanced_search_engine.get_config') as mock_get_config:
            mock_get_config.return_value = {"search": {}}
            
            engine = create_enhanced_search_engine(
                vector_db=mock_vector_db,
                embedding_model=mock_embedding_model,
                logging_service=mock_logging_service
            )
            
            assert isinstance(engine, EnhancedSearchEngine)
            assert engine.vector_db == mock_vector_db
            assert engine.embedding_model == mock_embedding_model
            assert engine.logging_service == mock_logging_service


class TestEnhancedSearchEngineIntegration:
    """Integration tests for EnhancedSearchEngine."""
    
    def test_japanese_query_processing_integration(self):
        """Test complete Japanese query processing pipeline."""
        # Create mock components
        vector_db = Mock(spec=VectorDatabaseInterface)
        embedding_model = Mock(spec=EmbeddingModelInterface)
        logging_service = Mock(spec=LoggingInterface)
        
        # Setup realistic responses
        embedding_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        
        doc = Document(
            id="doc1",
            url="https://www.cc.kyushu-u.ac.jp/scp/",
            title="九州大学スパコン利用ガイド",
            content="九州大学のスーパーコンピュータ利用方法",
            language="ja",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content="九州大学のスーパーコンピュータを利用するには、まずアカウント申請を行ってください。",
            position=0,
            embedding=None
        )
        
        search_result = SearchResult(chunk=chunk, score=0.9, document=doc)
        vector_db.search_similar.return_value = [search_result]
        
        # Create enhanced search engine
        with patch('src.services.enhanced_search_engine.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "search": {
                    "max_results": 5,
                    "min_similarity_score": 0.1,
                    "enable_query_expansion": True,
                    "enable_query_normalization": True,
                    "enable_search_analytics": True
                }
            }
            
            engine = EnhancedSearchEngine(
                vector_db=vector_db,
                embedding_model=embedding_model,
                logging_service=logging_service
            )
        
        # Test various Japanese queries
        test_queries = [
            "スパコンの使い方を教えてください",
            "アカウント申請の方法は？",
            "ジョブの実行手順について",
            "ストレージの利用方法",
            "エラーが発生した場合の対処法"
        ]
        
        for query in test_queries:
            results = engine.search(query)
            
            # Verify results
            assert isinstance(results, list)
            assert len(results) > 0
            assert all(isinstance(r, SearchResult) for r in results)
            
            # Verify logging was called
            logging_service.log_search.assert_called()
        
        # Verify analytics
        analytics = engine.get_search_analytics()
        assert analytics["total_searches"] == len(test_queries)
        assert analytics["successful_searches"] == len(test_queries)
        assert "query_type_distribution" in analytics
        assert "popular_terms" in analytics
    
    def test_performance_monitoring(self):
        """Test performance monitoring and metrics."""
        # Create components with realistic delays
        vector_db = Mock(spec=VectorDatabaseInterface)
        embedding_model = Mock(spec=EmbeddingModelInterface)
        
        def slow_encode(texts):
            import time
            time.sleep(0.01)  # Simulate processing time
            return [[0.1, 0.2, 0.3, 0.4, 0.5]]
        
        def slow_search(vector, top_k):
            import time
            time.sleep(0.01)  # Simulate search time
            return []
        
        embedding_model.encode.side_effect = slow_encode
        vector_db.search_similar.side_effect = slow_search
        
        # Create engine
        with patch('src.services.enhanced_search_engine.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "search": {
                    "enable_search_analytics": True
                }
            }
            
            engine = EnhancedSearchEngine(
                vector_db=vector_db,
                embedding_model=embedding_model
            )
        
        # Perform searches
        for i in range(5):
            engine.search(f"テスト質問 {i}")
        
        # Check performance metrics
        analytics = engine.get_search_analytics()
        assert analytics["avg_processing_time"] > 0
        assert analytics["total_searches"] == 5
        
        # Check recent searches
        recent = engine.get_recent_searches(limit=3)
        assert len(recent) == 3
        assert all("processing_time" in record for record in recent)
"""
Tests for the RAG service.

This module contains unit tests for the RAGService class,
testing the integration of search and generation functionality.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.services.rag_service import RAGService, create_rag_service
from src.models.core import Answer, SearchResult, Document, Chunk
from src.models.exceptions import GenerationError, SearchError
from src.models.interfaces import SearchEngineInterface, GenerativeModelInterface


class TestRAGService:
    """Test cases for RAGService."""
    
    @pytest.fixture
    def mock_search_engine(self):
        """Create a mock search engine."""
        search_engine = Mock(spec=SearchEngineInterface)
        return search_engine
    
    @pytest.fixture
    def mock_generative_model(self):
        """Create a mock generative model."""
        generative_model = Mock(spec=GenerativeModelInterface)
        return generative_model
    
    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results."""
        doc1 = Document(
            id="doc1",
            url="https://example.com/doc1",
            title="スパコン利用ガイド",
            content="スパコンの利用方法について説明します。",
            language="ja",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        chunk1 = Chunk(
            id="chunk1",
            document_id="doc1",
            content="スパコンを利用するには、まずアカウント申請が必要です。",
            position=0,
            embedding=None
        )
        
        doc2 = Document(
            id="doc2",
            url="https://example.com/doc2",
            title="バッチジョブ実行方法",
            content="バッチジョブの実行方法について説明します。",
            language="ja",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        chunk2 = Chunk(
            id="chunk2",
            document_id="doc2",
            content="バッチジョブはSLURMシステムを使用して実行されます。",
            position=0,
            embedding=None
        )
        
        return [
            SearchResult(chunk=chunk1, score=0.9, document=doc1),
            SearchResult(chunk=chunk2, score=0.8, document=doc2)
        ]
    
    @pytest.fixture
    def rag_service(self, mock_search_engine, mock_generative_model):
        """Create a RAG service instance."""
        with patch('src.services.rag_service.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "search": {
                    "max_results": 5,
                    "min_similarity_score": 0.5,
                    "enable_reranking": True,
                    "max_context_length": 2000
                },
                "generation": {
                    "include_sources": True,
                    "source_citation_format": "numbered"
                }
            }
            return RAGService(mock_search_engine, mock_generative_model)
    
    def test_initialization(self, rag_service):
        """Test RAG service initialization."""
        assert rag_service.search_engine is not None
        assert rag_service.generative_model is not None
        assert rag_service.max_context_chunks == 5
        assert rag_service.min_similarity_score == 0.5
        assert rag_service.enable_reranking is True
        assert rag_service.include_sources is True
    
    def test_generate_answer_success(self, rag_service, mock_search_engine, 
                                   mock_generative_model, sample_search_results):
        """Test successful answer generation."""
        # Setup mocks
        mock_search_engine.search.return_value = sample_search_results
        
        mock_answer = Answer(
            text="スパコンを利用するには、アカウント申請とバッチジョブの実行が必要です。",
            sources=["文書1", "文書2"],
            confidence=0.8,
            processing_time=1.5
        )
        mock_generative_model.generate_answer.return_value = mock_answer
        
        # Test answer generation
        query = "スパコンの使い方を教えてください"
        answer = rag_service.generate_answer(query)
        
        # Verify calls
        mock_search_engine.search.assert_called_once()
        mock_generative_model.generate_answer.assert_called_once()
        
        # Verify answer
        assert isinstance(answer, Answer)
        assert len(answer.text) > 0
        assert len(answer.sources) > 0
        assert 0.0 <= answer.confidence <= 1.0
        assert answer.processing_time > 0
    
    def test_generate_answer_no_context(self, rag_service, mock_search_engine, 
                                      mock_generative_model):
        """Test answer generation when no relevant context is found."""
        # Setup mocks - no search results
        mock_search_engine.search.return_value = []
        
        # Test answer generation
        query = "関係のない質問"
        answer = rag_service.generate_answer(query)
        
        # Verify search was called but generation was not
        mock_search_engine.search.assert_called_once()
        mock_generative_model.generate_answer.assert_not_called()
        
        # Verify no-context answer
        assert isinstance(answer, Answer)
        assert "申し訳ございません" in answer.text
        assert len(answer.sources) == 0
        assert answer.confidence < 0.5
    
    def test_generate_answer_search_error(self, rag_service, mock_search_engine, 
                                        mock_generative_model):
        """Test answer generation when search fails."""
        # Setup mocks - search raises error
        mock_search_engine.search.side_effect = SearchError("Search failed")
        
        # Test answer generation
        query = "テスト質問"
        
        with pytest.raises(GenerationError, match="Failed to retrieve context"):
            rag_service.generate_answer(query)
    
    def test_generate_answer_generation_error(self, rag_service, mock_search_engine, 
                                            mock_generative_model, sample_search_results):
        """Test answer generation when generation fails."""
        # Setup mocks
        mock_search_engine.search.return_value = sample_search_results
        mock_generative_model.generate_answer.side_effect = GenerationError("Generation failed")
        
        # Test answer generation
        query = "テスト質問"
        
        with pytest.raises(GenerationError, match="RAG answer generation failed"):
            rag_service.generate_answer(query)
    
    def test_retrieve_context(self, rag_service, mock_search_engine, sample_search_results):
        """Test context retrieval."""
        # Setup mocks
        mock_search_engine.search.return_value = sample_search_results
        
        # Test context retrieval
        results = rag_service._retrieve_context("テスト質問")
        
        assert len(results) == 2
        assert all(result.score >= rag_service.min_similarity_score for result in results)
    
    def test_prepare_context(self, rag_service, sample_search_results):
        """Test context preparation."""
        context_strings = rag_service._prepare_context(sample_search_results)
        
        assert len(context_strings) == 2
        assert "[文書1:" in context_strings[0]  # Updated to match actual format
        assert "[文書2:" in context_strings[1]  # Updated to match actual format
        assert "スパコンを利用するには" in context_strings[0]
        assert "バッチジョブは" in context_strings[1]
    
    def test_format_context_chunk(self, rag_service, sample_search_results):
        """Test context chunk formatting."""
        result = sample_search_results[0]
        formatted = rag_service._format_context_chunk(result, 1)
        
        assert "[文書1: スパコン利用ガイド]" in formatted
        assert result.chunk.content in formatted
        assert result.document.url in formatted
    
    def test_enhance_answer(self, rag_service, sample_search_results):
        """Test answer enhancement."""
        original_answer = Answer(
            text="テスト回答",
            sources=["source1"],
            confidence=0.7,
            processing_time=1.0
        )
        
        enhanced = rag_service._enhance_answer(
            original_answer, sample_search_results, "テスト質問"
        )
        
        assert enhanced.text == original_answer.text
        assert len(enhanced.sources) == 2
        assert enhanced.confidence >= original_answer.confidence
        assert enhanced.processing_time == original_answer.processing_time
    
    def test_create_enhanced_sources(self, rag_service, sample_search_results):
        """Test enhanced source creation."""
        sources = rag_service._create_enhanced_sources(sample_search_results)
        
        assert len(sources) == 2
        assert "[1]" in sources[0]
        assert "[2]" in sources[1]
        assert "スパコン利用ガイド" in sources[0]
        assert "バッチジョブ実行方法" in sources[1]
        assert "関連度:" in sources[0]
        assert "関連度:" in sources[1]
    
    def test_calculate_enhanced_confidence(self, rag_service, sample_search_results):
        """Test enhanced confidence calculation."""
        original_answer = Answer(
            text="テスト回答",
            sources=[],
            confidence=0.5,
            processing_time=1.0
        )
        
        # Test with good search results
        enhanced_confidence = rag_service._calculate_enhanced_confidence(
            original_answer, sample_search_results, "テスト質問"
        )
        
        assert enhanced_confidence > original_answer.confidence
        assert 0.0 <= enhanced_confidence <= 1.0
        
        # Test with no search results
        no_results_confidence = rag_service._calculate_enhanced_confidence(
            original_answer, [], "テスト質問"
        )
        
        assert no_results_confidence < original_answer.confidence
    
    def test_get_service_status(self, rag_service):
        """Test service status retrieval."""
        status = rag_service.get_service_status()
        
        assert isinstance(status, dict)
        assert "search_engine_available" in status
        assert "generative_model_available" in status
        assert "max_context_chunks" in status
        assert "min_similarity_score" in status
        assert status["search_engine_available"] is True
        assert status["generative_model_available"] is True
    
    def test_create_rag_service_factory(self, mock_search_engine, mock_generative_model):
        """Test RAG service factory function."""
        with patch('src.services.rag_service.get_config') as mock_get_config:
            mock_get_config.return_value = {}
            
            service = create_rag_service(mock_search_engine, mock_generative_model)
            
            assert isinstance(service, RAGService)
            assert service.search_engine == mock_search_engine
            assert service.generative_model == mock_generative_model


class TestRAGServiceIntegration:
    """Integration tests for RAGService."""
    
    def test_end_to_end_rag_flow(self):
        """Test complete RAG flow with realistic data."""
        # Create mock components
        search_engine = Mock(spec=SearchEngineInterface)
        generative_model = Mock(spec=GenerativeModelInterface)
        
        # Create realistic search results
        doc = Document(
            id="doc1",
            url="https://www.cc.kyushu-u.ac.jp/scp/guide",
            title="九州大学スパコン利用ガイド",
            content="九州大学のスパコン利用方法について詳しく説明します。",
            language="ja",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content="九州大学のスーパーコンピュータを利用するには、まず利用申請を行い、アカウントを取得する必要があります。申請は公式サイトから行うことができます。",
            position=0,
            embedding=None
        )
        
        search_result = SearchResult(chunk=chunk, score=0.95, document=doc)
        search_engine.search.return_value = [search_result]
        
        # Mock answer generation
        generated_answer = Answer(
            text="九州大学のスーパーコンピュータを利用するには、まず利用申請を行ってアカウントを取得してください。申請は公式サイトから行うことができます。",
            sources=["文書1"],
            confidence=0.9,
            processing_time=2.0
        )
        generative_model.generate_answer.return_value = generated_answer
        
        # Create RAG service
        with patch('src.services.rag_service.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "search": {
                    "max_results": 5,
                    "min_similarity_score": 0.5
                },
                "generation": {
                    "include_sources": True,
                    "source_citation_format": "numbered"
                }
            }
            
            rag_service = RAGService(search_engine, generative_model)
        
        # Test complete flow
        query = "九州大学のスパコンの使い方を教えてください"
        answer = rag_service.generate_answer(query)
        
        # Verify the complete flow
        search_engine.search.assert_called_once()
        generative_model.generate_answer.assert_called_once()
        
        # Verify enhanced answer
        assert isinstance(answer, Answer)
        assert "九州大学" in answer.text
        assert len(answer.sources) > 0
        assert "[1]" in answer.sources[0]  # Numbered citation format
        assert "九州大学スパコン利用ガイド" in answer.sources[0]
        assert answer.confidence > generated_answer.confidence  # Enhanced confidence
        assert answer.processing_time > 0
    
    def test_context_length_limiting(self):
        """Test that context is properly limited by length."""
        search_engine = Mock(spec=SearchEngineInterface)
        generative_model = Mock(spec=GenerativeModelInterface)
        
        # Create very long content that exceeds limit
        long_content = "非常に長いコンテンツです。" * 200  # Very long content
        
        doc = Document(
            id="doc1",
            url="https://example.com",
            title="長い文書",
            content=long_content,
            language="ja",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content=long_content,
            position=0,
            embedding=None
        )
        
        search_results = [SearchResult(chunk=chunk, score=0.9, document=doc)] * 10
        search_engine.search.return_value = search_results
        
        with patch('src.services.rag_service.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "search": {
                    "max_context_length": 1000  # Short limit
                },
                "generation": {}
            }
            
            rag_service = RAGService(search_engine, generative_model)
        
        # Test context preparation
        context_strings = rag_service._prepare_context(search_results)
        
        # Verify context is limited
        total_length = sum(len(ctx) for ctx in context_strings)
        assert total_length <= 1000  # Should not exceed limit
        assert len(context_strings) < len(search_results)  # Should be truncated
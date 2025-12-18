"""
Tests for the search logging service.

This module contains unit tests for the SearchLoggingService class,
testing comprehensive logging and analytics functionality.
"""

import pytest
import tempfile
import json
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from pathlib import Path

from src.services.search_logging_service import SearchLoggingService, create_search_logging_service
from src.models.core import SearchResult, Document, Chunk, Answer


class TestSearchLoggingService:
    """Test cases for SearchLoggingService."""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create a temporary log file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            return f.name
    
    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results."""
        doc = Document(
            id="doc1",
            url="https://example.com/doc1",
            title="テスト文書",
            content="テスト内容",
            language="ja",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content="これはテストチャンクです。",
            position=0,
            embedding=None
        )
        
        return [SearchResult(chunk=chunk, score=0.9, document=doc)]
    
    @pytest.fixture
    def sample_answer(self):
        """Create a sample answer."""
        return Answer(
            text="これはテスト回答です。",
            sources=["文書1", "文書2"],
            confidence=0.8,
            processing_time=1.5
        )
    
    @pytest.fixture
    def logging_service(self, temp_log_file):
        """Create a logging service instance."""
        with patch('src.services.search_logging_service.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "logging": {
                    "level": "INFO",
                    "file_path": temp_log_file,
                    "enable_file": True,
                    "enable_console": False
                },
                "performance": {
                    "enable_metrics": True,
                    "max_log_entries": 1000
                }
            }
            return SearchLoggingService()
    
    def test_initialization(self, logging_service):
        """Test logging service initialization."""
        assert logging_service.enable_analytics is True
        assert logging_service.max_log_entries == 1000
        assert len(logging_service.search_logs) == 0
        assert len(logging_service.answer_logs) == 0
        assert len(logging_service.document_logs) == 0
    
    def test_log_search(self, logging_service, sample_search_results):
        """Test search logging functionality."""
        query = "テスト質問"
        processing_time = 1.2
        
        # Log search
        logging_service.log_search(query, sample_search_results, processing_time)
        
        # Verify log entry
        assert len(logging_service.search_logs) == 1
        log_entry = logging_service.search_logs[0]
        
        assert log_entry["query"] == query
        assert log_entry["num_results"] == len(sample_search_results)
        assert log_entry["processing_time"] == processing_time
        assert log_entry["top_score"] == sample_search_results[0].score
        assert "timestamp" in log_entry
        assert "chunk_ids" in log_entry
        assert "document_ids" in log_entry
    
    def test_log_document_processing(self, logging_service):
        """Test document processing logging."""
        document_id = "doc123"
        status = "success"
        
        # Log successful processing
        logging_service.log_document_processing(document_id, status)
        
        # Verify log entry
        assert len(logging_service.document_logs) == 1
        log_entry = logging_service.document_logs[0]
        
        assert log_entry["document_id"] == document_id
        assert log_entry["status"] == status
        assert log_entry["error"] is None
        assert "timestamp" in log_entry
        
        # Log failed processing
        error_message = "Processing failed"
        logging_service.log_document_processing(document_id, "error", error_message)
        
        # Verify error log entry
        assert len(logging_service.document_logs) == 2
        error_log = logging_service.document_logs[1]
        assert error_log["status"] == "error"
        assert error_log["error"] == error_message
    
    def test_log_answer_generation(self, logging_service, sample_answer):
        """Test answer generation logging."""
        query = "テスト質問"
        chunks_used = ["chunk1", "chunk2", "chunk3"]
        
        # Log answer generation
        logging_service.log_answer_generation(query, sample_answer, chunks_used)
        
        # Verify log entry
        assert len(logging_service.answer_logs) == 1
        log_entry = logging_service.answer_logs[0]
        
        assert log_entry["query"] == query
        assert log_entry["answer_length"] == len(sample_answer.text)
        assert log_entry["confidence"] == sample_answer.confidence
        assert log_entry["processing_time"] == sample_answer.processing_time
        assert log_entry["num_sources"] == len(sample_answer.sources)
        assert log_entry["chunks_used"] == chunks_used
        assert "timestamp" in log_entry
    
    def test_search_metrics_update(self, logging_service, sample_search_results):
        """Test search metrics updating."""
        # Initial metrics
        assert logging_service.metrics["search_count"] == 0
        assert logging_service.metrics["total_search_time"] == 0.0
        
        # Log multiple searches
        queries = ["質問1", "質問2", "質問3"]
        processing_times = [1.0, 1.5, 2.0]
        
        for query, time in zip(queries, processing_times):
            logging_service.log_search(query, sample_search_results, time)
        
        # Verify metrics
        assert logging_service.metrics["search_count"] == len(queries)
        assert logging_service.metrics["total_search_time"] == sum(processing_times)
        assert logging_service.metrics["avg_search_time"] == sum(processing_times) / len(processing_times)
        
        # Verify query analytics
        assert len(logging_service.query_analytics["processing_times"]) == len(queries)
        # Check that some terms were recorded (the exact terms depend on word splitting)
        assert len(logging_service.query_analytics["popular_terms"]) > 0
    
    def test_get_search_statistics(self, logging_service, sample_search_results, sample_answer):
        """Test search statistics generation."""
        # Log some data
        logging_service.log_search("テスト質問1", sample_search_results, 1.0)
        logging_service.log_search("テスト質問2", [], 0.5)  # No results
        logging_service.log_answer_generation("テスト質問1", sample_answer, ["chunk1"])
        logging_service.log_document_processing("doc1", "success")
        
        # Get statistics
        stats = logging_service.get_search_statistics()
        
        # Verify overview
        assert stats["overview"]["total_searches"] == 2
        assert stats["overview"]["total_answers"] == 1
        assert stats["overview"]["total_documents_processed"] == 1
        assert stats["overview"]["avg_search_time"] == 0.75  # (1.0 + 0.5) / 2
        
        # Verify query analytics
        assert "popular_terms" in stats["query_analytics"]
        assert "result_distribution" in stats["query_analytics"]
        
        # Verify performance metrics
        assert "min_processing_time" in stats["performance"]
        assert "max_processing_time" in stats["performance"]
        assert "avg_processing_time" in stats["performance"]
    
    def test_get_recent_searches(self, logging_service, sample_search_results):
        """Test recent searches retrieval."""
        # Log multiple searches
        queries = [f"質問{i}" for i in range(5)]
        for query in queries:
            logging_service.log_search(query, sample_search_results, 1.0)
        
        # Get recent searches
        recent = logging_service.get_recent_searches(limit=3)
        
        assert len(recent) == 3
        # Should return the most recent searches
        for i, log_entry in enumerate(recent):
            expected_query = queries[-(3-i)]  # Last 3 in reverse order
            assert log_entry["query"] == expected_query
    
    def test_get_search_trends(self, logging_service, sample_search_results):
        """Test search trends analysis."""
        # Log searches with different timestamps
        base_time = datetime.now()
        
        # Mock timestamps for different hours
        with patch('src.services.search_logging_service.datetime') as mock_datetime:
            for i in range(3):
                mock_datetime.now.return_value = base_time - timedelta(hours=i)
                mock_datetime.fromisoformat = datetime.fromisoformat
                logging_service.log_search(f"質問{i}", sample_search_results, 1.0 + i * 0.5)
        
        # Get trends for last 24 hours
        trends = logging_service.get_search_trends(hours=24)
        
        assert trends["period_hours"] == 24
        assert trends["total_searches"] == 3
        assert "hourly_distribution" in trends
        assert "averages" in trends
        assert "ranges" in trends
        
        # Verify averages
        assert trends["averages"]["query_length"] > 0
        assert trends["averages"]["result_count"] > 0
        assert trends["averages"]["processing_time"] > 0
    
    def test_export_logs_json(self, logging_service, sample_search_results, sample_answer):
        """Test JSON log export."""
        # Log some data
        logging_service.log_search("テスト質問", sample_search_results, 1.0)
        logging_service.log_answer_generation("テスト質問", sample_answer, ["chunk1"])
        logging_service.log_document_processing("doc1", "success")
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        # Test export
        success = logging_service.export_logs(export_path, format="json")
        assert success is True
        
        # Verify exported data
        with open(export_path, 'r', encoding='utf-8') as f:
            exported_data = json.load(f)
        
        assert "metadata" in exported_data
        assert "search_logs" in exported_data
        assert "answer_logs" in exported_data
        assert "document_logs" in exported_data
        assert "metrics" in exported_data
        
        assert len(exported_data["search_logs"]) == 1
        assert len(exported_data["answer_logs"]) == 1
        assert len(exported_data["document_logs"]) == 1
        
        # Cleanup
        Path(export_path).unlink()
    
    def test_export_logs_csv(self, logging_service, sample_search_results):
        """Test CSV log export."""
        # Log some data
        logging_service.log_search("テスト質問", sample_search_results, 1.0)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            export_path = f.name
        
        # Test export
        success = logging_service.export_logs(export_path, format="csv")
        assert success is True
        
        # Verify file exists and has content
        csv_path = Path(export_path)
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0
        
        # Cleanup
        csv_path.unlink()
    
    def test_reset_metrics(self, logging_service, sample_search_results, sample_answer):
        """Test metrics reset functionality."""
        # Log some data
        logging_service.log_search("テスト質問", sample_search_results, 1.0)
        logging_service.log_answer_generation("テスト質問", sample_answer, ["chunk1"])
        logging_service.log_document_processing("doc1", "success")
        
        # Verify data exists
        assert len(logging_service.search_logs) > 0
        assert len(logging_service.answer_logs) > 0
        assert len(logging_service.document_logs) > 0
        assert logging_service.metrics["search_count"] > 0
        
        # Reset metrics
        logging_service.reset_metrics()
        
        # Verify everything is cleared
        assert len(logging_service.search_logs) == 0
        assert len(logging_service.answer_logs) == 0
        assert len(logging_service.document_logs) == 0
        assert logging_service.metrics["search_count"] == 0
        assert logging_service.metrics["total_search_time"] == 0.0
        assert len(logging_service.query_analytics["popular_terms"]) == 0
    
    def test_analytics_disabled(self):
        """Test behavior when analytics are disabled."""
        with patch('src.services.search_logging_service.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "logging": {"level": "INFO"},
                "performance": {"enable_metrics": False}
            }
            
            service = SearchLoggingService()
            assert service.enable_analytics is False
            
            # Test that analytics methods return appropriate responses
            stats = service.get_search_statistics()
            assert stats == {"analytics_disabled": True}
            
            recent = service.get_recent_searches()
            assert recent == []
            
            trends = service.get_search_trends()
            assert trends == {"analytics_disabled": True}
    
    def test_factory_function(self):
        """Test logging service factory function."""
        with patch('src.services.search_logging_service.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "logging": {"level": "INFO"},
                "performance": {"enable_metrics": True}
            }
            
            service = create_search_logging_service()
            assert isinstance(service, SearchLoggingService)
            assert service.enable_analytics is True


class TestSearchLoggingServiceIntegration:
    """Integration tests for SearchLoggingService."""
    
    def test_comprehensive_logging_workflow(self):
        """Test complete logging workflow with realistic data."""
        with patch('src.services.search_logging_service.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "logging": {
                    "level": "INFO",
                    "enable_file": False,  # Disable file logging for test
                    "enable_console": False
                },
                "performance": {
                    "enable_metrics": True,
                    "max_log_entries": 100
                }
            }
            
            service = SearchLoggingService()
        
        # Simulate a complete RAG workflow
        queries = [
            "九州大学のスパコンの使い方を教えてください",
            "アカウント申請の方法は？",
            "ジョブの実行手順について",
            "エラーが発生した場合の対処法",
            "ストレージの利用方法"
        ]
        
        for i, query in enumerate(queries):
            # Create realistic search results
            doc = Document(
                id=f"doc{i+1}",
                url=f"https://www.cc.kyushu-u.ac.jp/scp/guide{i+1}",
                title=f"ガイド{i+1}",
                content=f"ガイド{i+1}の内容",
                language="ja",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            chunk = Chunk(
                id=f"chunk{i+1}",
                document_id=f"doc{i+1}",
                content=f"これは{query}に関する情報です。",
                position=0,
                embedding=None
            )
            
            results = [SearchResult(chunk=chunk, score=0.9 - i * 0.1, document=doc)]
            
            # Log search
            service.log_search(query, results, 1.0 + i * 0.2)
            
            # Log document processing
            service.log_document_processing(f"doc{i+1}", "success")
            
            # Log answer generation
            answer = Answer(
                text=f"これは{query}に対する回答です。",
                sources=[f"文書{i+1}"],
                confidence=0.8 + i * 0.02,
                processing_time=1.5 + i * 0.1
            )
            service.log_answer_generation(query, answer, [f"chunk{i+1}"])
        
        # Verify comprehensive statistics
        stats = service.get_search_statistics()
        
        assert stats["overview"]["total_searches"] == len(queries)
        assert stats["overview"]["total_answers"] == len(queries)
        assert stats["overview"]["total_documents_processed"] == len(queries)
        
        # Verify query analytics
        popular_terms = stats["query_analytics"]["popular_terms"]
        assert "スパコン" in popular_terms or "九州大学" in popular_terms
        
        # Verify trends
        trends = service.get_search_trends(hours=1)
        assert trends["total_searches"] == len(queries)
        assert trends["averages"]["processing_time"] > 0
        
        # Test export functionality
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        success = service.export_logs(export_path, format="json")
        assert success is True
        
        # Verify exported data
        with open(export_path, 'r', encoding='utf-8') as f:
            exported_data = json.load(f)
        
        assert exported_data["metadata"]["total_searches"] == len(queries)
        assert len(exported_data["search_logs"]) == len(queries)
        assert len(exported_data["answer_logs"]) == len(queries)
        assert len(exported_data["document_logs"]) == len(queries)
        
        # Cleanup
        Path(export_path).unlink()
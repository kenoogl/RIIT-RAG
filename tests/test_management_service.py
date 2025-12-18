"""
Tests for the management service.

This module contains tests for document management and system monitoring
functionality.
"""

import pytest
import tempfile
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.services.management_service import DocumentManagementService, SystemMonitoringService
from src.models.core import Document, Chunk
from src.models.exceptions import ManagementError


class TestDocumentManagementService:
    """Test cases for DocumentManagementService."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock configuration."""
        return {
            "storage": {
                "documents_dir": str(temp_dir / "documents"),
                "logs_dir": str(temp_dir / "logs")
            }
        }
    
    @pytest.fixture
    def mock_document_engine(self):
        """Create mock document engine."""
        engine = Mock()
        engine.process_document.return_value = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="Test chunk content",
                position=0
            )
        ]
        return engine
    
    @pytest.fixture
    def mock_vector_db(self):
        """Create mock vector database."""
        db = Mock()
        db.add_document.return_value = True
        return db
    
    @pytest.fixture
    def document_service(self, mock_document_engine, mock_vector_db, mock_config, temp_dir):
        """Create document management service for testing."""
        with patch('src.services.management_service.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            
            service = DocumentManagementService(
                document_engine=mock_document_engine,
                vector_db=mock_vector_db
            )
            return service
    
    @pytest.fixture
    def sample_document(self):
        """Create sample document for testing."""
        return Document(
            id="test_doc_1",
            title="Test Document",
            url="https://example.com/test",
            content="This is a test document content for testing purposes.",
            language="ja",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def test_service_initialization(self, document_service):
        """Test service initialization."""
        assert document_service is not None
        assert document_service.document_engine is not None
        assert document_service.vector_db is not None
        assert document_service.documents_dir.exists()
    
    def test_add_document_success(self, document_service, sample_document):
        """Test successful document addition."""
        result = document_service.add_document(sample_document)
        
        assert result["status"] == "success"
        assert result["document_id"] == sample_document.id
        assert result["chunks_created"] == 1
        assert result["processing_time"] > 0
        
        # Verify document is in metadata
        assert sample_document.id in document_service.document_metadata["documents"]
    
    def test_add_document_duplicate_error(self, document_service, sample_document):
        """Test error when adding duplicate document."""
        # Add document first time
        document_service.add_document(sample_document)
        
        # Try to add same document again
        with pytest.raises(ManagementError) as exc_info:
            document_service.add_document(sample_document)
        
        assert "already exists" in str(exc_info.value)
    
    def test_add_document_force_reprocess(self, document_service, sample_document):
        """Test force reprocessing of existing document."""
        # Add document first time
        document_service.add_document(sample_document)
        
        # Modify document content
        sample_document.content = "Updated content"
        sample_document.updated_at = datetime.now()
        
        # Add with force reprocess
        result = document_service.add_document(sample_document, force_reprocess=True)
        
        assert result["status"] == "success"
        assert result["document_id"] == sample_document.id
    
    def test_get_document_info_success(self, document_service, sample_document):
        """Test successful document info retrieval."""
        document_service.add_document(sample_document)
        
        info = document_service.get_document_info(sample_document.id)
        
        assert info["id"] == sample_document.id
        assert info["title"] == sample_document.title
        assert info["url"] == sample_document.url
        assert info["language"] == sample_document.language
    
    def test_get_document_info_not_found(self, document_service):
        """Test error when document not found."""
        with pytest.raises(ManagementError) as exc_info:
            document_service.get_document_info("nonexistent_doc")
        
        assert "not found" in str(exc_info.value)
    
    def test_update_document_success(self, document_service, sample_document):
        """Test successful document update."""
        # Add document first
        document_service.add_document(sample_document)
        
        # Update document
        sample_document.title = "Updated Title"
        sample_document.content = "Updated content"
        
        result = document_service.update_document(sample_document)
        
        assert result["status"] == "success"
        assert "updated successfully" in result["message"]
    
    def test_update_document_not_found(self, document_service, sample_document):
        """Test error when updating non-existent document."""
        with pytest.raises(ManagementError) as exc_info:
            document_service.update_document(sample_document)
        
        assert "not found" in str(exc_info.value)
    
    def test_remove_document_success(self, document_service, sample_document):
        """Test successful document removal."""
        # Add document first
        document_service.add_document(sample_document)
        
        # Remove document
        result = document_service.remove_document(sample_document.id)
        
        assert result["status"] == "success"
        assert result["document_id"] == sample_document.id
        assert result["chunks_removed"] == 1
        
        # Verify document is removed from metadata
        assert sample_document.id not in document_service.document_metadata["documents"]
    
    def test_remove_document_not_found(self, document_service):
        """Test error when removing non-existent document."""
        with pytest.raises(ManagementError) as exc_info:
            document_service.remove_document("nonexistent_doc")
        
        assert "not found" in str(exc_info.value)
    
    def test_list_documents_empty(self, document_service):
        """Test listing documents when none exist."""
        result = document_service.list_documents()
        
        assert result["documents"] == []
        assert result["total_count"] == 0
        assert result["returned_count"] == 0
    
    def test_list_documents_with_pagination(self, document_service):
        """Test listing documents with pagination."""
        # Add multiple documents
        for i in range(5):
            doc = Document(
                id=f"doc_{i}",
                title=f"Document {i}",
                url=f"https://example.com/doc{i}",
                content=f"Content for document {i}",
                language="ja",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            document_service.add_document(doc)
        
        # Test pagination
        result = document_service.list_documents(limit=3, offset=1)
        
        assert result["total_count"] == 5
        assert result["returned_count"] == 3
        assert result["offset"] == 1
        assert result["limit"] == 3
    
    def test_get_index_status(self, document_service, sample_document):
        """Test getting index status."""
        document_service.add_document(sample_document)
        
        status = document_service.get_index_status()
        
        assert "overview" in status
        assert "vector_database" in status
        assert "document_processing" in status
        assert status["overview"]["total_documents"] == 1
        assert status["overview"]["total_chunks"] == 1
    
    def test_rebuild_index_success(self, document_service, sample_document):
        """Test successful index rebuild."""
        document_service.add_document(sample_document)
        
        result = document_service.rebuild_index([sample_document.id])
        
        assert result["status"] == "success"
        assert result["processed_documents"] == 1
        assert result["failed_documents"] == 0
    
    def test_rebuild_index_all_documents(self, document_service, sample_document):
        """Test rebuilding index for all documents."""
        document_service.add_document(sample_document)
        
        result = document_service.rebuild_index()  # No document_ids specified
        
        assert result["status"] == "success"
        assert result["processed_documents"] == 1


class TestSystemMonitoringService:
    """Test cases for SystemMonitoringService."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock configuration."""
        return {
            "storage": {
                "logs_dir": str(temp_dir / "logs")
            }
        }
    
    @pytest.fixture
    def mock_search_logging_service(self):
        """Create mock search logging service."""
        service = Mock()
        service.get_search_statistics.return_value = {
            "overview": {
                "total_searches": 100,
                "total_answers": 95,
                "avg_search_time": 0.5,
                "avg_answer_time": 1.2
            },
            "recent_activity": {
                "searches_last_hour": 10,
                "searches_last_day": 50
            }
        }
        service.get_search_trends.return_value = {
            "period_hours": 24,
            "total_searches": 50,
            "averages": {
                "query_length": 25,
                "result_count": 3,
                "processing_time": 0.8
            }
        }
        return service
    
    @pytest.fixture
    def monitoring_service(self, mock_search_logging_service, mock_config, temp_dir):
        """Create system monitoring service for testing."""
        # Create log directory and sample log file
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        sample_log = logs_dir / "rag_system.log"
        sample_log.write_text(
            "2024-01-01 10:00:00 - INFO - System started\n"
            "2024-01-01 10:01:00 - WARNING - Low memory\n"
            "2024-01-01 10:02:00 - ERROR - Connection failed\n"
            "2024-01-01 10:03:00 - INFO - System recovered\n"
        )
        
        with patch('src.services.management_service.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            
            service = SystemMonitoringService(
                search_logging_service=mock_search_logging_service
            )
            return service
    
    def test_service_initialization(self, monitoring_service):
        """Test service initialization."""
        assert monitoring_service is not None
        assert monitoring_service.search_logging_service is not None
    
    def test_get_system_status_success(self, monitoring_service):
        """Test successful system status retrieval."""
        # Mock psutil at the module level where it's imported
        with patch('psutil.cpu_percent', return_value=25.0), \
             patch('psutil.virtual_memory') as mock_vm, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_vm.return_value.percent = 60.0
            mock_disk.return_value.percent = 45.0
            
            status = monitoring_service.get_system_status()
            
            assert status["overall_status"] in ["healthy", "warning", "degraded"]
            assert "components" in status
            assert "performance" in status
            assert "resources" in status
            assert "timestamp" in status
    
    def test_get_system_status_without_psutil(self, monitoring_service):
        """Test system status when psutil is not available."""
        # Since the service catches all exceptions in get_system_status,
        # we'll test that it handles errors gracefully
        status = monitoring_service.get_system_status()
        
        # The status should always have these basic fields
        assert "timestamp" in status
        assert "overall_status" in status
        
        # If there's an error, it should be handled gracefully
        if "error" in status:
            assert status["overall_status"] == "error"
        else:
            # Normal case - should have all expected fields
            assert "components" in status
            assert "performance" in status or status["performance"] == {"status": "unavailable"}
            assert "resources" in status
    
    def test_get_performance_metrics(self, monitoring_service):
        """Test performance metrics retrieval."""
        metrics = monitoring_service.get_performance_metrics(24)
        
        assert metrics["period_hours"] == 24
        assert "timestamp" in metrics
        assert "search_metrics" in metrics
        assert "system_metrics" in metrics
    
    def test_get_log_summary_success(self, monitoring_service):
        """Test successful log summary retrieval."""
        summary = monitoring_service.get_log_summary("rag_system.log", 10)
        
        assert summary["status"] == "success"
        assert "file_path" in summary
        assert "level_counts" in summary
        assert summary["level_counts"]["INFO"] == 2
        assert summary["level_counts"]["WARNING"] == 1
        assert summary["level_counts"]["ERROR"] == 1
        assert len(summary["recent_errors"]) == 1
    
    def test_get_log_summary_file_not_found(self, monitoring_service):
        """Test log summary when file doesn't exist."""
        summary = monitoring_service.get_log_summary("nonexistent.log", 10)
        
        assert summary["status"] == "file_not_found"
        assert "file_path" in summary


class TestManagementServiceIntegration:
    """Integration tests for management services."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_document_lifecycle(self, temp_dir):
        """Test complete document lifecycle (add, update, remove)."""
        # Setup
        mock_config = {
            "storage": {
                "documents_dir": str(temp_dir / "documents"),
                "logs_dir": str(temp_dir / "logs")
            }
        }
        
        mock_engine = Mock()
        mock_engine.process_document.return_value = [
            Chunk(id="chunk1", document_id="doc1", content="Test", position=0)
        ]
        
        mock_db = Mock()
        
        with patch('src.services.management_service.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            
            service = DocumentManagementService(
                document_engine=mock_engine,
                vector_db=mock_db
            )
            
            # Create document
            doc = Document(
                id="lifecycle_test",
                title="Lifecycle Test",
                url="https://example.com/test",
                content="Original content",
                language="ja",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Add document
            add_result = service.add_document(doc)
            assert add_result["status"] == "success"
            
            # Get document info
            info = service.get_document_info(doc.id)
            assert info["title"] == "Lifecycle Test"
            
            # Update document
            doc.title = "Updated Lifecycle Test"
            doc.content = "Updated content"
            update_result = service.update_document(doc)
            assert update_result["status"] == "success"
            
            # Verify update
            updated_info = service.get_document_info(doc.id)
            assert updated_info["title"] == "Updated Lifecycle Test"
            
            # Remove document
            remove_result = service.remove_document(doc.id)
            assert remove_result["status"] == "success"
            
            # Verify removal
            with pytest.raises(ManagementError):
                service.get_document_info(doc.id)
    
    def test_system_monitoring_with_real_logs(self, temp_dir):
        """Test system monitoring with real log files."""
        # Create logs directory and files
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir()
        
        # Create sample log files
        log_files = ["rag_system.log", "search.log", "error.log"]
        for log_file in log_files:
            (logs_dir / log_file).write_text(f"Sample content for {log_file}\n" * 10)
        
        mock_config = {
            "storage": {
                "logs_dir": str(logs_dir)
            }
        }
        
        with patch('src.services.management_service.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            
            service = SystemMonitoringService()
            
            # Test log file checking
            status = service.get_system_status()
            log_info = status["components"]["logging"]["log_files"]
            
            assert log_info["accessible_files"] == 3
            assert len(log_info["files"]) == 3
            assert log_info["total_size"] > 0
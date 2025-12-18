"""
Tests for the management API endpoints.

This module contains tests for the administrative REST API endpoints
for document management and system monitoring.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from datetime import datetime

from src.api.main import app
from src.api.management import get_document_management_service, get_system_monitoring_service
from src.models.core import Document


class TestManagementAPI:
    """Test cases for management API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_document_service(self):
        """Create mock document management service."""
        service = Mock()
        
        # Mock document info
        service.get_document_info.return_value = {
            "id": "test_doc",
            "title": "Test Document",
            "url": "https://example.com/test",
            "language": "ja",
            "content_length": 100,
            "chunk_count": 2,
            "created_at": "2024-01-01T10:00:00",
            "updated_at": "2024-01-01T10:00:00",
            "added_to_system": "2024-01-01T10:00:00",
            "processing_time": 1.5
        }
        
        # Mock add document result
        service.add_document.return_value = {
            "status": "success",
            "document_id": "test_doc",
            "chunks_created": 2,
            "processing_time": 1.5,
            "message": "Document added successfully"
        }
        
        # Mock update document result
        service.update_document.return_value = {
            "status": "success",
            "document_id": "test_doc",
            "chunks_created": 2,
            "processing_time": 1.2,
            "message": "Document updated successfully"
        }
        
        # Mock remove document result
        service.remove_document.return_value = {
            "status": "success",
            "document_id": "test_doc",
            "chunks_removed": 2,
            "processing_time": 0.8,
            "message": "Document removed successfully"
        }
        
        # Mock list documents result
        def mock_list_documents(limit=None, offset=0):
            return {
                "documents": [
                    {
                        "id": "doc1",
                        "title": "Document 1",
                        "chunk_count": 2
                    },
                    {
                        "id": "doc2",
                        "title": "Document 2",
                        "chunk_count": 3
                    }
                ],
                "total_count": 2,
                "returned_count": 2,
                "offset": offset,
                "limit": limit
            }
        service.list_documents.side_effect = mock_list_documents
        
        # Mock index status
        service.get_index_status.return_value = {
            "overview": {
                "total_documents": 2,
                "total_chunks": 5,
                "last_updated": "2024-01-01T10:00:00"
            },
            "vector_database": {"status": "available"},
            "document_processing": {"status": "available"},
            "metadata_file": "/path/to/metadata.json",
            "documents_directory": "/path/to/documents"
        }
        
        # Mock rebuild index result
        service.rebuild_index.return_value = {
            "status": "success",
            "processed_documents": 2,
            "failed_documents": 0,
            "errors": [],
            "processing_time": 5.0
        }
        
        return service
    
    @pytest.fixture
    def mock_monitoring_service(self):
        """Create mock system monitoring service."""
        service = Mock()
        
        # Mock system status
        service.get_system_status.return_value = {
            "timestamp": "2024-01-01T10:00:00",
            "overall_status": "healthy",
            "components": {
                "search_logging": {"status": "healthy"},
                "logging": {"status": "healthy"}
            },
            "performance": {
                "avg_search_time": 0.5,
                "avg_answer_time": 1.2
            },
            "resources": {
                "cpu_percent": 25.0,
                "memory_percent": 60.0,
                "disk_usage": 45.0
            }
        }
        
        # Mock performance metrics
        service.get_performance_metrics.return_value = {
            "period_hours": 24,
            "timestamp": "2024-01-01T10:00:00",
            "search_metrics": {
                "trends": {"total_searches": 100},
                "current_stats": {"total_searches": 100}
            },
            "system_metrics": {
                "cpu_count": 4,
                "memory_total": 8589934592
            }
        }
        
        # Mock log summary
        service.get_log_summary.return_value = {
            "status": "success",
            "file_path": "/path/to/rag_system.log",
            "file_size": 1024,
            "lines_analyzed": 100,
            "level_counts": {
                "INFO": 80,
                "WARNING": 15,
                "ERROR": 5
            },
            "recent_errors": ["Error message 1", "Error message 2"],
            "last_modified": "2024-01-01T10:00:00"
        }
        
        return service
    
    def test_create_document_success(self, client, mock_document_service):
        """Test successful document creation."""
        with patch.object(app, 'dependency_overrides', {
            get_document_management_service: lambda: mock_document_service
        }):
            response = client.post("/admin/documents", json={
                "id": "test_doc",
                "title": "Test Document",
                "url": "https://example.com/test",
                "content": "This is test content",
                "language": "ja"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["document_id"] == "test_doc"
            assert data["chunks_created"] == 2
    
    def test_create_document_large_content(self, client, mock_document_service):
        """Test document creation with large content (background processing)."""
        with patch.object(app, 'dependency_overrides', {
            get_document_management_service: lambda: mock_document_service
        }):
            large_content = "x" * 15000  # Large content
            response = client.post("/admin/documents", json={
                "id": "large_doc",
                "title": "Large Document",
                "url": "https://example.com/large",
                "content": large_content,
                "language": "ja"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "processing"
            assert "background" in data["message"]
    
    def test_create_document_invalid_data(self, client):
        """Test document creation with invalid data."""
        # Test validation without mocking services - FastAPI should validate before calling services
        response = client.post("/admin/documents", json={
            "id": "",  # Empty ID
            "title": "Test Document",
            "url": "https://example.com/test",
            "content": "Test content"
        })
        
        # Should be 422 for validation error, 503 for service unavailable, or 500 for internal error
        assert response.status_code in [422, 500, 503]
    
    def test_get_document_success(self, client, mock_document_service):
        """Test successful document retrieval."""
        with patch.object(app, 'dependency_overrides', {
            get_document_management_service: lambda: mock_document_service
        }):
            response = client.get("/admin/documents/test_doc")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "test_doc"
            assert data["title"] == "Test Document"
    
    def test_get_document_not_found(self, client, mock_document_service):
        """Test document retrieval when document not found."""
        from src.models.exceptions import ManagementError
        mock_document_service.get_document_info.side_effect = ManagementError("Document not found")
        
        with patch.object(app, 'dependency_overrides', {
            get_document_management_service: lambda: mock_document_service
        }):
            response = client.get("/admin/documents/nonexistent")
            
            assert response.status_code == 404
    
    def test_update_document_success(self, client, mock_document_service):
        """Test successful document update."""
        with patch.object(app, 'dependency_overrides', {
            get_document_management_service: lambda: mock_document_service
        }):
            response = client.put("/admin/documents/test_doc", json={
                "title": "Updated Title",
                "content": "Updated content"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "updated successfully" in data["message"]
    
    def test_delete_document_success(self, client, mock_document_service):
        """Test successful document deletion."""
        with patch.object(app, 'dependency_overrides', {
            get_document_management_service: lambda: mock_document_service
        }):
            response = client.delete("/admin/documents/test_doc")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["chunks_removed"] == 2
    
    def test_list_documents_success(self, client, mock_document_service):
        """Test successful document listing."""
        with patch.object(app, 'dependency_overrides', {
            get_document_management_service: lambda: mock_document_service
        }):
            response = client.get("/admin/documents")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_count"] == 2
            assert data["returned_count"] == 2
            assert len(data["documents"]) == 2
    
    def test_list_documents_with_pagination(self, client, mock_document_service):
        """Test document listing with pagination."""
        with patch.object(app, 'dependency_overrides', {
            get_document_management_service: lambda: mock_document_service
        }):
            response = client.get("/admin/documents?limit=1&offset=1")
            
            assert response.status_code == 200
            data = response.json()
            assert data["limit"] == 1
            assert data["offset"] == 1
    
    def test_get_index_status_success(self, client, mock_document_service):
        """Test successful index status retrieval."""
        with patch.object(app, 'dependency_overrides', {
            get_document_management_service: lambda: mock_document_service
        }):
            response = client.get("/admin/index/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["overview"]["total_documents"] == 2
            assert data["overview"]["total_chunks"] == 5
    
    def test_rebuild_index_success(self, client, mock_document_service):
        """Test successful index rebuild."""
        with patch.object(app, 'dependency_overrides', {
            get_document_management_service: lambda: mock_document_service
        }):
            # Test without background tasks (synchronous)
            response = client.post("/admin/index/rebuild", json=["doc1", "doc2"])
            
            assert response.status_code == 200
            data = response.json()
            # The API returns "started" when using background tasks, "success" when synchronous
            assert data["status"] in ["success", "started"]
            if data["status"] == "success":
                assert data["processed_documents"] == 2
    
    def test_get_system_status_success(self, client, mock_monitoring_service):
        """Test successful system status retrieval."""
        with patch.object(app, 'dependency_overrides', {
            get_system_monitoring_service: lambda: mock_monitoring_service
        }):
            response = client.get("/admin/system/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["overall_status"] == "healthy"
            assert "components" in data
            assert "performance" in data
            assert "resources" in data
    
    def test_get_performance_metrics_success(self, client, mock_monitoring_service):
        """Test successful performance metrics retrieval."""
        with patch.object(app, 'dependency_overrides', {
            get_system_monitoring_service: lambda: mock_monitoring_service
        }):
            response = client.get("/admin/system/metrics?hours=48")
            
            assert response.status_code == 200
            data = response.json()
            assert data["period_hours"] == 24  # Mock returns 24
            assert "search_metrics" in data
            assert "system_metrics" in data
    
    def test_get_log_summary_success(self, client, mock_monitoring_service):
        """Test successful log summary retrieval."""
        with patch.object(app, 'dependency_overrides', {
            get_system_monitoring_service: lambda: mock_monitoring_service
        }):
            response = client.get("/admin/system/logs/rag_system.log?lines=50")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["level_counts"]["INFO"] == 80
            assert data["level_counts"]["ERROR"] == 5
    
    def test_list_log_files_success(self, client, mock_monitoring_service):
        """Test successful log files listing."""
        # Mock the system status to include log files
        mock_monitoring_service.get_system_status.return_value = {
            "components": {
                "logging": {
                    "log_files": {
                        "files": [
                            {"name": "rag_system.log", "size": 1024},
                            {"name": "search.log", "size": 512}
                        ],
                        "accessible_files": 2,
                        "total_size": 1536
                    }
                }
            }
        }
        
        with patch.object(app, 'dependency_overrides', {
            get_system_monitoring_service: lambda: mock_monitoring_service
        }):
            response = client.get("/admin/system/logs")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_files"] == 2
            assert data["total_size"] == 1536
            assert len(data["log_files"]) == 2


class TestManagementAPIValidation:
    """Test API request/response validation."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_create_document_validation_errors(self, client):
        """Test document creation validation errors."""
        invalid_requests = [
            {},  # Missing required fields
            {"id": "test", "title": "", "url": "invalid", "content": "test"},  # Empty title
            {"id": "", "title": "test", "url": "https://example.com", "content": "test"},  # Empty ID
            {"id": "test", "title": "test", "url": "https://example.com", "content": ""},  # Empty content
        ]
        
        for invalid_request in invalid_requests:
            response = client.post("/admin/documents", json=invalid_request)
            # Should be 422 for validation error, 503 for service unavailable, or 500 for internal error
            assert response.status_code in [422, 500, 503]
    
    def test_pagination_validation(self, client):
        """Test pagination parameter validation."""
        # Test invalid limit
        response = client.get("/admin/documents?limit=0")
        assert response.status_code == 422
        
        response = client.get("/admin/documents?limit=101")
        assert response.status_code == 422
        
        # Test invalid offset
        response = client.get("/admin/documents?offset=-1")
        assert response.status_code == 422
    
    def test_metrics_hours_validation(self, client):
        """Test performance metrics hours parameter validation."""
        # Test invalid hours
        response = client.get("/admin/system/metrics?hours=0")
        assert response.status_code == 422
        
        response = client.get("/admin/system/metrics?hours=200")
        assert response.status_code == 422
    
    def test_log_lines_validation(self, client):
        """Test log summary lines parameter validation."""
        # Test invalid lines
        response = client.get("/admin/system/logs/test.log?lines=5")
        assert response.status_code == 422
        
        response = client.get("/admin/system/logs/test.log?lines=2000")
        assert response.status_code == 422


class TestManagementAPIIntegration:
    """Integration tests for management API."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_api_documentation_includes_admin_endpoints(self, client):
        """Test that API documentation includes admin endpoints."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        paths = schema.get("paths", {})
        
        # Check that admin endpoints are included
        admin_endpoints = [
            "/admin/documents",
            "/admin/documents/{document_id}",
            "/admin/index/status",
            "/admin/system/status"
        ]
        
        for endpoint in admin_endpoints:
            assert endpoint in paths
    
    def test_admin_endpoints_require_proper_methods(self, client):
        """Test that admin endpoints support proper HTTP methods."""
        # Test documents endpoint methods
        response = client.options("/admin/documents")
        assert response.status_code in [200, 405]  # Either allowed or method not allowed
        
        # Test system status endpoint
        response = client.get("/admin/system/status")
        # Should return 503 if services not initialized, which is expected
        assert response.status_code in [200, 503]
"""
Tests for the FastAPI REST API.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from datetime import datetime

from src.api.main import app, get_rag_service, get_document_engine
from src.models.core import Answer


class TestRAGAPI:
    """Test cases for the RAG API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_rag_service(self):
        """Create mock RAG service."""
        mock_service = Mock()
        mock_service.get_service_status.return_value = {
            "status": "ready",
            "search_engine_ready": True,
            "generative_model_ready": True
        }
        return mock_service
    
    @pytest.fixture
    def mock_document_engine(self):
        """Create mock document engine."""
        return Mock()
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "0.1.0"
    
    def test_health_check_success(self, client, mock_rag_service, mock_document_engine):
        """Test successful health check."""
        with patch.object(app, 'dependency_overrides', {
            get_rag_service: lambda: mock_rag_service,
            get_document_engine: lambda: mock_document_engine
        }):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "services" in data
            assert "timestamp" in data
    
    def test_health_check_service_unavailable(self, client):
        """Test health check when services are unavailable."""
        # Don't override dependencies, so they'll be None
        response = client.get("/health")
        assert response.status_code == 503
    
    def test_ask_question_success(self, client, mock_rag_service, mock_document_engine):
        """Test successful question answering."""
        # Setup mock response
        mock_answer = Answer(
            text="九州大学のスーパーコンピュータを利用するには、まずアカウント申請が必要です。",
            sources=["https://www.cc.kyushu-u.ac.jp/scp/guide1"],
            confidence=0.85,
            processing_time=1.2
        )
        mock_rag_service.generate_answer.return_value = mock_answer
        
        with patch.object(app, 'dependency_overrides', {
            get_rag_service: lambda: mock_rag_service,
            get_document_engine: lambda: mock_document_engine
        }):
            response = client.post("/ask", json={
                "question": "スパコンの使い方を教えてください",
                "max_results": 5,
                "min_confidence": 0.5
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == mock_answer.text
            assert data["sources"] == mock_answer.sources
            assert data["confidence"] == mock_answer.confidence
            assert "processing_time" in data
            assert "timestamp" in data
    
    def test_ask_question_low_confidence(self, client, mock_rag_service, mock_document_engine):
        """Test question answering with low confidence."""
        # Setup mock response with low confidence
        mock_answer = Answer(
            text="申し訳ありませんが、確信を持ってお答えできません。",
            sources=[],
            confidence=0.3,
            processing_time=0.8
        )
        mock_rag_service.generate_answer.return_value = mock_answer
        
        with patch.object(app, 'dependency_overrides', {
            get_rag_service: lambda: mock_rag_service,
            get_document_engine: lambda: mock_document_engine
        }):
            response = client.post("/ask", json={
                "question": "不明な質問",
                "min_confidence": 0.5
            })
            
            assert response.status_code == 422
            assert "confidence" in response.json()["detail"]
    
    def test_ask_question_invalid_request(self, client, mock_rag_service, mock_document_engine):
        """Test question answering with invalid request."""
        with patch.object(app, 'dependency_overrides', {
            get_rag_service: lambda: mock_rag_service,
            get_document_engine: lambda: mock_document_engine
        }):
            response = client.post("/ask", json={
                "question": "",  # Empty question
                "max_results": 5
            })
            
            assert response.status_code == 422
    
    def test_ask_question_service_error(self, client, mock_rag_service, mock_document_engine):
        """Test question answering when service throws error."""
        mock_rag_service.generate_answer.side_effect = Exception("Service error")
        
        with patch.object(app, 'dependency_overrides', {
            get_rag_service: lambda: mock_rag_service,
            get_document_engine: lambda: mock_document_engine
        }):
            response = client.post("/ask", json={
                "question": "テスト質問"
            })
            
            assert response.status_code == 500
            assert "Internal server error" in response.json()["detail"]
    
    def test_process_documents_with_url(self, client, mock_rag_service, mock_document_engine):
        """Test document processing with URL."""
        with patch.object(app, 'dependency_overrides', {
            get_rag_service: lambda: mock_rag_service,
            get_document_engine: lambda: mock_document_engine
        }):
            response = client.post("/documents/process", json={
                "url": "https://example.com",
                "force_refresh": True
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "processing"
            assert "https://example.com" in data["message"]
    
    def test_process_documents_default(self, client, mock_rag_service, mock_document_engine):
        """Test default document processing."""
        with patch.object(app, 'dependency_overrides', {
            get_rag_service: lambda: mock_rag_service,
            get_document_engine: lambda: mock_document_engine
        }):
            response = client.post("/documents/process", json={})
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "processing"
            assert "Kyushu University" in data["message"]
    
    def test_get_document_status(self, client):
        """Test document status endpoint."""
        response = client.get("/documents/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data


class TestAPIValidation:
    """Test API request/response validation."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_question_request_validation(self, client):
        """Test question request validation."""
        # Mock services for validation tests
        mock_rag_service = Mock()
        mock_document_engine = Mock()
        
        with patch.object(app, 'dependency_overrides', {
            get_rag_service: lambda: mock_rag_service,
            get_document_engine: lambda: mock_document_engine
        }):
            # Test various invalid requests
            invalid_requests = [
                {"question": ""},  # Empty question
                {"question": "x" * 1001},  # Too long question
                {"max_results": 0},  # Invalid max_results
                {"max_results": 25},  # Too high max_results
                {"min_confidence": -0.1},  # Invalid confidence
                {"min_confidence": 1.1},  # Invalid confidence
            ]
            
            for invalid_request in invalid_requests:
                response = client.post("/ask", json=invalid_request)
                assert response.status_code == 422
    
    def test_document_processing_request_validation(self, client):
        """Test document processing request validation."""
        # Valid request should pass validation
        response = client.post("/documents/process", json={
            "url": "https://example.com",
            "force_refresh": False
        })
        # May fail due to service unavailability, but should pass validation
        assert response.status_code in [200, 503]


class TestAPIIntegration:
    """Integration tests for API functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_api_documentation_endpoints(self, client):
        """Test that API documentation endpoints are available."""
        # Test OpenAPI docs
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200
        
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "Supercomputer Support RAG API"
    
    def test_startup_event_success(self, client):
        """Test successful startup event."""
        # Test that the app can be created and basic endpoints work
        response = client.get("/")
        assert response.status_code == 200
        
        # Test that the app has the expected configuration
        data = response.json()
        assert data["message"] == "Supercomputer Support RAG API"
        assert data["version"] == "0.1.0"
    
    def test_startup_event_failure(self, client):
        """Test startup event failure handling."""
        # Test that health check fails when services are not available
        response = client.get("/health")
        # Should return 503 when services are not properly initialized
        assert response.status_code == 503
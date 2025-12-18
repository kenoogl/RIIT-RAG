"""
Management API endpoints for the RAG system.

This module provides REST API endpoints for document management,
system monitoring, and administrative operations.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..services.management_service import DocumentManagementService, SystemMonitoringService
from ..services.document_engine import create_document_engine
from ..services.vector_database import LocalVectorDatabase
from ..services.search_logging_service import create_search_logging_service
from ..models.core import Document
from ..models.exceptions import ManagementError

# Create router for management endpoints
router = APIRouter(prefix="/admin", tags=["administration"])

# Global service instances
document_management_service: Optional[DocumentManagementService] = None
system_monitoring_service: Optional[SystemMonitoringService] = None


def get_document_management_service() -> DocumentManagementService:
    """Dependency to get document management service instance."""
    global document_management_service
    if document_management_service is None:
        # Initialize services
        document_engine = create_document_engine()
        vector_db = LocalVectorDatabase()
        document_management_service = DocumentManagementService(
            document_engine=document_engine,
            vector_db=vector_db
        )
    return document_management_service


def get_system_monitoring_service() -> SystemMonitoringService:
    """Dependency to get system monitoring service instance."""
    global system_monitoring_service
    if system_monitoring_service is None:
        search_logging_service = create_search_logging_service()
        system_monitoring_service = SystemMonitoringService(
            search_logging_service=search_logging_service
        )
    return system_monitoring_service


# Pydantic models for API requests/responses
class DocumentCreateRequest(BaseModel):
    """Request model for creating a document."""
    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    url: str = Field(..., description="Document URL")
    content: str = Field(..., min_length=1, description="Document content")
    language: str = Field("ja", description="Document language")
    force_reprocess: bool = Field(False, description="Force reprocessing if document exists")


class DocumentUpdateRequest(BaseModel):
    """Request model for updating a document."""
    title: Optional[str] = Field(None, description="Updated document title")
    url: Optional[str] = Field(None, description="Updated document URL")
    content: Optional[str] = Field(None, description="Updated document content")
    language: Optional[str] = Field(None, description="Updated document language")


class DocumentResponse(BaseModel):
    """Response model for document operations."""
    status: str = Field(..., description="Operation status")
    document_id: str = Field(..., description="Document identifier")
    message: str = Field(..., description="Status message")
    chunks_created: Optional[int] = Field(None, description="Number of chunks created")
    chunks_removed: Optional[int] = Field(None, description="Number of chunks removed")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Operation timestamp")


class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    documents: List[Dict[str, Any]] = Field(..., description="List of documents")
    total_count: int = Field(..., description="Total number of documents")
    returned_count: int = Field(..., description="Number of documents returned")
    offset: int = Field(..., description="Offset used for pagination")
    limit: Optional[int] = Field(None, description="Limit used for pagination")


class IndexStatusResponse(BaseModel):
    """Response model for index status."""
    overview: Dict[str, Any] = Field(..., description="Index overview")
    vector_database: Dict[str, Any] = Field(..., description="Vector database status")
    document_processing: Dict[str, Any] = Field(..., description="Document processing status")
    metadata_file: str = Field(..., description="Path to metadata file")
    documents_directory: str = Field(..., description="Path to documents directory")


class SystemStatusResponse(BaseModel):
    """Response model for system status."""
    timestamp: datetime = Field(..., description="Status timestamp")
    overall_status: str = Field(..., description="Overall system status")
    components: Dict[str, Any] = Field(..., description="Component status")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")
    resources: Dict[str, Any] = Field(..., description="Resource usage")


class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics."""
    period_hours: int = Field(..., description="Analysis period in hours")
    timestamp: datetime = Field(..., description="Metrics timestamp")
    search_metrics: Dict[str, Any] = Field(..., description="Search-related metrics")
    system_metrics: Dict[str, Any] = Field(..., description="System-related metrics")


class LogSummaryResponse(BaseModel):
    """Response model for log summary."""
    status: str = Field(..., description="Summary status")
    file_path: str = Field(..., description="Log file path")
    file_size: Optional[int] = Field(None, description="Log file size in bytes")
    lines_analyzed: Optional[int] = Field(None, description="Number of lines analyzed")
    level_counts: Optional[Dict[str, int]] = Field(None, description="Count of log levels")
    recent_errors: Optional[List[str]] = Field(None, description="Recent error messages")
    last_modified: Optional[datetime] = Field(None, description="Last modification time")


# Document Management Endpoints
@router.post("/documents", response_model=DocumentResponse)
async def create_document(
    request: DocumentCreateRequest,
    background_tasks: BackgroundTasks,
    doc_mgmt: DocumentManagementService = Depends(get_document_management_service)
):
    """
    Create a new document in the system.
    
    This endpoint adds a new document to the RAG system, processes it into chunks,
    and adds it to the vector database for searching.
    """
    try:
        # Create document object
        document = Document(
            id=request.id,
            title=request.title,
            url=request.url,
            content=request.content,
            language=request.language,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Process document in background for large documents
        def process_document():
            try:
                doc_mgmt.add_document(document, force_reprocess=request.force_reprocess)
            except Exception as e:
                # Log error - in production, you'd want better error handling
                import logging
                logging.getLogger(__name__).error(f"Background document processing failed: {e}")
        
        if len(request.content) > 10000:  # Large document
            background_tasks.add_task(process_document)
            return DocumentResponse(
                status="processing",
                document_id=request.id,
                message="Document processing started in background",
                processing_time=0.0
            )
        else:
            # Process immediately for small documents
            result = doc_mgmt.add_document(document, force_reprocess=request.force_reprocess)
            return DocumentResponse(**result)
        
    except ManagementError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    doc_mgmt: DocumentManagementService = Depends(get_document_management_service)
):
    """Get information about a specific document."""
    try:
        document_info = doc_mgmt.get_document_info(document_id)
        return document_info
    except ManagementError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.put("/documents/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: str,
    request: DocumentUpdateRequest,
    doc_mgmt: DocumentManagementService = Depends(get_document_management_service)
):
    """Update an existing document."""
    try:
        # Get existing document info
        existing_doc = doc_mgmt.get_document_info(document_id)
        
        # Create updated document
        document = Document(
            id=document_id,
            title=request.title or existing_doc["title"],
            url=request.url or existing_doc["url"],
            content=request.content or existing_doc.get("content", ""),
            language=request.language or existing_doc["language"],
            created_at=datetime.fromisoformat(existing_doc["created_at"]),
            updated_at=datetime.now()
        )
        
        result = doc_mgmt.update_document(document)
        return DocumentResponse(**result)
        
    except ManagementError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/documents/{document_id}", response_model=DocumentResponse)
async def delete_document(
    document_id: str,
    doc_mgmt: DocumentManagementService = Depends(get_document_management_service)
):
    """Delete a document from the system."""
    try:
        result = doc_mgmt.remove_document(document_id)
        return DocumentResponse(**result)
    except ManagementError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    limit: Optional[int] = Query(None, ge=1, le=100, description="Maximum number of documents to return"),
    offset: int = Query(0, ge=0, description="Number of documents to skip"),
    doc_mgmt: DocumentManagementService = Depends(get_document_management_service)
):
    """List all documents in the system with pagination."""
    try:
        result = doc_mgmt.list_documents(limit=limit, offset=offset)
        return DocumentListResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/index/status", response_model=IndexStatusResponse)
async def get_index_status(
    doc_mgmt: DocumentManagementService = Depends(get_document_management_service)
):
    """Get the current status of document indices."""
    try:
        status = doc_mgmt.get_index_status()
        return IndexStatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/index/rebuild")
async def rebuild_index(
    document_ids: Optional[List[str]] = None,
    background_tasks: BackgroundTasks = None,
    doc_mgmt: DocumentManagementService = Depends(get_document_management_service)
):
    """Rebuild the document index for specified documents or all documents."""
    try:
        def rebuild_task():
            try:
                doc_mgmt.rebuild_index(document_ids)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Index rebuild failed: {e}")
        
        if background_tasks:
            background_tasks.add_task(rebuild_task)
            return {
                "status": "started",
                "message": "Index rebuild started in background",
                "document_count": len(document_ids) if document_ids else "all"
            }
        else:
            result = doc_mgmt.rebuild_index(document_ids)
            return result
            
    except ManagementError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# System Monitoring Endpoints
@router.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status(
    sys_monitor: SystemMonitoringService = Depends(get_system_monitoring_service)
):
    """Get overall system status and health."""
    try:
        status = sys_monitor.get_system_status()
        return SystemStatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/system/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to analyze"),
    sys_monitor: SystemMonitoringService = Depends(get_system_monitoring_service)
):
    """Get performance metrics for the specified time period."""
    try:
        metrics = sys_monitor.get_performance_metrics(hours)
        return PerformanceMetricsResponse(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/system/logs/{log_file}", response_model=LogSummaryResponse)
async def get_log_summary(
    log_file: str,
    lines: int = Query(100, ge=10, le=1000, description="Number of recent lines to analyze"),
    sys_monitor: SystemMonitoringService = Depends(get_system_monitoring_service)
):
    """Get a summary of recent log entries."""
    try:
        summary = sys_monitor.get_log_summary(log_file, lines)
        return LogSummaryResponse(**summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/system/logs")
async def list_log_files(
    sys_monitor: SystemMonitoringService = Depends(get_system_monitoring_service)
):
    """List available log files."""
    try:
        # Get log files from system status
        status = sys_monitor.get_system_status()
        log_files = status.get("components", {}).get("logging", {}).get("log_files", {})
        
        return {
            "log_files": log_files.get("files", []),
            "total_files": log_files.get("accessible_files", 0),
            "total_size": log_files.get("total_size", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
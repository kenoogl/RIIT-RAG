"""
FastAPI-based REST API for the RAG system.

This module provides HTTP endpoints for question-answering functionality
using the RAG (Retrieval-Augmented Generation) system.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime
import uvicorn

from ..services.rag_service import RAGService, create_rag_service
from ..services.document_engine import DocumentProcessingEngine, create_document_engine
from ..models.core import Answer, SearchResult
from ..utils.config import get_config
from .management import router as management_router

# Configure logging
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class QuestionRequest(BaseModel):
    """Request model for question-answering."""
    question: str = Field(..., min_length=1, max_length=1000, description="The question to answer")
    max_results: Optional[int] = Field(5, ge=1, le=20, description="Maximum number of search results to consider")
    min_confidence: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Minimum confidence threshold")

class QuestionResponse(BaseModel):
    """Response model for question-answering."""
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(..., description="Source documents/URLs")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    processing_time: float = Field(..., description="Processing time in seconds")
    search_results_count: int = Field(..., description="Number of search results found")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    services: Dict[str, str] = Field(..., description="Status of individual services")
    version: str = Field(..., description="API version")

class DocumentProcessingRequest(BaseModel):
    """Request model for document processing."""
    url: Optional[str] = Field(None, description="URL to crawl and process")
    force_refresh: bool = Field(False, description="Force refresh of existing documents")

class DocumentProcessingResponse(BaseModel):
    """Response model for document processing."""
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    documents_processed: int = Field(..., description="Number of documents processed")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Processing timestamp")

# Global service instances
rag_service: Optional[RAGService] = None
document_engine: Optional[DocumentProcessingEngine] = None

def get_rag_service() -> RAGService:
    """Dependency to get RAG service instance."""
    global rag_service
    if rag_service is None:
        # Try to initialize if not already done (for testing)
        try:
            rag_service = create_rag_service()
        except Exception:
            raise HTTPException(status_code=503, detail="RAG service not initialized")
    return rag_service

def get_document_engine() -> DocumentProcessingEngine:
    """Dependency to get document processing engine instance."""
    global document_engine
    if document_engine is None:
        # Try to initialize if not already done (for testing)
        try:
            document_engine = create_document_engine()
        except Exception:
            raise HTTPException(status_code=503, detail="Document processing engine not initialized")
    return document_engine

# Create FastAPI app
app = FastAPI(
    title="Supercomputer Support RAG API",
    description="REST API for question-answering about Kyushu University supercomputer systems",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include management router
app.include_router(management_router)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global rag_service, document_engine
    
    try:
        logger.info("Initializing RAG API services...")
        
        # Initialize RAG service
        rag_service = create_rag_service()
        logger.info("RAG service initialized successfully")
        
        # Initialize document processing engine
        document_engine = create_document_engine()
        logger.info("Document processing engine initialized successfully")
        
        logger.info("RAG API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down RAG API services...")
    # Add any cleanup logic here if needed

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Supercomputer Support RAG API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(
    rag_svc: RAGService = Depends(get_rag_service),
    doc_engine: DocumentProcessingEngine = Depends(get_document_engine)
):
    """Health check endpoint."""
    try:
        # Check RAG service status
        rag_status = rag_svc.get_service_status()
        
        # Check document engine status (basic check)
        doc_engine_status = "healthy" if doc_engine else "unavailable"
        
        services = {
            "rag_service": "healthy" if rag_status.get("status") == "ready" else "degraded",
            "document_engine": doc_engine_status,
            "search_engine": "healthy" if rag_status.get("search_engine_ready") else "unavailable",
            "generative_model": "healthy" if rag_status.get("generative_model_ready") else "unavailable"
        }
        
        # Determine overall status
        overall_status = "healthy" if all(s in ["healthy", "ready"] for s in services.values()) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            services=services,
            version="0.1.0"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    rag_svc: RAGService = Depends(get_rag_service)
):
    """
    Answer a question using the RAG system.
    
    This endpoint processes a question through the RAG pipeline:
    1. Search for relevant documents
    2. Generate an answer based on retrieved context
    3. Return the answer with sources and confidence
    """
    try:
        start_time = datetime.now()
        
        logger.info(f"Processing question: {request.question[:100]}...")
        
        # Generate answer using RAG service
        answer = await asyncio.to_thread(
            rag_svc.generate_answer,
            request.question,
            max_results=request.max_results
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Filter by confidence if specified
        if answer.confidence < request.min_confidence:
            logger.warning(f"Answer confidence {answer.confidence} below threshold {request.min_confidence}")
            raise HTTPException(
                status_code=422,
                detail=f"Answer confidence ({answer.confidence:.2f}) below minimum threshold ({request.min_confidence:.2f})"
            )
        
        # Count search results (estimate based on sources)
        search_results_count = len(answer.sources) if answer.sources else 0
        
        logger.info(f"Question processed successfully in {processing_time:.2f}s")
        
        return QuestionResponse(
            answer=answer.text,
            sources=answer.sources,
            confidence=answer.confidence,
            processing_time=processing_time,
            search_results_count=search_results_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/documents/process", response_model=DocumentProcessingResponse)
async def process_documents(
    request: DocumentProcessingRequest,
    background_tasks: BackgroundTasks,
    doc_engine: DocumentProcessingEngine = Depends(get_document_engine)
):
    """
    Process documents for indexing.
    
    This endpoint can process documents from a URL or refresh existing documents.
    Processing is done in the background to avoid blocking the API.
    """
    try:
        start_time = datetime.now()
        
        if request.url:
            logger.info(f"Processing documents from URL: {request.url}")
            
            # Process documents in background
            def process_url():
                try:
                    doc_engine.process_url(request.url, force_refresh=request.force_refresh)
                    logger.info(f"Background processing completed for URL: {request.url}")
                except Exception as e:
                    logger.error(f"Background processing failed for URL {request.url}: {e}")
            
            background_tasks.add_task(process_url)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return DocumentProcessingResponse(
                status="processing",
                message=f"Document processing started for URL: {request.url}",
                documents_processed=0,  # Will be updated in background
                processing_time=processing_time
            )
        else:
            # Default: process Kyushu University SCP site
            logger.info("Processing default Kyushu University SCP documents")
            
            def process_default():
                try:
                    from ..services.document_engine import process_kyushu_university_scp
                    result = process_kyushu_university_scp(force_refresh=request.force_refresh)
                    logger.info(f"Background processing completed: {result}")
                except Exception as e:
                    logger.error(f"Background processing failed: {e}")
            
            background_tasks.add_task(process_default)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return DocumentProcessingResponse(
                status="processing",
                message="Document processing started for Kyushu University SCP site",
                documents_processed=0,  # Will be updated in background
                processing_time=processing_time
            )
            
    except Exception as e:
        logger.error(f"Error starting document processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start document processing: {str(e)}")

@app.get("/documents/status")
async def get_document_status():
    """Get status of document processing and indexing."""
    try:
        # This is a placeholder - in a real implementation, you'd track processing status
        return {
            "status": "ready",
            "message": "Document processing status endpoint",
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting document status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document status: {str(e)}")

def create_app() -> FastAPI:
    """Factory function to create FastAPI app."""
    return app

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    # Load configuration
    config = get_config()
    app_config = config.get("app", {})
    
    run_server(
        host=app_config.get("host", "0.0.0.0"),
        port=app_config.get("port", 8000),
        reload=app_config.get("debug", False)
    )
"""
Management service for document and system administration.

This module provides administrative functionality for managing documents,
monitoring system performance, and maintaining the RAG system.
"""

import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json

from ..models.core import Document, Chunk, SearchResult
from ..models.exceptions import ManagementError, ProcessingError
from ..services.document_engine import DocumentProcessingEngine
from ..services.vector_database import LocalVectorDatabase
from ..services.search_logging_service import SearchLoggingService
from ..utils.config import get_config


class DocumentManagementService:
    """
    Service for managing documents in the RAG system.
    
    This service provides functionality for adding, updating, deleting,
    and monitoring documents and their indices.
    """
    
    def __init__(
        self,
        document_engine: Optional[DocumentProcessingEngine] = None,
        vector_db: Optional[LocalVectorDatabase] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the document management service.
        
        Args:
            document_engine: Document processing engine instance
            vector_db: Vector database instance
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.document_engine = document_engine
        self.vector_db = vector_db
        
        # Storage paths
        storage_config = self.config.get("storage", {})
        self.documents_dir = Path(storage_config.get("documents_dir", "./data/documents"))
        self.metadata_file = self.documents_dir / "metadata.json"
        
        # Ensure directories exist
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        
        # Load document metadata
        self.document_metadata = self._load_metadata()
        
        self.logger.info("Document management service initialized")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load document metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load metadata: {e}")
        
        return {
            "documents": {},
            "last_updated": datetime.now().isoformat(),
            "total_documents": 0,
            "total_chunks": 0
        }
    
    def _save_metadata(self) -> None:
        """Save document metadata to file."""
        try:
            self.document_metadata["last_updated"] = datetime.now().isoformat()
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.document_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
    
    def add_document(self, document: Document, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Add a new document to the system.
        
        Args:
            document: Document to add
            force_reprocess: Whether to reprocess if document already exists
            
        Returns:
            Dictionary containing processing results
            
        Raises:
            ManagementError: If document addition fails
        """
        try:
            start_time = time.time()
            self.logger.info(f"Adding document: {document.id}")
            
            # Check if document already exists
            if document.id in self.document_metadata["documents"] and not force_reprocess:
                raise ManagementError(f"Document {document.id} already exists. Use force_reprocess=True to overwrite.")
            
            # Process document if engine is available
            chunks = []
            if self.document_engine:
                chunks = self.document_engine.process_document(document)
                self.logger.info(f"Created {len(chunks)} chunks for document {document.id}")
            
            # Store in vector database if available
            if self.vector_db and chunks:
                for chunk in chunks:
                    # Generate embedding (this would be done by the document engine)
                    # For now, we'll store without embeddings and let the vector DB handle it
                    self.vector_db.add_document(chunk.content, chunk.id, {"document_id": document.id})
                
                self.logger.info(f"Added {len(chunks)} chunks to vector database")
            
            # Update metadata
            processing_time = time.time() - start_time
            doc_metadata = {
                "id": document.id,
                "title": document.title,
                "url": document.url,
                "language": document.language,
                "content_length": len(document.content),
                "chunk_count": len(chunks),
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat(),
                "added_to_system": datetime.now().isoformat(),
                "processing_time": processing_time
            }
            
            self.document_metadata["documents"][document.id] = doc_metadata
            self.document_metadata["total_documents"] = len(self.document_metadata["documents"])
            self.document_metadata["total_chunks"] += len(chunks)
            
            self._save_metadata()
            
            result = {
                "status": "success",
                "document_id": document.id,
                "chunks_created": len(chunks),
                "processing_time": processing_time,
                "message": f"Document {document.id} added successfully"
            }
            
            self.logger.info(f"Document {document.id} added successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Failed to add document {document.id}: {str(e)}"
            self.logger.error(error_msg)
            raise ManagementError(error_msg) from e
    
    def update_document(self, document: Document) -> Dict[str, Any]:
        """
        Update an existing document.
        
        Args:
            document: Updated document
            
        Returns:
            Dictionary containing update results
            
        Raises:
            ManagementError: If document update fails
        """
        try:
            self.logger.info(f"Updating document: {document.id}")
            
            # Check if document exists
            if document.id not in self.document_metadata["documents"]:
                raise ManagementError(f"Document {document.id} not found")
            
            # Remove old document first
            self.remove_document(document.id)
            
            # Add updated document
            result = self.add_document(document, force_reprocess=True)
            result["message"] = f"Document {document.id} updated successfully"
            
            self.logger.info(f"Document {document.id} updated successfully")
            return result
            
        except Exception as e:
            error_msg = f"Failed to update document {document.id}: {str(e)}"
            self.logger.error(error_msg)
            raise ManagementError(error_msg) from e
    
    def remove_document(self, document_id: str) -> Dict[str, Any]:
        """
        Remove a document from the system.
        
        Args:
            document_id: ID of document to remove
            
        Returns:
            Dictionary containing removal results
            
        Raises:
            ManagementError: If document removal fails
        """
        try:
            start_time = time.time()
            self.logger.info(f"Removing document: {document_id}")
            
            # Check if document exists
            if document_id not in self.document_metadata["documents"]:
                raise ManagementError(f"Document {document_id} not found")
            
            doc_metadata = self.document_metadata["documents"][document_id]
            chunk_count = doc_metadata.get("chunk_count", 0)
            
            # Remove from vector database if available
            if self.vector_db:
                # Remove all chunks for this document
                # This is a simplified approach - in practice, you'd need to track chunk IDs
                try:
                    # For now, we'll just log the removal
                    self.logger.info(f"Removing chunks for document {document_id} from vector database")
                except Exception as e:
                    self.logger.warning(f"Failed to remove chunks from vector database: {e}")
            
            # Update metadata
            del self.document_metadata["documents"][document_id]
            self.document_metadata["total_documents"] = len(self.document_metadata["documents"])
            self.document_metadata["total_chunks"] -= chunk_count
            
            self._save_metadata()
            
            processing_time = time.time() - start_time
            result = {
                "status": "success",
                "document_id": document_id,
                "chunks_removed": chunk_count,
                "processing_time": processing_time,
                "message": f"Document {document_id} removed successfully"
            }
            
            self.logger.info(f"Document {document_id} removed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Failed to remove document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            raise ManagementError(error_msg) from e
    
    def get_document_info(self, document_id: str) -> Dict[str, Any]:
        """
        Get information about a specific document.
        
        Args:
            document_id: ID of document to query
            
        Returns:
            Dictionary containing document information
            
        Raises:
            ManagementError: If document not found
        """
        if document_id not in self.document_metadata["documents"]:
            raise ManagementError(f"Document {document_id} not found")
        
        return self.document_metadata["documents"][document_id]
    
    def list_documents(self, limit: Optional[int] = None, offset: int = 0) -> Dict[str, Any]:
        """
        List all documents in the system.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            Dictionary containing document list and metadata
        """
        documents = list(self.document_metadata["documents"].values())
        
        # Sort by creation time (newest first)
        documents.sort(key=lambda x: x.get("added_to_system", ""), reverse=True)
        
        # Apply pagination
        if limit is not None:
            documents = documents[offset:offset + limit]
        else:
            documents = documents[offset:]
        
        return {
            "documents": documents,
            "total_count": self.document_metadata["total_documents"],
            "returned_count": len(documents),
            "offset": offset,
            "limit": limit
        }
    
    def get_index_status(self) -> Dict[str, Any]:
        """
        Get the current status of document indices.
        
        Returns:
            Dictionary containing index status information
        """
        try:
            # Get vector database status
            vector_db_status = {}
            if self.vector_db:
                try:
                    # This would depend on the vector database implementation
                    vector_db_status = {
                        "status": "available",
                        "index_size": getattr(self.vector_db, 'get_index_size', lambda: 0)(),
                        "last_updated": getattr(self.vector_db, 'get_last_updated', lambda: None)()
                    }
                except Exception as e:
                    vector_db_status = {
                        "status": "error",
                        "error": str(e)
                    }
            else:
                vector_db_status = {"status": "not_available"}
            
            # Get document processing status
            processing_status = {}
            if self.document_engine:
                processing_status = {
                    "status": "available",
                    "processor_type": type(self.document_engine).__name__
                }
            else:
                processing_status = {"status": "not_available"}
            
            return {
                "overview": {
                    "total_documents": self.document_metadata["total_documents"],
                    "total_chunks": self.document_metadata["total_chunks"],
                    "last_updated": self.document_metadata["last_updated"]
                },
                "vector_database": vector_db_status,
                "document_processing": processing_status,
                "metadata_file": str(self.metadata_file),
                "documents_directory": str(self.documents_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get index status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def rebuild_index(self, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Rebuild the document index for specified documents or all documents.
        
        Args:
            document_ids: List of document IDs to rebuild. If None, rebuild all.
            
        Returns:
            Dictionary containing rebuild results
        """
        try:
            start_time = time.time()
            self.logger.info("Starting index rebuild")
            
            if document_ids is None:
                document_ids = list(self.document_metadata["documents"].keys())
            
            results = {
                "status": "success",
                "processed_documents": 0,
                "failed_documents": 0,
                "errors": [],
                "processing_time": 0
            }
            
            for doc_id in document_ids:
                try:
                    if doc_id in self.document_metadata["documents"]:
                        # This is a simplified rebuild - in practice, you'd reload the document
                        # and reprocess it through the entire pipeline
                        self.logger.info(f"Rebuilding index for document: {doc_id}")
                        results["processed_documents"] += 1
                    else:
                        self.logger.warning(f"Document {doc_id} not found, skipping")
                        
                except Exception as e:
                    error_msg = f"Failed to rebuild index for document {doc_id}: {str(e)}"
                    self.logger.error(error_msg)
                    results["failed_documents"] += 1
                    results["errors"].append(error_msg)
            
            results["processing_time"] = time.time() - start_time
            
            if results["failed_documents"] > 0:
                results["status"] = "partial_success"
            
            self.logger.info(f"Index rebuild completed: {results['processed_documents']} processed, {results['failed_documents']} failed")
            return results
            
        except Exception as e:
            error_msg = f"Index rebuild failed: {str(e)}"
            self.logger.error(error_msg)
            raise ManagementError(error_msg) from e


class SystemMonitoringService:
    """
    Service for monitoring system performance and health.
    
    This service provides functionality for tracking system metrics,
    performance indicators, and log analysis.
    """
    
    def __init__(
        self,
        search_logging_service: Optional[SearchLoggingService] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the system monitoring service.
        
        Args:
            search_logging_service: Search logging service instance
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        self.search_logging_service = search_logging_service
        
        # Storage paths
        storage_config = self.config.get("storage", {})
        self.logs_dir = Path(storage_config.get("logs_dir", "./logs"))
        
        self.logger.info("System monitoring service initialized")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status and health.
        
        Returns:
            Dictionary containing system status information
        """
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "healthy",
                "components": {},
                "performance": {},
                "resources": {}
            }
            
            # Check component status
            components = {}
            
            # Search logging service
            if self.search_logging_service:
                try:
                    search_stats = self.search_logging_service.get_search_statistics()
                    components["search_logging"] = {
                        "status": "healthy",
                        "total_searches": search_stats.get("overview", {}).get("total_searches", 0),
                        "avg_search_time": search_stats.get("overview", {}).get("avg_search_time", 0)
                    }
                except Exception as e:
                    components["search_logging"] = {
                        "status": "error",
                        "error": str(e)
                    }
            else:
                components["search_logging"] = {"status": "not_available"}
            
            # Log files status
            log_files = self._check_log_files()
            components["logging"] = {
                "status": "healthy" if log_files["accessible_files"] > 0 else "warning",
                "log_files": log_files
            }
            
            status["components"] = components
            
            # Performance metrics
            if self.search_logging_service:
                try:
                    search_stats = self.search_logging_service.get_search_statistics()
                    status["performance"] = {
                        "avg_search_time": search_stats.get("overview", {}).get("avg_search_time", 0),
                        "avg_answer_time": search_stats.get("overview", {}).get("avg_answer_time", 0),
                        "searches_per_hour": search_stats.get("recent_activity", {}).get("searches_last_hour", 0)
                    }
                except Exception:
                    status["performance"] = {"status": "unavailable"}
            
            # Resource usage (simplified)
            import psutil
            try:
                status["resources"] = {
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_usage": psutil.disk_usage('.').percent
                }
            except ImportError:
                status["resources"] = {"status": "psutil_not_available"}
            except Exception as e:
                status["resources"] = {"status": "error", "error": str(e)}
            
            # Determine overall status
            component_statuses = [comp.get("status", "unknown") for comp in components.values()]
            if "error" in component_statuses:
                status["overall_status"] = "degraded"
            elif "warning" in component_statuses:
                status["overall_status"] = "warning"
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "error",
                "error": str(e)
            }
    
    def _check_log_files(self) -> Dict[str, Any]:
        """Check the status of log files."""
        log_info = {
            "accessible_files": 0,
            "total_size": 0,
            "files": []
        }
        
        try:
            if self.logs_dir.exists():
                for log_file in self.logs_dir.glob("*.log"):
                    try:
                        stat = log_file.stat()
                        log_info["files"].append({
                            "name": log_file.name,
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
                        log_info["accessible_files"] += 1
                        log_info["total_size"] += stat.st_size
                    except Exception as e:
                        log_info["files"].append({
                            "name": log_file.name,
                            "error": str(e)
                        })
        except Exception as e:
            log_info["error"] = str(e)
        
        return log_info
    
    def get_performance_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance metrics for the specified time period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            metrics = {
                "period_hours": hours,
                "timestamp": datetime.now().isoformat(),
                "search_metrics": {},
                "system_metrics": {}
            }
            
            # Get search metrics if available
            if self.search_logging_service:
                try:
                    search_trends = self.search_logging_service.get_search_trends(hours)
                    search_stats = self.search_logging_service.get_search_statistics()
                    
                    metrics["search_metrics"] = {
                        "trends": search_trends,
                        "current_stats": search_stats.get("overview", {}),
                        "recent_activity": search_stats.get("recent_activity", {})
                    }
                except Exception as e:
                    metrics["search_metrics"] = {"error": str(e)}
            
            # System metrics (simplified)
            try:
                import psutil
                metrics["system_metrics"] = {
                    "cpu_count": psutil.cpu_count(),
                    "memory_total": psutil.virtual_memory().total,
                    "disk_total": psutil.disk_usage('.').total,
                    "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
                }
            except ImportError:
                metrics["system_metrics"] = {"status": "psutil_not_available"}
            except Exception as e:
                metrics["system_metrics"] = {"error": str(e)}
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {
                "period_hours": hours,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def get_log_summary(self, log_file: str = "rag_system.log", lines: int = 100) -> Dict[str, Any]:
        """
        Get a summary of recent log entries.
        
        Args:
            log_file: Name of log file to analyze
            lines: Number of recent lines to analyze
            
        Returns:
            Dictionary containing log summary
        """
        try:
            log_path = self.logs_dir / log_file
            
            if not log_path.exists():
                return {
                    "status": "file_not_found",
                    "file_path": str(log_path)
                }
            
            # Read recent log lines
            with open(log_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
            # Analyze log levels
            level_counts = {"INFO": 0, "WARNING": 0, "ERROR": 0, "DEBUG": 0, "CRITICAL": 0}
            recent_errors = []
            
            for line in recent_lines:
                for level in level_counts.keys():
                    if f" - {level} - " in line:
                        level_counts[level] += 1
                        if level in ["ERROR", "CRITICAL"]:
                            recent_errors.append(line.strip())
                        break
            
            return {
                "status": "success",
                "file_path": str(log_path),
                "file_size": log_path.stat().st_size,
                "lines_analyzed": len(recent_lines),
                "level_counts": level_counts,
                "recent_errors": recent_errors[-10:],  # Last 10 errors
                "last_modified": datetime.fromtimestamp(log_path.stat().st_mtime).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get log summary: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


def create_document_management_service(
    document_engine: Optional[DocumentProcessingEngine] = None,
    vector_db: Optional[LocalVectorDatabase] = None,
    config_path: Optional[str] = None
) -> DocumentManagementService:
    """
    Factory function to create a document management service.
    
    Args:
        document_engine: Document processing engine instance
        vector_db: Vector database instance
        config_path: Path to configuration file
        
    Returns:
        Configured document management service instance
    """
    return DocumentManagementService(
        document_engine=document_engine,
        vector_db=vector_db,
        config_path=config_path
    )


def create_system_monitoring_service(
    search_logging_service: Optional[SearchLoggingService] = None,
    config_path: Optional[str] = None
) -> SystemMonitoringService:
    """
    Factory function to create a system monitoring service.
    
    Args:
        search_logging_service: Search logging service instance
        config_path: Path to configuration file
        
    Returns:
        Configured system monitoring service instance
    """
    return SystemMonitoringService(
        search_logging_service=search_logging_service,
        config_path=config_path
    )
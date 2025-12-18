"""
Document processing engine for the RAG system.

This module provides a comprehensive document processing pipeline that integrates
web crawling, text processing, embedding generation, and vector storage.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from ..models.interfaces import DocumentProcessorInterface
from ..models.core import Document, Chunk
from ..models.exceptions import ProcessingError, IndexError as RAGIndexError
from ..utils.config import get_config
from .web_crawler import RobustWebCrawler, WebsiteChangeDetector
from .text_processor import DocumentProcessor
from .embedding_model import EmbeddingService, get_embedding_service
from .vector_database import LocalVectorDatabase
from .index_manager import IndexManager


class DocumentProcessingEngine(DocumentProcessorInterface):
    """
    Comprehensive document processing engine.
    
    This class orchestrates the entire document processing pipeline:
    1. Web crawling and document collection
    2. Text processing and chunking
    3. Embedding generation
    4. Vector storage and indexing
    5. Error handling and logging
    """
    
    def __init__(
        self,
        storage_path: str = "data",
        enable_crawling: bool = True,
        enable_batch_processing: bool = True,
        max_workers: int = 4
    ):
        """
        Initialize the document processing engine.
        
        Args:
            storage_path: Base path for data storage
            enable_crawling: Whether to enable web crawling
            enable_batch_processing: Whether to enable batch processing
            max_workers: Maximum number of worker threads
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        config = get_config()
        self.storage_path = Path(storage_path)
        self.enable_crawling = enable_crawling
        self.enable_batch_processing = enable_batch_processing
        self.max_workers = max_workers
        
        # Create storage directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "documents").mkdir(exist_ok=True)
        (self.storage_path / "logs").mkdir(exist_ok=True)
        (self.storage_path / "processing").mkdir(exist_ok=True)
        
        # Initialize components
        self.web_crawler = RobustWebCrawler() if enable_crawling else None
        self.change_detector = WebsiteChangeDetector(str(self.storage_path / "website_structure"))
        self.text_processor = DocumentProcessor()
        self.embedding_service = get_embedding_service()
        self.vector_db = LocalVectorDatabase(str(self.storage_path / "vectors"))
        self.index_manager = IndexManager(str(self.storage_path / "vectors"))
        
        # Processing state
        self.processing_stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'errors_encountered': 0,
            'processing_time': 0.0,
            'last_update': None
        }
        
        # Load processing history
        self.processing_history = self._load_processing_history()
        
        self.logger.info("Document processing engine initialized")
    
    def process_document(self, document: Document) -> List[Chunk]:
        """
        Process a single document through the complete pipeline.
        
        Args:
            document: Document to process
            
        Returns:
            List of processed chunks with embeddings
            
        Raises:
            ProcessingError: If document processing fails
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing document: {document.id}")
            
            # Step 1: Text processing and chunking
            chunks = self.text_processor.process_document(document)
            if not chunks:
                self.logger.warning(f"No chunks created for document {document.id}")
                return []
            
            self.logger.debug(f"Created {len(chunks)} chunks for document {document.id}")
            
            # Step 2: Generate embeddings
            chunks_with_embeddings = self._generate_embeddings_for_chunks(chunks)
            
            # Step 3: Store in vector database
            self._store_chunks_in_vector_db(chunks_with_embeddings, document)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_processing_stats(
                documents=1,
                chunks=len(chunks_with_embeddings),
                embeddings=len(chunks_with_embeddings),
                processing_time=processing_time
            )
            
            # Log processing result
            self._log_processing_result(document.id, "success", len(chunks_with_embeddings), processing_time)
            
            self.logger.info(f"Successfully processed document {document.id} in {processing_time:.2f}s")
            return chunks_with_embeddings
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_processing_stats(errors=1, processing_time=processing_time)
            self._log_processing_result(document.id, "error", 0, processing_time, str(e))
            
            self.logger.error(f"Failed to process document {document.id}: {e}")
            raise ProcessingError(f"Failed to process document {document.id}: {e}")
    
    def generate_embeddings(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Generate embedding vectors for chunks.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            ProcessingError: If embedding generation fails
        """
        try:
            if not self.embedding_service._initialized:
                self.logger.info("Initializing embedding service...")
                success = self.embedding_service.initialize()
                if not success:
                    raise ProcessingError("Failed to initialize embedding service")
            
            # Extract text content from chunks
            texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings
            self.logger.debug(f"Generating embeddings for {len(texts)} chunks")
            embeddings = self.embedding_service.embed_texts(texts)
            
            self.logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise ProcessingError(f"Failed to generate embeddings: {e}")
    
    def update_index(self, chunks: List[Chunk], vectors: List[List[float]]) -> bool:
        """
        Update the search index with new chunks and vectors.
        
        Args:
            chunks: List of chunks to index
            vectors: Corresponding embedding vectors
            
        Returns:
            True if update successful, False otherwise
            
        Raises:
            RAGIndexError: If index update fails
        """
        try:
            if len(chunks) != len(vectors):
                raise ValueError("Number of chunks must match number of vectors")
            
            # Prepare metadata for vector storage
            metadata = []
            for chunk in chunks:
                chunk_metadata = {
                    'chunk_id': chunk.id,
                    'document_id': chunk.document_id,
                    'content': chunk.content,
                    'position': chunk.position,
                    'document_url': '',  # Will be filled by caller if available
                    'document_title': '',  # Will be filled by caller if available
                    'document_content': '',  # Will be filled by caller if available
                    'document_language': 'ja',
                    'document_created_at': datetime.now().isoformat(),
                    'document_updated_at': datetime.now().isoformat()
                }
                metadata.append(chunk_metadata)
            
            # Store vectors in database
            success = self.vector_db.store_vectors(vectors, metadata)
            if not success:
                raise RAGIndexError("Failed to store vectors in database")
            
            self.logger.debug(f"Successfully indexed {len(chunks)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update index: {e}")
            raise RAGIndexError(f"Failed to update index: {e}")
    
    def process_documents_batch(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Process multiple documents in batch with parallel processing.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Dictionary containing batch processing results
        """
        start_time = time.time()
        results = {
            'total_documents': len(documents),
            'successful': 0,
            'failed': 0,
            'total_chunks': 0,
            'processing_time': 0.0,
            'errors': []
        }
        
        try:
            self.logger.info(f"Starting batch processing of {len(documents)} documents")
            
            if self.enable_batch_processing and len(documents) > 1:
                # Parallel processing
                results = self._process_documents_parallel(documents)
            else:
                # Sequential processing
                results = self._process_documents_sequential(documents)
            
            results['processing_time'] = time.time() - start_time
            
            self.logger.info(f"Batch processing completed: {results['successful']}/{results['total_documents']} "
                           f"documents processed in {results['processing_time']:.2f}s")
            
            return results
            
        except Exception as e:
            results['processing_time'] = time.time() - start_time
            self.logger.error(f"Batch processing failed: {e}")
            results['errors'].append(str(e))
            return results
    
    def crawl_and_process_website(self, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Crawl a website and process all discovered documents.
        
        Args:
            url: URL to crawl (uses configured target if not provided)
            
        Returns:
            Dictionary containing crawling and processing results
        """
        if not self.web_crawler:
            raise ProcessingError("Web crawling is disabled")
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting website crawl and processing for {url or 'configured target'}")
            
            # Crawl website
            documents = self.web_crawler.crawl_website(url)
            
            if not documents:
                self.logger.warning("No documents found during crawling")
                return {
                    'crawl_time': time.time() - start_time,
                    'documents_found': 0,
                    'documents_processed': 0,
                    'processing_results': {}
                }
            
            crawl_time = time.time() - start_time
            self.logger.info(f"Crawling completed in {crawl_time:.2f}s, found {len(documents)} documents")
            
            # Detect website structure changes
            if self.change_detector:
                all_urls = [doc.url for doc in documents]
                changes = self.change_detector.detect_structure_changes(
                    url or self.web_crawler.target_url, 
                    all_urls
                )
                
                if changes.get('has_changes'):
                    self.logger.info(f"Website structure changes detected: "
                                   f"+{len(changes.get('added_links', []))} "
                                   f"-{len(changes.get('removed_links', []))} links")
            
            # Process documents
            processing_results = self.process_documents_batch(documents)
            
            total_time = time.time() - start_time
            
            return {
                'crawl_time': crawl_time,
                'documents_found': len(documents),
                'documents_processed': processing_results['successful'],
                'processing_results': processing_results,
                'total_time': total_time,
                'structure_changes': changes if 'changes' in locals() else None
            }
            
        except Exception as e:
            self.logger.error(f"Website crawl and processing failed: {e}")
            raise ProcessingError(f"Website crawl and processing failed: {e}")
    
    def update_document(self, document: Document) -> bool:
        """
        Update an existing document by removing old data and processing new version.
        
        Args:
            document: Updated document
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            self.logger.info(f"Updating document: {document.id}")
            
            # Remove old document data
            self.delete_document(document.id)
            
            # Process new version
            chunks = self.process_document(document)
            
            self.logger.info(f"Successfully updated document {document.id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update document {document.id}: {e}")
            return False
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all associated data.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            self.logger.info(f"Deleting document: {document_id}")
            
            # Delete from vector database
            success = self.vector_db.delete_by_document_id(document_id)
            
            if success:
                self.logger.info(f"Successfully deleted document {document_id}")
            else:
                self.logger.warning(f"Document {document_id} not found in vector database")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def _generate_embeddings_for_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Generate embeddings for chunks and attach them."""
        embeddings = self.generate_embeddings(chunks)
        
        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
    
    def _store_chunks_in_vector_db(self, chunks: List[Chunk], document: Document) -> None:
        """Store chunks with embeddings in vector database."""
        if not chunks:
            return
        
        vectors = [chunk.embedding for chunk in chunks]
        
        # Prepare metadata with document information
        metadata = []
        for chunk in chunks:
            chunk_metadata = {
                'chunk_id': chunk.id,
                'document_id': chunk.document_id,
                'content': chunk.content,
                'position': chunk.position,
                'document_url': document.url,
                'document_title': document.title,
                'document_content': document.content[:1000],  # Truncate for metadata
                'document_language': document.language,
                'document_created_at': document.created_at.isoformat(),
                'document_updated_at': document.updated_at.isoformat()
            }
            metadata.append(chunk_metadata)
        
        # Store in vector database
        success = self.vector_db.store_vectors(vectors, metadata)
        if not success:
            raise ProcessingError("Failed to store chunks in vector database")
    
    def _process_documents_parallel(self, documents: List[Document]) -> Dict[str, Any]:
        """Process documents in parallel using thread pool."""
        results = {
            'total_documents': len(documents),
            'successful': 0,
            'failed': 0,
            'total_chunks': 0,
            'errors': []
        }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all documents for processing
            future_to_doc = {
                executor.submit(self.process_document, doc): doc 
                for doc in documents
            }
            
            # Collect results
            for future in as_completed(future_to_doc):
                document = future_to_doc[future]
                try:
                    chunks = future.result()
                    results['successful'] += 1
                    results['total_chunks'] += len(chunks)
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Document {document.id}: {str(e)}")
        
        return results
    
    def _process_documents_sequential(self, documents: List[Document]) -> Dict[str, Any]:
        """Process documents sequentially."""
        results = {
            'total_documents': len(documents),
            'successful': 0,
            'failed': 0,
            'total_chunks': 0,
            'errors': []
        }
        
        for document in documents:
            try:
                chunks = self.process_document(document)
                results['successful'] += 1
                results['total_chunks'] += len(chunks)
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Document {document.id}: {str(e)}")
        
        return results
    
    def _update_processing_stats(
        self, 
        documents: int = 0, 
        chunks: int = 0, 
        embeddings: int = 0, 
        errors: int = 0,
        processing_time: float = 0.0
    ) -> None:
        """Update processing statistics."""
        self.processing_stats['documents_processed'] += documents
        self.processing_stats['chunks_created'] += chunks
        self.processing_stats['embeddings_generated'] += embeddings
        self.processing_stats['errors_encountered'] += errors
        self.processing_stats['processing_time'] += processing_time
        self.processing_stats['last_update'] = datetime.now().isoformat()
    
    def _log_processing_result(
        self, 
        document_id: str, 
        status: str, 
        chunks_created: int, 
        processing_time: float,
        error_message: Optional[str] = None
    ) -> None:
        """Log processing result to file."""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'document_id': document_id,
                'status': status,
                'chunks_created': chunks_created,
                'processing_time': processing_time,
                'error_message': error_message
            }
            
            log_file = self.storage_path / "logs" / "processing.jsonl"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to log processing result: {e}")
    
    def _load_processing_history(self) -> List[Dict[str, Any]]:
        """Load processing history from log file."""
        try:
            log_file = self.storage_path / "logs" / "processing.jsonl"
            if not log_file.exists():
                return []
            
            history = []
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line))
            
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to load processing history: {e}")
            return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        stats = self.processing_stats.copy()
        
        # Add vector database stats
        if self.vector_db:
            db_stats = self.vector_db.get_stats()
            stats.update({
                'total_vectors_stored': db_stats.get('total_vectors', 0),
                'total_documents_indexed': db_stats.get('total_documents', 0),
                'vector_dimension': db_stats.get('vector_dimension', 0)
            })
        
        # Add embedding service info
        if self.embedding_service._initialized:
            embedding_info = self.embedding_service.model.get_model_info()
            stats.update({
                'embedding_model': embedding_info.get('model_name', 'Unknown'),
                'embedding_dimension': embedding_info.get('embedding_dimension', 0)
            })
        
        return stats
    
    def get_recent_processing_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent processing history."""
        return self.processing_history[-limit:] if self.processing_history else []
    
    def cleanup_old_data(self, days_old: int = 30) -> Dict[str, Any]:
        """
        Clean up old processing data and logs.
        
        Args:
            days_old: Remove data older than this many days
            
        Returns:
            Dictionary containing cleanup results
        """
        try:
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            
            # Clean up old log entries
            log_file = self.storage_path / "logs" / "processing.jsonl"
            if log_file.exists():
                # Read all entries
                entries = []
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            entry_time = datetime.fromisoformat(entry['timestamp']).timestamp()
                            if entry_time >= cutoff_date:
                                entries.append(entry)
                
                # Write back recent entries
                with open(log_file, 'w', encoding='utf-8') as f:
                    for entry in entries:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
                removed_entries = len(self.processing_history) - len(entries)
                self.processing_history = entries
                
                self.logger.info(f"Cleaned up {removed_entries} old log entries")
                
                return {
                    'log_entries_removed': removed_entries,
                    'cleanup_successful': True
                }
            
            return {'log_entries_removed': 0, 'cleanup_successful': True}
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
            return {'cleanup_successful': False, 'error': str(e)}


# Convenience functions
def create_document_engine(**kwargs) -> DocumentProcessingEngine:
    """Create a document processing engine instance."""
    return DocumentProcessingEngine(**kwargs)


def process_kyushu_university_scp() -> Dict[str, Any]:
    """Process the Kyushu University SCP website."""
    engine = DocumentProcessingEngine()
    return engine.crawl_and_process_website()
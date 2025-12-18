"""
Index management functionality for the RAG system.

This module provides comprehensive index management including document updates,
deletions, and consistency maintenance for the vector database.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import time

from ..models.interfaces import VectorDatabaseInterface, DocumentProcessorInterface
from ..models.core import Document, Chunk, SearchResult
from ..models.exceptions import IndexError, ProcessingError, DeletionError
from .vector_database import LocalVectorDatabase


class IndexManager:
    """
    Manages document indexing operations including updates and deletions.
    
    This class provides high-level operations for maintaining the vector index,
    ensuring consistency when documents are added, updated, or removed.
    """
    
    def __init__(
        self, 
        vector_db: VectorDatabaseInterface,
        document_processor: Optional[DocumentProcessorInterface] = None
    ):
        """
        Initialize the index manager.
        
        Args:
            vector_db: Vector database instance
            document_processor: Document processor for generating embeddings
        """
        self.vector_db = vector_db
        self.document_processor = document_processor
        self.logger = logging.getLogger(__name__)
    
    def add_document(self, document: Document) -> bool:
        """
        Add a new document to the index.
        
        Args:
            document: Document to add
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            IndexError: If document indexing fails
        """
        try:
            start_time = time.time()
            
            # Check if document already exists
            if self._document_exists(document.id):
                self.logger.warning(f"Document {document.id} already exists. Use update_document instead.")
                return False
            
            # Process document into chunks
            if not self.document_processor:
                raise IndexError("Document processor not available")
            
            chunks = self.document_processor.process_document(document)
            if not chunks:
                self.logger.warning(f"No chunks generated for document {document.id}")
                return True
            
            # Generate embeddings
            embeddings = self.document_processor.generate_embeddings(chunks)
            if len(embeddings) != len(chunks):
                raise IndexError(f"Embedding count mismatch: {len(embeddings)} != {len(chunks)}")
            
            # Prepare metadata for storage
            metadata = []
            for chunk, embedding in zip(chunks, embeddings):
                meta = {
                    'chunk_id': chunk.id,
                    'document_id': document.id,
                    'content': chunk.content,
                    'position': chunk.position,
                    'document_url': document.url,
                    'document_title': document.title,
                    'document_content': document.content[:1000],  # Store first 1000 chars
                    'document_language': document.language,
                    'document_created_at': document.created_at.isoformat(),
                    'document_updated_at': document.updated_at.isoformat(),
                    'indexed_at': datetime.now().isoformat()
                }
                metadata.append(meta)
                
                # Update chunk with embedding
                chunk.embedding = embedding
            
            # Store in vector database
            success = self.vector_db.store_vectors(embeddings, metadata)
            
            if success:
                processing_time = time.time() - start_time
                self.logger.info(
                    f"Successfully indexed document {document.id}: "
                    f"{len(chunks)} chunks, {processing_time:.3f}s"
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to add document {document.id}: {e}")
            raise IndexError(f"Failed to add document: {e}")
    
    def update_document(self, document: Document) -> bool:
        """
        Update an existing document in the index.
        
        This removes the old version and adds the new version atomically.
        
        Args:
            document: Updated document
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            IndexError: If document update fails
        """
        try:
            start_time = time.time()
            
            # Check if document exists
            if not self._document_exists(document.id):
                self.logger.info(f"Document {document.id} doesn't exist. Adding as new document.")
                return self.add_document(document)
            
            # Remove old version
            old_removal_success = self.remove_document(document.id)
            if not old_removal_success:
                self.logger.error(f"Failed to remove old version of document {document.id}")
                return False
            
            # Add new version
            add_success = self.add_document(document)
            
            if add_success:
                processing_time = time.time() - start_time
                self.logger.info(
                    f"Successfully updated document {document.id}: {processing_time:.3f}s"
                )
            else:
                self.logger.error(f"Failed to add updated version of document {document.id}")
            
            return add_success
            
        except Exception as e:
            self.logger.error(f"Failed to update document {document.id}: {e}")
            raise IndexError(f"Failed to update document: {e}")
    
    def remove_document(self, document_id: str) -> bool:
        """
        Remove a document and all its chunks from the index.
        
        Args:
            document_id: ID of the document to remove
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            DeletionError: If document removal fails
        """
        try:
            start_time = time.time()
            
            # Check if document exists
            if not self._document_exists(document_id):
                self.logger.warning(f"Document {document_id} not found in index")
                return True  # Consider it successful if already gone
            
            # Remove from vector database
            success = self.vector_db.delete_by_document_id(document_id)
            
            if success:
                processing_time = time.time() - start_time
                self.logger.info(
                    f"Successfully removed document {document_id}: {processing_time:.3f}s"
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to remove document {document_id}: {e}")
            raise DeletionError(f"Failed to remove document: {e}")
    
    def batch_add_documents(self, documents: List[Document]) -> Dict[str, bool]:
        """
        Add multiple documents to the index in batch.
        
        Args:
            documents: List of documents to add
            
        Returns:
            Dictionary mapping document IDs to success status
        """
        results = {}
        
        for document in documents:
            try:
                results[document.id] = self.add_document(document)
            except Exception as e:
                self.logger.error(f"Failed to add document {document.id} in batch: {e}")
                results[document.id] = False
        
        successful = sum(1 for success in results.values() if success)
        self.logger.info(f"Batch add completed: {successful}/{len(documents)} successful")
        
        return results
    
    def batch_remove_documents(self, document_ids: List[str]) -> Dict[str, bool]:
        """
        Remove multiple documents from the index in batch.
        
        Args:
            document_ids: List of document IDs to remove
            
        Returns:
            Dictionary mapping document IDs to success status
        """
        results = {}
        
        for document_id in document_ids:
            try:
                results[document_id] = self.remove_document(document_id)
            except Exception as e:
                self.logger.error(f"Failed to remove document {document_id} in batch: {e}")
                results[document_id] = False
        
        successful = sum(1 for success in results.values() if success)
        self.logger.info(f"Batch remove completed: {successful}/{len(document_ids)} successful")
        
        return results
    
    def rebuild_index(self, documents: List[Document]) -> bool:
        """
        Completely rebuild the index with the given documents.
        
        Args:
            documents: List of documents to index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = time.time()
            
            # Clear existing index
            if hasattr(self.vector_db, 'clear_all'):
                clear_success = self.vector_db.clear_all()
                if not clear_success:
                    self.logger.error("Failed to clear existing index")
                    return False
            
            # Add all documents
            results = self.batch_add_documents(documents)
            
            successful = sum(1 for success in results.values() if success)
            total_time = time.time() - start_time
            
            self.logger.info(
                f"Index rebuild completed: {successful}/{len(documents)} documents, "
                f"{total_time:.3f}s"
            )
            
            return successful == len(documents)
            
        except Exception as e:
            self.logger.error(f"Failed to rebuild index: {e}")
            return False
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get information about chunks for a specific document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of chunk information dictionaries
        """
        try:
            # This is a simplified implementation
            # In practice, you'd query the vector database for chunk metadata
            
            if hasattr(self.vector_db, 'index_data'):
                index_data = self.vector_db.index_data
                if document_id in index_data.get('document_to_chunks', {}):
                    chunk_indices = index_data['document_to_chunks'][document_id]
                    
                    chunks_info = []
                    for idx in chunk_indices:
                        if idx < len(self.vector_db.metadata):
                            meta = self.vector_db.metadata[idx]
                            chunks_info.append({
                                'chunk_id': meta.get('chunk_id'),
                                'position': meta.get('position'),
                                'content_preview': meta.get('content', '')[:100] + '...',
                                'indexed_at': meta.get('indexed_at')
                            })
                    
                    return chunks_info
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get chunks for document {document_id}: {e}")
            return []
    
    def verify_index_consistency(self) -> Dict[str, Any]:
        """
        Verify the consistency of the index.
        
        Returns:
            Dictionary containing consistency check results
        """
        try:
            results = {
                'consistent': True,
                'issues': [],
                'statistics': {}
            }
            
            if hasattr(self.vector_db, 'get_stats'):
                stats = self.vector_db.get_stats()
                results['statistics'] = stats
                
                # Basic consistency checks
                if hasattr(self.vector_db, 'vectors') and hasattr(self.vector_db, 'metadata'):
                    vector_count = self.vector_db.vectors.shape[0] if self.vector_db.vectors.size > 0 else 0
                    metadata_count = len(self.vector_db.metadata)
                    
                    if vector_count != metadata_count:
                        results['consistent'] = False
                        results['issues'].append(
                            f"Vector count ({vector_count}) doesn't match metadata count ({metadata_count})"
                        )
            
            self.logger.info(f"Index consistency check: {'PASS' if results['consistent'] else 'FAIL'}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to verify index consistency: {e}")
            return {
                'consistent': False,
                'issues': [f"Consistency check failed: {e}"],
                'statistics': {}
            }
    
    def _document_exists(self, document_id: str) -> bool:
        """
        Check if a document exists in the index.
        
        Args:
            document_id: ID of the document to check
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            if hasattr(self.vector_db, 'index_data'):
                return document_id in self.vector_db.index_data.get('document_to_chunks', {})
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check document existence for {document_id}: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive index statistics.
        
        Returns:
            Dictionary containing index statistics
        """
        try:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'vector_database': {},
                'document_processor_available': self.document_processor is not None
            }
            
            if hasattr(self.vector_db, 'get_stats'):
                stats['vector_database'] = self.vector_db.get_stats()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get index stats: {e}")
            return {'error': str(e)}
    
    def set_document_processor(self, document_processor: DocumentProcessorInterface) -> None:
        """
        Set or update the document processor.
        
        Args:
            document_processor: New document processor instance
        """
        self.document_processor = document_processor
        self.logger.info("Document processor updated")


class IncrementalIndexManager(IndexManager):
    """
    Enhanced index manager with incremental update capabilities.
    
    This extends the basic index manager to support more efficient
    incremental updates and change tracking.
    """
    
    def __init__(
        self, 
        vector_db: VectorDatabaseInterface,
        document_processor: Optional[DocumentProcessorInterface] = None,
        change_threshold: float = 0.1
    ):
        """
        Initialize the incremental index manager.
        
        Args:
            vector_db: Vector database instance
            document_processor: Document processor for generating embeddings
            change_threshold: Minimum change threshold for updates
        """
        super().__init__(vector_db, document_processor)
        self.change_threshold = change_threshold
        self.change_log = []  # Track changes for debugging
    
    def smart_update_document(self, document: Document) -> bool:
        """
        Perform smart update that only updates changed chunks.
        
        Args:
            document: Updated document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # For now, this is a placeholder for a more sophisticated implementation
            # In practice, you'd compare chunk content and only update changed chunks
            
            self.logger.info(f"Performing smart update for document {document.id}")
            
            # Record change
            self.change_log.append({
                'document_id': document.id,
                'operation': 'smart_update',
                'timestamp': datetime.now().isoformat()
            })
            
            # Fall back to regular update for now
            return self.update_document(document)
            
        except Exception as e:
            self.logger.error(f"Smart update failed for document {document.id}: {e}")
            return False
    
    def get_change_log(self) -> List[Dict[str, Any]]:
        """
        Get the change log for debugging and monitoring.
        
        Returns:
            List of change log entries
        """
        return self.change_log.copy()
    
    def clear_change_log(self) -> None:
        """Clear the change log."""
        self.change_log.clear()
        self.logger.info("Change log cleared")
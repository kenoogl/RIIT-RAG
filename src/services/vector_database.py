"""
Local vector database implementation for the RAG system.

This module provides a file-based vector storage system that implements
the VectorDatabaseInterface. It supports storing, searching, and managing
embedding vectors with associated metadata using local file storage.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from ..models.interfaces import VectorDatabaseInterface
from ..models.core import SearchResult, Document, Chunk
from ..models.exceptions import StorageError, SearchError, DeletionError


class LocalVectorDatabase(VectorDatabaseInterface):
    """
    File-based vector database implementation.
    
    This class provides local storage for embedding vectors and their metadata
    using a combination of JSON files for metadata and pickle files for vectors.
    It supports cosine similarity search and index management operations.
    """
    
    def __init__(self, storage_path: str = "data/vectors"):
        """
        Initialize the local vector database.
        
        Args:
            storage_path: Path to the directory for storing vector data
        """
        self.storage_path = Path(storage_path)
        self.vectors_file = self.storage_path / "vectors.pkl"
        self.metadata_file = self.storage_path / "metadata.json"
        self.index_file = self.storage_path / "index.json"
        
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage files if they don't exist
        self._initialize_storage()
        
        # Load existing data
        self._load_data()
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_storage(self) -> None:
        """Initialize storage files if they don't exist."""
        if not self.vectors_file.exists():
            with open(self.vectors_file, 'wb') as f:
                pickle.dump([], f)
        
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
        
        if not self.index_file.exists():
            index_data = {
                'document_to_chunks': {},  # document_id -> list of chunk_indices
                'chunk_to_index': {},      # chunk_id -> vector_index
                'next_index': 0
            }
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    def _load_data(self) -> None:
        """Load existing vectors and metadata from storage."""
        try:
            # Load vectors
            with open(self.vectors_file, 'rb') as f:
                self.vectors = pickle.load(f)
            
            # Load metadata
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # Load index
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self.index_data = json.load(f)
            
            # Convert vectors to numpy array for efficient computation
            if self.vectors:
                self.vectors = np.array(self.vectors)
            else:
                self.vectors = np.array([]).reshape(0, 0)
                
        except Exception as e:
            self.logger.error(f"Failed to load vector database data: {e}")
            raise StorageError(f"Failed to load vector database: {e}")
    
    def _save_data(self) -> None:
        """Save vectors and metadata to storage."""
        try:
            # Save vectors
            vectors_to_save = self.vectors.tolist() if self.vectors.size > 0 else []
            with open(self.vectors_file, 'wb') as f:
                pickle.dump(vectors_to_save, f)
            
            # Save metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            # Save index
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save vector database data: {e}")
            raise StorageError(f"Failed to save vector database: {e}")
    
    def store_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> bool:
        """
        Store vectors with associated metadata.
        
        Args:
            vectors: List of embedding vectors
            metadata: Associated metadata for each vector
            
        Returns:
            True if storage successful, False otherwise
            
        Raises:
            StorageError: If vector storage fails
        """
        try:
            if len(vectors) != len(metadata):
                raise ValueError("Number of vectors must match number of metadata entries")
            
            if not vectors:
                return True  # Nothing to store
            
            # Convert vectors to numpy array
            new_vectors = np.array(vectors)
            
            # Validate vector dimensions
            if self.vectors.size > 0:
                if new_vectors.shape[1] != self.vectors.shape[1]:
                    raise ValueError(f"Vector dimension mismatch: expected {self.vectors.shape[1]}, got {new_vectors.shape[1]}")
            
            # Add timestamp to metadata
            current_time = datetime.now().isoformat()
            for meta in metadata:
                meta['stored_at'] = current_time
            
            # Update index for each new vector
            start_index = self.index_data['next_index']
            for i, meta in enumerate(metadata):
                vector_index = start_index + i
                chunk_id = meta.get('chunk_id')
                document_id = meta.get('document_id')
                
                if chunk_id:
                    self.index_data['chunk_to_index'][chunk_id] = vector_index
                
                if document_id:
                    if document_id not in self.index_data['document_to_chunks']:
                        self.index_data['document_to_chunks'][document_id] = []
                    self.index_data['document_to_chunks'][document_id].append(vector_index)
            
            # Update next index
            self.index_data['next_index'] = start_index + len(vectors)
            
            # Append new vectors and metadata
            if self.vectors.size == 0:
                self.vectors = new_vectors
            else:
                self.vectors = np.vstack([self.vectors, new_vectors])
            
            self.metadata.extend(metadata)
            
            # Save to storage
            self._save_data()
            
            self.logger.info(f"Successfully stored {len(vectors)} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store vectors: {e}")
            raise StorageError(f"Failed to store vectors: {e}")
    
    def search_similar(self, query_vector: List[float], top_k: int) -> List[SearchResult]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of search results ordered by similarity
            
        Raises:
            SearchError: If similarity search fails
        """
        try:
            if self.vectors.size == 0:
                return []
            
            if not query_vector:
                raise ValueError("Query vector cannot be empty")
            
            # Convert query vector to numpy array
            query_vec = np.array(query_vector)
            
            # Validate dimensions
            if query_vec.shape[0] != self.vectors.shape[1]:
                raise ValueError(f"Query vector dimension mismatch: expected {self.vectors.shape[1]}, got {query_vec.shape[0]}")
            
            # Calculate cosine similarities
            similarities = self._cosine_similarity(query_vec, self.vectors)
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Build search results
            results = []
            for idx in top_indices:
                if similarities[idx] <= 0:  # Skip non-positive similarities
                    continue
                    
                meta = self.metadata[idx]
                
                # Create chunk from metadata
                chunk = Chunk(
                    id=meta['chunk_id'],
                    document_id=meta['document_id'],
                    content=meta['content'],
                    position=meta['position'],
                    embedding=self.vectors[idx].tolist()
                )
                
                # Create document from metadata
                document = Document(
                    id=meta['document_id'],
                    url=meta.get('document_url', ''),
                    title=meta.get('document_title', ''),
                    content=meta.get('document_content', ''),
                    language=meta.get('document_language', 'ja'),
                    created_at=datetime.fromisoformat(meta.get('document_created_at', datetime.now().isoformat())),
                    updated_at=datetime.fromisoformat(meta.get('document_updated_at', datetime.now().isoformat()))
                )
                
                # Create search result
                result = SearchResult(
                    chunk=chunk,
                    score=float(similarities[idx]),
                    document=document
                )
                
                results.append(result)
            
            self.logger.info(f"Found {len(results)} similar vectors for query")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search similar vectors: {e}")
            raise SearchError(f"Failed to search similar vectors: {e}")
    
    def delete_by_document_id(self, document_id: str) -> bool:
        """
        Delete all vectors associated with a document.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if deletion successful, False otherwise
            
        Raises:
            DeletionError: If vector deletion fails
        """
        try:
            if document_id not in self.index_data['document_to_chunks']:
                self.logger.warning(f"Document {document_id} not found in index")
                return True  # Nothing to delete
            
            # Get indices to delete
            indices_to_delete = self.index_data['document_to_chunks'][document_id]
            indices_to_delete = sorted(indices_to_delete, reverse=True)  # Delete from end to avoid index shifts
            
            # Remove vectors and metadata
            for idx in indices_to_delete:
                if idx < len(self.metadata):
                    # Remove from chunk index
                    chunk_id = self.metadata[idx].get('chunk_id')
                    if chunk_id and chunk_id in self.index_data['chunk_to_index']:
                        del self.index_data['chunk_to_index'][chunk_id]
                    
                    # Remove metadata
                    del self.metadata[idx]
                    
                    # Remove vector
                    if self.vectors.size > 0 and idx < self.vectors.shape[0]:
                        self.vectors = np.delete(self.vectors, idx, axis=0)
            
            # Update document index
            del self.index_data['document_to_chunks'][document_id]
            
            # Update indices in the index data (shift down indices that are greater than deleted ones)
            for doc_id, chunk_indices in self.index_data['document_to_chunks'].items():
                updated_indices = []
                for chunk_idx in chunk_indices:
                    # Count how many indices were deleted before this one
                    shift = sum(1 for del_idx in indices_to_delete if del_idx < chunk_idx)
                    updated_indices.append(chunk_idx - shift)
                self.index_data['document_to_chunks'][doc_id] = updated_indices
            
            # Update chunk to index mapping
            for chunk_id, idx in list(self.index_data['chunk_to_index'].items()):
                shift = sum(1 for del_idx in indices_to_delete if del_idx < idx)
                self.index_data['chunk_to_index'][chunk_id] = idx - shift
            
            # Update next index
            self.index_data['next_index'] -= len(indices_to_delete)
            
            # Save changes
            self._save_data()
            
            self.logger.info(f"Successfully deleted {len(indices_to_delete)} vectors for document {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete vectors for document {document_id}: {e}")
            raise DeletionError(f"Failed to delete vectors for document {document_id}: {e}")
    
    def _cosine_similarity(self, query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query vector and all stored vectors.
        
        Args:
            query_vector: Query vector
            vectors: Matrix of stored vectors
            
        Returns:
            Array of cosine similarities
        """
        # Normalize vectors
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return np.zeros(vectors.shape[0])
        
        vectors_norm = np.linalg.norm(vectors, axis=1)
        vectors_norm[vectors_norm == 0] = 1  # Avoid division by zero
        
        # Calculate cosine similarity
        similarities = np.dot(vectors, query_vector) / (vectors_norm * query_norm)
        
        return similarities
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary containing database statistics
        """
        return {
            'total_vectors': self.vectors.shape[0] if self.vectors.size > 0 else 0,
            'vector_dimension': self.vectors.shape[1] if self.vectors.size > 0 else 0,
            'total_documents': len(self.index_data['document_to_chunks']),
            'storage_path': str(self.storage_path),
            'next_index': self.index_data['next_index']
        }
    
    def clear_all(self) -> bool:
        """
        Clear all stored vectors and metadata.
        
        Returns:
            True if clearing successful, False otherwise
        """
        try:
            self.vectors = np.array([]).reshape(0, 0)
            self.metadata = []
            self.index_data = {
                'document_to_chunks': {},
                'chunk_to_index': {},
                'next_index': 0
            }
            
            self._save_data()
            self.logger.info("Successfully cleared all vector data")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear vector data: {e}")
            return False
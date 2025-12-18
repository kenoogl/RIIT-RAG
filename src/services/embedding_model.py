"""
Embedding model service for the RAG system.

This module provides local embedding model functionality using SentenceTransformers
for generating semantic embeddings from Japanese text content.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

from ..models.interfaces import EmbeddingModelInterface
from ..models.exceptions import ModelLoadError, EncodingError
from ..utils.config import get_config


class LocalEmbeddingModel(EmbeddingModelInterface):
    """
    Local embedding model implementation using SentenceTransformers.
    
    This class provides Japanese-compatible text embedding functionality
    using pre-trained multilingual models that work offline.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        batch_size: int = 32
    ):
        """
        Initialize the local embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model
            model_path: Local path to store/load the model
            device: Device to run the model on ('cpu' or 'cuda')
            max_seq_length: Maximum sequence length for tokenization
            batch_size: Batch size for encoding
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        config = get_config()
        embedding_config = config.get('embedding', {})
        
        self.model_name = model_name or embedding_config.get('model_name', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.model_path = Path(model_path or embedding_config.get('model_path', './models/embedding'))
        self.device = device or embedding_config.get('device', 'cpu')
        self.max_seq_length = max_seq_length or embedding_config.get('max_seq_length', 512)
        self.batch_size = batch_size or embedding_config.get('batch_size', 32)
        
        self.model: Optional[SentenceTransformer] = None
        self.is_loaded = False
        
        # Create model directory if it doesn't exist
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized embedding model service with model: {self.model_name}")
    
    def load_model(self, model_name: Optional[str] = None) -> bool:
        """
        Load the embedding model.
        
        Args:
            model_name: Optional model name to override default
            
        Returns:
            True if model loaded successfully, False otherwise
            
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            target_model = model_name or self.model_name
            local_model_path = self.model_path / target_model.replace('/', '_')
            
            self.logger.info(f"Loading embedding model: {target_model}")
            
            # Try to load from local path first
            if local_model_path.exists() and any(local_model_path.iterdir()):
                self.logger.info(f"Loading model from local path: {local_model_path}")
                self.model = SentenceTransformer(str(local_model_path), device=self.device)
            else:
                self.logger.info(f"Downloading and loading model: {target_model}")
                self.model = SentenceTransformer(target_model, device=self.device)
                
                # Save model locally for future use
                self.logger.info(f"Saving model to local path: {local_model_path}")
                self.model.save(str(local_model_path))
            
            # Configure model settings
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.max_seq_length
            
            self.is_loaded = True
            
            # Log model information
            embedding_dim = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Model loaded successfully. Embedding dimension: {embedding_dim}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise ModelLoadError(f"Failed to load embedding model {target_model}: {e}")
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode texts into embedding vectors.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of embedding vectors
            
        Raises:
            EncodingError: If encoding fails
        """
        if not self.is_loaded or self.model is None:
            raise EncodingError("Model not loaded. Call load_model() first.")
        
        if not texts:
            return []
        
        try:
            self.logger.debug(f"Encoding {len(texts)} texts")
            
            # Filter out empty texts
            non_empty_texts = [text.strip() for text in texts if text.strip()]
            if not non_empty_texts:
                self.logger.warning("All input texts are empty")
                return [[0.0] * self.model.get_sentence_embedding_dimension()] * len(texts)
            
            # Encode texts in batches
            embeddings = []
            for i in range(0, len(non_empty_texts), self.batch_size):
                batch_texts = non_empty_texts[i:i + self.batch_size]
                
                # Generate embeddings
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=len(batch_texts)
                )
                
                embeddings.extend(batch_embeddings.tolist())
            
            # Handle empty texts by mapping back to original positions
            result_embeddings = []
            non_empty_idx = 0
            empty_embedding = [0.0] * self.model.get_sentence_embedding_dimension()
            
            for original_text in texts:
                if original_text.strip():
                    result_embeddings.append(embeddings[non_empty_idx])
                    non_empty_idx += 1
                else:
                    result_embeddings.append(empty_embedding)
            
            self.logger.debug(f"Successfully encoded {len(texts)} texts")
            return result_embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to encode texts: {e}")
            raise EncodingError(f"Failed to encode texts: {e}")
    
    def encode_single(self, text: str) -> List[float]:
        """
        Encode a single text into an embedding vector.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
            
        Raises:
            EncodingError: If encoding fails
        """
        embeddings = self.encode([text])
        return embeddings[0] if embeddings else []
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
            
        Raises:
            ModelLoadError: If model is not loaded
        """
        if not self.is_loaded or self.model is None:
            raise ModelLoadError("Model not loaded. Call load_model() first.")
        
        return self.model.get_sentence_embedding_dimension()
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before encoding.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Basic text preprocessing
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (leave some buffer for tokenization)
        max_chars = self.max_seq_length * 4  # Rough estimate for Japanese
        if len(text) > max_chars:
            text = text[:max_chars]
            self.logger.debug(f"Truncated text to {max_chars} characters")
        
        return text
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if not self.is_loaded or self.model is None:
            return {
                'loaded': False,
                'model_name': self.model_name,
                'model_path': str(self.model_path)
            }
        
        return {
            'loaded': True,
            'model_name': self.model_name,
            'model_path': str(self.model_path),
            'embedding_dimension': self.model.get_sentence_embedding_dimension(),
            'max_seq_length': self.max_seq_length,
            'device': self.device,
            'batch_size': self.batch_size
        }
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("Model unloaded successfully")


class EmbeddingService:
    """
    High-level embedding service that manages the embedding model.
    
    This class provides a convenient interface for embedding operations
    and handles model lifecycle management.
    """
    
    def __init__(self):
        """Initialize the embedding service."""
        self.logger = logging.getLogger(__name__)
        self.model = LocalEmbeddingModel()
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize the embedding service by loading the model.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if not self._initialized:
                success = self.model.load_model()
                if success:
                    self._initialized = True
                    self.logger.info("Embedding service initialized successfully")
                return success
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding service: {e}")
            return False
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not self._initialized:
            raise RuntimeError("Embedding service not initialized")
        
        preprocessed_text = self.model.preprocess_text(text)
        return self.model.encode_single(preprocessed_text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            raise RuntimeError("Embedding service not initialized")
        
        preprocessed_texts = [self.model.preprocess_text(text) for text in texts]
        return self.model.encode(preprocessed_texts)
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        if not self._initialized:
            raise RuntimeError("Embedding service not initialized")
        
        return self.model.get_embedding_dimension()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score
        """
        if not self._initialized:
            raise RuntimeError("Embedding service not initialized")
        
        embeddings = self.embed_texts([text1, text2])
        return self.model.similarity(embeddings[0], embeddings[1])
    
    def shutdown(self) -> None:
        """Shutdown the embedding service."""
        if self._initialized:
            self.model.unload_model()
            self._initialized = False
            self.logger.info("Embedding service shutdown")


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Get the global embedding service instance.
    
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
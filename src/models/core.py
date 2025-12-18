"""
Core data models for the RAG system.

This module defines the fundamental data structures used throughout the system:
- Document: Represents a source document
- Chunk: Represents a processed text chunk with embeddings
- SearchResult: Represents a search result with relevance score
- Answer: Represents a generated answer with metadata
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
from abc import ABC, abstractmethod


@dataclass
class Document:
    """Represents a source document in the system."""
    
    id: str
    url: str
    title: str
    content: str
    language: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate document data after initialization."""
        if not self.id:
            raise ValueError("Document ID cannot be empty")
        if not self.url:
            raise ValueError("Document URL cannot be empty")
        if not self.title:
            raise ValueError("Document title cannot be empty")
        if not self.content:
            raise ValueError("Document content cannot be empty")
        if not self.language:
            raise ValueError("Document language cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for serialization."""
        return {
            'id': self.id,
            'url': self.url,
            'title': self.title,
            'content': self.content,
            'language': self.language,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create document from dictionary."""
        return cls(
            id=data['id'],
            url=data['url'],
            title=data['title'],
            content=data['content'],
            language=data['language'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at'])
        )
    
    def to_json(self) -> str:
        """Convert document to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Document':
        """Create document from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class Chunk:
    """Represents a processed text chunk with optional embeddings."""
    
    id: str
    document_id: str
    content: str
    position: int
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate chunk data after initialization."""
        if not self.id:
            raise ValueError("Chunk ID cannot be empty")
        if not self.document_id:
            raise ValueError("Document ID cannot be empty")
        if not self.content:
            raise ValueError("Chunk content cannot be empty")
        if self.position < 0:
            raise ValueError("Chunk position must be non-negative")
        if self.embedding is not None and len(self.embedding) == 0:
            raise ValueError("Embedding vector cannot be empty if provided")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            'id': self.id,
            'document_id': self.document_id,
            'content': self.content,
            'position': self.position,
            'embedding': self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create chunk from dictionary."""
        return cls(
            id=data['id'],
            document_id=data['document_id'],
            content=data['content'],
            position=data['position'],
            embedding=data.get('embedding')
        )
    
    def to_json(self) -> str:
        """Convert chunk to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Chunk':
        """Create chunk from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class SearchResult:
    """Represents a search result with relevance score."""
    
    chunk: Chunk
    score: float
    document: Document
    
    def __post_init__(self):
        """Validate search result data after initialization."""
        if not isinstance(self.chunk, Chunk):
            raise ValueError("chunk must be a Chunk instance")
        if not isinstance(self.document, Document):
            raise ValueError("document must be a Document instance")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError("Score must be between 0.0 and 1.0")
        if self.chunk.document_id != self.document.id:
            raise ValueError("Chunk document_id must match document id")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary for serialization."""
        return {
            'chunk': self.chunk.to_dict(),
            'score': self.score,
            'document': self.document.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """Create search result from dictionary."""
        return cls(
            chunk=Chunk.from_dict(data['chunk']),
            score=data['score'],
            document=Document.from_dict(data['document'])
        )
    
    def to_json(self) -> str:
        """Convert search result to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SearchResult':
        """Create search result from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class Answer:
    """Represents a generated answer with metadata."""
    
    text: str
    sources: List[str]
    confidence: float
    processing_time: float
    
    def __post_init__(self):
        """Validate answer data after initialization."""
        if not self.text:
            raise ValueError("Answer text cannot be empty")
        if not isinstance(self.sources, list):
            raise ValueError("Sources must be a list")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.processing_time < 0:
            raise ValueError("Processing time must be non-negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert answer to dictionary for serialization."""
        return {
            'text': self.text,
            'sources': self.sources,
            'confidence': self.confidence,
            'processing_time': self.processing_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Answer':
        """Create answer from dictionary."""
        return cls(
            text=data['text'],
            sources=data['sources'],
            confidence=data['confidence'],
            processing_time=data['processing_time']
        )
    
    def to_json(self) -> str:
        """Convert answer to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Answer':
        """Create answer from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


# Base classes for serialization
class Serializable(ABC):
    """Abstract base class for serializable objects."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """Create object from dictionary."""
        pass
    
    def to_json(self) -> str:
        """Convert object to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Serializable':
        """Create object from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
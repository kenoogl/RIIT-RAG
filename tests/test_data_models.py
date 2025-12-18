"""
Basic tests for core data models.

This module contains simple tests to verify that the data models
work correctly with validation and serialization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime
from models import Document, Chunk, SearchResult, Answer


def test_document_creation():
    """Test basic document creation and validation."""
    doc = Document(
        id="test-doc-1",
        url="https://example.com/doc1",
        title="Test Document",
        content="This is test content",
        language="ja"
    )
    
    assert doc.id == "test-doc-1"
    assert doc.url == "https://example.com/doc1"
    assert doc.title == "Test Document"
    assert doc.content == "This is test content"
    assert doc.language == "ja"
    assert isinstance(doc.created_at, datetime)
    assert isinstance(doc.updated_at, datetime)
    
    print("✓ Document creation test passed")


def test_document_serialization():
    """Test document serialization and deserialization."""
    doc = Document(
        id="test-doc-2",
        url="https://example.com/doc2",
        title="Serialization Test",
        content="Test serialization content",
        language="ja"
    )
    
    # Test to_dict and from_dict
    doc_dict = doc.to_dict()
    doc_restored = Document.from_dict(doc_dict)
    
    assert doc_restored.id == doc.id
    assert doc_restored.url == doc.url
    assert doc_restored.title == doc.title
    assert doc_restored.content == doc.content
    assert doc_restored.language == doc.language
    
    # Test JSON serialization
    json_str = doc.to_json()
    doc_from_json = Document.from_json(json_str)
    
    assert doc_from_json.id == doc.id
    assert doc_from_json.url == doc.url
    
    print("✓ Document serialization test passed")


def test_chunk_creation():
    """Test basic chunk creation and validation."""
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="This is a chunk of text",
        position=0,
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    
    assert chunk.id == "chunk-1"
    assert chunk.document_id == "doc-1"
    assert chunk.content == "This is a chunk of text"
    assert chunk.position == 0
    assert chunk.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
    
    print("✓ Chunk creation test passed")


def test_search_result_creation():
    """Test search result creation with validation."""
    doc = Document(
        id="doc-1",
        url="https://example.com/doc1",
        title="Test Document",
        content="Test content",
        language="ja"
    )
    
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="Test chunk content",
        position=0
    )
    
    result = SearchResult(
        chunk=chunk,
        score=0.85,
        document=doc
    )
    
    assert result.chunk == chunk
    assert result.score == 0.85
    assert result.document == doc
    
    print("✓ SearchResult creation test passed")


def test_answer_creation():
    """Test answer creation and validation."""
    answer = Answer(
        text="This is the generated answer",
        sources=["doc-1", "doc-2"],
        confidence=0.9,
        processing_time=1.5
    )
    
    assert answer.text == "This is the generated answer"
    assert answer.sources == ["doc-1", "doc-2"]
    assert answer.confidence == 0.9
    assert answer.processing_time == 1.5
    
    print("✓ Answer creation test passed")


def test_validation_errors():
    """Test that validation errors are raised correctly."""
    try:
        # Test empty document ID
        Document(id="", url="https://example.com", title="Test", content="Test", language="ja")
        assert False, "Should have raised ValueError for empty ID"
    except ValueError as e:
        assert "Document ID cannot be empty" in str(e)
    
    try:
        # Test invalid score
        doc = Document(id="doc-1", url="https://example.com", title="Test", content="Test", language="ja")
        chunk = Chunk(id="chunk-1", document_id="doc-1", content="Test", position=0)
        SearchResult(chunk=chunk, score=1.5, document=doc)  # Invalid score > 1.0
        assert False, "Should have raised ValueError for invalid score"
    except ValueError as e:
        assert "Score must be between 0.0 and 1.0" in str(e)
    
    print("✓ Validation error test passed")


if __name__ == "__main__":
    test_document_creation()
    test_document_serialization()
    test_chunk_creation()
    test_search_result_creation()
    test_answer_creation()
    test_validation_errors()
    print("\n🎉 All data model tests passed!")
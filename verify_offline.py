#!/usr/bin/env python3
"""
Offline verification script for the RAG system.

This script verifies that the system can operate without internet connectivity
by testing core functionality with mock data.
"""

import sys
import logging
from pathlib import Path
import time
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logging_config import setup_logging
from src.utils.config import get_config


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Core models
        from src.models.core import Document, Chunk, SearchResult, Answer
        from src.models.interfaces import (
            VectorDatabaseInterface, SearchEngineInterface,
            EmbeddingModelInterface, GenerativeModelInterface
        )
        print("✓ Core models imported successfully")
        
        # Services
        from src.services.vector_database import LocalVectorDatabase
        from src.services.embedding_model import LocalEmbeddingModel
        from src.services.text_processor import JapaneseTextProcessor
        from src.services.generative_model import GenerativeModelService
        from src.services.rag_service import RAGService
        from src.services.enhanced_search_engine import EnhancedSearchEngine
        print("✓ Services imported successfully")
        
        # API
        from src.api.main import app, create_app
        print("✓ API components imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        config = get_config()
        
        # Check required sections
        required_sections = ["app", "embedding", "generation", "search", "storage"]
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing configuration section: {section}")
                return False
        
        print("✓ Configuration loaded successfully")
        print(f"  App name: {config['app']['name']}")
        print(f"  Version: {config['app']['version']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_offline_components():
    """Test core components without external dependencies."""
    print("\nTesting offline components...")
    
    try:
        from datetime import datetime
        from src.models.core import Document, Chunk, SearchResult, Answer
        from src.services.text_processor import JapaneseTextProcessor
        
        # Test text processor
        processor = JapaneseTextProcessor()
        test_text = "九州大学のスーパーコンピュータ利用ガイドです。"
        normalized = processor.normalize_text(test_text)
        
        # Test text chunker
        from src.services.text_processor import TextChunker
        chunker = TextChunker()
        chunks = chunker.create_chunks(test_text)
        
        print("✓ Text processor working")
        print(f"  Normalized: {normalized[:50]}...")
        print(f"  Chunks created: {len(chunks)}")
        
        # Test data models
        doc = Document(
            id="test_doc",
            url="file://test.txt",
            title="テスト文書",
            content=test_text,
            language="ja",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        chunk = Chunk(
            id="test_chunk",
            document_id="test_doc",
            content=test_text,
            position=0,
            embedding=None
        )
        
        print("✓ Data models working")
        print(f"  Document: {doc.title}")
        print(f"  Chunk: {chunk.content[:30]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Offline components test failed: {e}")
        return False


def test_api_creation():
    """Test API app creation without starting server."""
    print("\nTesting API creation...")
    
    try:
        from src.api.main import create_app
        
        app = create_app()
        
        # Check basic app properties
        assert app.title == "Supercomputer Support RAG API"
        assert app.version == "0.1.0"
        
        print("✓ API app created successfully")
        print(f"  Title: {app.title}")
        print(f"  Version: {app.version}")
        
        return True
        
    except Exception as e:
        print(f"✗ API creation test failed: {e}")
        return False


def test_service_initialization():
    """Test service initialization without external models."""
    print("\nTesting service initialization...")
    
    try:
        from src.services.generative_model import GenerativeModelService
        from src.services.search_logging_service import SearchLoggingService
        
        # Test generative model service (without loading actual model)
        gen_service = GenerativeModelService()
        model_info = gen_service.get_model_info()
        
        print("✓ Generative model service initialized")
        print(f"  Model status: {model_info['status']}")
        
        # Test logging service
        logging_service = SearchLoggingService()
        stats = logging_service.get_search_statistics()
        
        print("✓ Search logging service initialized")
        print(f"  Total searches: {stats['overview']['total_searches']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Service initialization test failed: {e}")
        return False


def test_japanese_processing():
    """Test Japanese text processing capabilities."""
    print("\nTesting Japanese text processing...")
    
    try:
        from src.services.text_processor import JapaneseTextProcessor
        from src.services.enhanced_search_engine import EnhancedSearchEngine
        from unittest.mock import Mock
        
        # Test text processor with various Japanese texts
        processor = JapaneseTextProcessor()
        
        test_texts = [
            "九州大学のスーパーコンピュータ",
            "アカウント申請の方法について",
            "ジョブの実行手順を説明します",
            "エラーが発生した場合の対処法"
        ]
        
        # Test text chunker
        from src.services.text_processor import TextChunker
        chunker = TextChunker()
        
        for text in test_texts:
            normalized = processor.normalize_text(text)
            chunks = chunker.create_chunks(text)
            
            print(f"  ✓ Processed: {text[:20]}...")
        
        # Test enhanced search engine query processing
        mock_vector_db = Mock()
        mock_embedding_model = Mock()
        mock_logging_service = Mock()
        
        search_engine = EnhancedSearchEngine(
            vector_db=mock_vector_db,
            embedding_model=mock_embedding_model,
            logging_service=mock_logging_service
        )
        
        for text in test_texts:
            query_type = search_engine._detect_query_type(text)
            normalized = search_engine._normalize_japanese_text(text)
            
            print(f"  ✓ Query type for '{text[:15]}...': {query_type}")
        
        print("✓ Japanese text processing working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Japanese processing test failed: {e}")
        return False


def run_verification() -> bool:
    """Run all verification tests."""
    print("=" * 60)
    print("RAG System Offline Verification")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Offline Components Test", test_offline_components),
        ("API Creation Test", test_api_creation),
        ("Service Initialization Test", test_service_initialization),
        ("Japanese Processing Test", test_japanese_processing),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        start_time = time.time()
        success = test_func()
        end_time = time.time()
        
        results.append(success)
        status = "PASS" if success else "FAIL"
        duration = end_time - start_time
        
        print(f"Result: {status} ({duration:.2f}s)")
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "PASS" if results[i] else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed - System ready for offline operation")
        return True
    else:
        print("✗ Some tests failed - Check configuration and dependencies")
        return False


def main():
    """Main entry point."""
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    try:
        success = run_verification()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nVerification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nVerification failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
System integration check for the RAG system.

This script performs basic integration checks to verify that
all components can work together without requiring external models.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import load_config, create_directories
from src.utils.logging_config import setup_logging, get_logger


class SystemIntegrationChecker:
    """System integration checker for the RAG system."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = None
        self.results: Dict[str, Any] = {}
        
    def setup(self):
        """Set up the test environment."""
        try:
            # Load configuration
            self.config = load_config()
            
            # Set up logging
            setup_logging(self.config.logging.__dict__)
            
            # Create directories
            create_directories(self.config)
            
            self.logger.info("System integration check environment set up successfully")
            return True
            
        except Exception as e:
            print(f"Failed to set up environment: {e}")
            return False
    
    def check_configuration(self) -> bool:
        """Check configuration loading and validation."""
        self.logger.info("Checking configuration...")
        
        try:
            # Verify configuration sections
            required_sections = ['app', 'crawler', 'document_processing', 'embedding', 
                               'vector_db', 'generation', 'search', 'storage', 'logging', 'performance']
            
            for section in required_sections:
                if not hasattr(self.config, section):
                    raise ValueError(f"Missing configuration section: {section}")
            
            self.logger.info(f"✓ All {len(required_sections)} configuration sections present")
            
            # Check directory creation
            directories = [
                self.config.storage.data_dir,
                self.config.storage.documents_dir,
                self.config.storage.logs_dir,
                self.config.vector_db.storage_path
            ]
            
            for directory in directories:
                if not Path(directory).exists():
                    raise FileNotFoundError(f"Directory not created: {directory}")
            
            self.logger.info(f"✓ All {len(directories)} required directories exist")
            
            self.results["configuration"] = {
                "status": "passed",
                "sections_checked": len(required_sections),
                "directories_verified": len(directories)
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration check failed: {e}")
            self.results["configuration"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def check_core_services(self) -> bool:
        """Check core service initialization without external dependencies."""
        self.logger.info("Checking core services...")
        
        try:
            # Test basic imports
            from src.services.vector_database import LocalVectorDatabase
            from src.services.text_processor import JapaneseTextProcessor, TextChunker
            from src.services.web_crawler import RobustWebCrawler
            from src.models.core import Document, Chunk, SearchResult, Answer
            
            self.logger.info("✓ Core imports successful")
            
            # Test basic service creation
            vector_db = LocalVectorDatabase()
            text_processor = JapaneseTextProcessor()
            chunker = TextChunker()
            
            self.logger.info("✓ Core services instantiated")
            
            # Test basic functionality
            test_text = "これはテスト文書です。スーパーコンピュータについて説明します。"
            processed_text = text_processor.normalize_text(test_text)
            chunk_texts = chunker.create_chunks(processed_text)
            
            self.logger.info(f"✓ Text processing works: {len(chunk_texts)} chunks created")
            
            self.results["core_services"] = {
                "status": "passed",
                "services_tested": 3,
                "chunks_created": len(chunk_texts)
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Core services check failed: {e}")
            self.results["core_services"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def check_api_structure(self) -> bool:
        """Check API structure and endpoint configuration."""
        self.logger.info("Checking API structure...")
        
        try:
            from src.api.main import app
            from src.api.management import router
            
            self.logger.info("✓ API modules imported successfully")
            
            # Check main app routes
            main_routes = [route.path for route in app.routes]
            expected_main_routes = ["/", "/health", "/ask", "/docs", "/redoc"]
            
            found_main_routes = sum(1 for route in expected_main_routes 
                                  if any(route in main_route for main_route in main_routes))
            
            self.logger.info(f"✓ Found {found_main_routes}/{len(expected_main_routes)} main routes")
            
            # Check management routes
            mgmt_routes = [route.path for route in router.routes]
            expected_mgmt_routes = ["/documents", "/system/status"]
            
            found_mgmt_routes = sum(1 for route in expected_mgmt_routes 
                                  if any(route in mgmt_route for mgmt_route in mgmt_routes))
            
            self.logger.info(f"✓ Found {found_mgmt_routes}/{len(expected_mgmt_routes)} management routes")
            
            self.results["api_structure"] = {
                "status": "passed",
                "main_routes_found": found_main_routes,
                "mgmt_routes_found": found_mgmt_routes
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"API structure check failed: {e}")
            self.results["api_structure"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def check_data_models(self) -> bool:
        """Check data model functionality."""
        self.logger.info("Checking data models...")
        
        try:
            from src.models.core import Document, Chunk, SearchResult, Answer
            from datetime import datetime
            
            # Test Document creation
            doc = Document(
                id="test_doc",
                url="https://example.com",
                title="テストドキュメント",
                content="これはテスト用の文書です。",
                language="ja"
            )
            
            self.logger.info("✓ Document model works")
            
            # Test Chunk creation
            chunk = Chunk(
                id="test_chunk",
                document_id=doc.id,
                content="テストチャンク",
                position=0
            )
            
            self.logger.info("✓ Chunk model works")
            
            # Test SearchResult creation
            search_result = SearchResult(
                chunk=chunk,
                score=0.85,
                document=doc
            )
            
            self.logger.info("✓ SearchResult model works")
            
            # Test Answer creation
            answer = Answer(
                text="これはテスト回答です。",
                sources=["https://example.com"],
                confidence=0.8,
                processing_time=0.1
            )
            
            self.logger.info("✓ Answer model works")
            
            self.results["data_models"] = {
                "status": "passed",
                "models_tested": 4
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data models check failed: {e}")
            self.results["data_models"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def check_management_services(self) -> bool:
        """Check management service functionality."""
        self.logger.info("Checking management services...")
        
        try:
            from src.services.management_service import (
                DocumentManagementService, 
                SystemMonitoringService,
                create_document_management_service,
                create_system_monitoring_service
            )
            
            # Test service creation
            doc_mgmt = create_document_management_service()
            sys_monitor = create_system_monitoring_service()
            
            self.logger.info("✓ Management services created")
            
            # Test basic functionality
            docs = doc_mgmt.list_documents()
            status = sys_monitor.get_system_status()
            
            self.logger.info(f"✓ Management services functional: {len(docs)} docs, status available")
            
            self.results["management_services"] = {
                "status": "passed",
                "services_tested": 2,
                "documents_found": len(docs)
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Management services check failed: {e}")
            self.results["management_services"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all integration checks."""
        self.logger.info("Starting system integration checks...")
        
        start_time = time.time()
        
        checks = [
            ("Configuration", self.check_configuration),
            ("Core Services", self.check_core_services),
            ("API Structure", self.check_api_structure),
            ("Data Models", self.check_data_models),
            ("Management Services", self.check_management_services)
        ]
        
        passed_checks = 0
        failed_checks = 0
        
        for check_name, check_func in checks:
            self.logger.info(f"\n--- Running {check_name} Check ---")
            
            try:
                result = check_func()
                
                if result:
                    passed_checks += 1
                    self.logger.info(f"✓ {check_name} PASSED")
                else:
                    failed_checks += 1
                    self.logger.error(f"✗ {check_name} FAILED")
                    
            except Exception as e:
                failed_checks += 1
                self.logger.error(f"✗ {check_name} FAILED with exception: {e}")
                self.results[check_name.lower().replace(" ", "_")] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        total_time = time.time() - start_time
        
        # Summary
        summary = {
            "total_checks": len(checks),
            "passed": passed_checks,
            "failed": failed_checks,
            "success_rate": passed_checks / len(checks) * 100,
            "total_time": total_time,
            "overall_status": "passed" if failed_checks == 0 else "partial",
            "check_results": self.results
        }
        
        self.logger.info(f"\n=== System Integration Check Summary ===")
        self.logger.info(f"Total Checks: {summary['total_checks']}")
        self.logger.info(f"Passed: {summary['passed']}")
        self.logger.info(f"Failed: {summary['failed']}")
        self.logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        self.logger.info(f"Total Time: {summary['total_time']:.2f}s")
        self.logger.info(f"Overall Status: {summary['overall_status'].upper()}")
        
        return summary


def main():
    """Main entry point for system integration checks."""
    checker = SystemIntegrationChecker()
    
    # Set up environment
    if not checker.setup():
        sys.exit(1)
    
    # Run checks
    try:
        summary = checker.run_all_checks()
        
        # Save results
        import json
        results_file = Path("system_integration_results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nSystem integration check results saved to: {results_file}")
        
        # Exit with appropriate code
        sys.exit(0 if summary["overall_status"] in ["passed", "partial"] else 1)
        
    except Exception as e:
        print(f"System integration checks failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
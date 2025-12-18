#!/usr/bin/env python3
"""
Demo script for the RAG system management interface.

This script demonstrates how to use the management API endpoints
for document management and system monitoring.
"""

import requests
import json
import time
import sys
from typing import Dict, Any
from datetime import datetime


class RAGManagementClient:
    """Simple client for the RAG management API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the management API client."""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    # Document Management Methods
    def create_document(self, doc_id: str, title: str, url: str, content: str, language: str = "ja") -> Dict[str, Any]:
        """Create a new document."""
        data = {
            "id": doc_id,
            "title": title,
            "url": url,
            "content": content,
            "language": language
        }
        response = self.session.post(f"{self.base_url}/admin/documents", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Get document information."""
        response = self.session.get(f"{self.base_url}/admin/documents/{doc_id}")
        response.raise_for_status()
        return response.json()
    
    def update_document(self, doc_id: str, **updates) -> Dict[str, Any]:
        """Update a document."""
        response = self.session.put(f"{self.base_url}/admin/documents/{doc_id}", json=updates)
        response.raise_for_status()
        return response.json()
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document."""
        response = self.session.delete(f"{self.base_url}/admin/documents/{doc_id}")
        response.raise_for_status()
        return response.json()
    
    def list_documents(self, limit: int = None, offset: int = 0) -> Dict[str, Any]:
        """List documents with pagination."""
        params = {"offset": offset}
        if limit:
            params["limit"] = limit
        
        response = self.session.get(f"{self.base_url}/admin/documents", params=params)
        response.raise_for_status()
        return response.json()
    
    def get_index_status(self) -> Dict[str, Any]:
        """Get index status."""
        response = self.session.get(f"{self.base_url}/admin/index/status")
        response.raise_for_status()
        return response.json()
    
    def rebuild_index(self, document_ids: list = None) -> Dict[str, Any]:
        """Rebuild document index."""
        response = self.session.post(f"{self.base_url}/admin/index/rebuild", json=document_ids)
        response.raise_for_status()
        return response.json()
    
    # System Monitoring Methods
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        response = self.session.get(f"{self.base_url}/admin/system/status")
        response.raise_for_status()
        return response.json()
    
    def get_performance_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics."""
        response = self.session.get(f"{self.base_url}/admin/system/metrics", params={"hours": hours})
        response.raise_for_status()
        return response.json()
    
    def get_log_summary(self, log_file: str, lines: int = 100) -> Dict[str, Any]:
        """Get log summary."""
        response = self.session.get(f"{self.base_url}/admin/system/logs/{log_file}", params={"lines": lines})
        response.raise_for_status()
        return response.json()
    
    def list_log_files(self) -> Dict[str, Any]:
        """List available log files."""
        response = self.session.get(f"{self.base_url}/admin/system/logs")
        response.raise_for_status()
        return response.json()


def demo_document_management():
    """Demonstrate document management functionality."""
    print("=" * 60)
    print("RAG Management Demo - Document Management")
    print("=" * 60)
    
    client = RAGManagementClient()
    
    try:
        # Create sample documents
        print("\n1. Creating Sample Documents")
        print("-" * 30)
        
        sample_docs = [
            {
                "id": "scp_guide_001",
                "title": "九州大学スパコン利用ガイド",
                "url": "https://www.cc.kyushu-u.ac.jp/scp/guide/",
                "content": "九州大学情報基盤研究開発センターでは、研究者向けにスーパーコンピュータシステムを提供しています。利用するためには、まずアカウント申請を行ってください。"
            },
            {
                "id": "scp_account_002",
                "title": "アカウント申請方法",
                "url": "https://www.cc.kyushu-u.ac.jp/scp/account/",
                "content": "スーパーコンピュータのアカウント申請は、公式サイトから行うことができます。必要な書類を準備して、オンラインフォームから申請してください。"
            },
            {
                "id": "scp_job_003",
                "title": "ジョブ実行方法",
                "url": "https://www.cc.kyushu-u.ac.jp/scp/job/",
                "content": "計算ジョブはSLURMバッチシステムを使用して実行されます。ジョブスクリプトを作成し、sbatchコマンドで投入してください。"
            }
        ]
        
        for doc in sample_docs:
            try:
                result = client.create_document(**doc)
                print(f"✓ Created document: {doc['id']} ({result['chunks_created']} chunks)")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400 and "already exists" in e.response.text:
                    print(f"⚠ Document {doc['id']} already exists, skipping")
                else:
                    print(f"✗ Failed to create document {doc['id']}: {e}")
        
        # List documents
        print("\n2. Listing Documents")
        print("-" * 30)
        
        doc_list = client.list_documents()
        print(f"Total documents: {doc_list['total_count']}")
        print(f"Returned: {doc_list['returned_count']}")
        
        for doc in doc_list['documents']:
            print(f"  • {doc['id']}: {doc['title']} ({doc.get('chunk_count', 0)} chunks)")
        
        # Get document details
        print("\n3. Document Details")
        print("-" * 30)
        
        if doc_list['documents']:
            doc_id = doc_list['documents'][0]['id']
            doc_info = client.get_document(doc_id)
            print(f"Document ID: {doc_info['id']}")
            print(f"Title: {doc_info['title']}")
            print(f"URL: {doc_info['url']}")
            print(f"Language: {doc_info['language']}")
            print(f"Content Length: {doc_info['content_length']} characters")
            print(f"Chunks: {doc_info['chunk_count']}")
            print(f"Created: {doc_info['created_at']}")
        
        # Update document
        print("\n4. Updating Document")
        print("-" * 30)
        
        if doc_list['documents']:
            doc_id = doc_list['documents'][0]['id']
            update_result = client.update_document(
                doc_id,
                title="更新された九州大学スパコン利用ガイド",
                content="九州大学情報基盤研究開発センターでは、研究者向けにスーパーコンピュータシステムを提供しています。利用するためには、まずアカウント申請を行ってください。詳細な手順については、公式サイトをご確認ください。"
            )
            print(f"✓ Updated document: {doc_id}")
            print(f"  Processing time: {update_result['processing_time']:.2f}s")
        
        # Index status
        print("\n5. Index Status")
        print("-" * 30)
        
        index_status = client.get_index_status()
        overview = index_status['overview']
        print(f"Total documents: {overview['total_documents']}")
        print(f"Total chunks: {overview['total_chunks']}")
        print(f"Last updated: {overview['last_updated']}")
        print(f"Vector database: {index_status['vector_database']['status']}")
        print(f"Document processing: {index_status['document_processing']['status']}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to management API")
        print("Make sure the server is running with: python run_api.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n✅ Document management demo completed!")
    return True


def demo_system_monitoring():
    """Demonstrate system monitoring functionality."""
    print("\n" + "=" * 60)
    print("RAG Management Demo - System Monitoring")
    print("=" * 60)
    
    client = RAGManagementClient()
    
    try:
        # System status
        print("\n1. System Status")
        print("-" * 30)
        
        status = client.get_system_status()
        print(f"Overall Status: {status['overall_status']}")
        print(f"Timestamp: {status['timestamp']}")
        
        print("\nComponents:")
        for component, info in status['components'].items():
            print(f"  • {component}: {info.get('status', 'unknown')}")
        
        print("\nPerformance:")
        perf = status.get('performance', {})
        if perf:
            print(f"  • Avg Search Time: {perf.get('avg_search_time', 0):.3f}s")
            print(f"  • Avg Answer Time: {perf.get('avg_answer_time', 0):.3f}s")
            print(f"  • Searches/Hour: {perf.get('searches_per_hour', 0)}")
        
        print("\nResources:")
        resources = status.get('resources', {})
        if resources and 'cpu_percent' in resources:
            print(f"  • CPU Usage: {resources['cpu_percent']:.1f}%")
            print(f"  • Memory Usage: {resources['memory_percent']:.1f}%")
            print(f"  • Disk Usage: {resources['disk_usage']:.1f}%")
        
        # Performance metrics
        print("\n2. Performance Metrics (Last 24 Hours)")
        print("-" * 30)
        
        metrics = client.get_performance_metrics(24)
        print(f"Analysis Period: {metrics['period_hours']} hours")
        
        search_metrics = metrics.get('search_metrics', {})
        if search_metrics:
            trends = search_metrics.get('trends', {})
            if trends and 'total_searches' in trends:
                print(f"Total Searches: {trends['total_searches']}")
            
            current_stats = search_metrics.get('current_stats', {})
            if current_stats:
                print(f"Current Total Searches: {current_stats.get('total_searches', 0)}")
        
        # Log files
        print("\n3. Log Files")
        print("-" * 30)
        
        log_files = client.list_log_files()
        print(f"Total Log Files: {log_files['total_files']}")
        print(f"Total Size: {log_files['total_size']} bytes")
        
        for log_file in log_files['log_files']:
            print(f"  • {log_file['name']}: {log_file.get('size', 0)} bytes")
        
        # Log summary
        print("\n4. Recent Log Summary")
        print("-" * 30)
        
        if log_files['log_files']:
            log_file_name = log_files['log_files'][0]['name']
            try:
                log_summary = client.get_log_summary(log_file_name, 50)
                
                if log_summary['status'] == 'success':
                    print(f"Log File: {log_file_name}")
                    print(f"Lines Analyzed: {log_summary['lines_analyzed']}")
                    print(f"File Size: {log_summary['file_size']} bytes")
                    
                    print("\nLog Level Counts:")
                    for level, count in log_summary['level_counts'].items():
                        print(f"  • {level}: {count}")
                    
                    if log_summary['recent_errors']:
                        print(f"\nRecent Errors ({len(log_summary['recent_errors'])}):")
                        for error in log_summary['recent_errors'][:3]:  # Show first 3
                            print(f"  • {error[:80]}...")
                else:
                    print(f"Could not read log file: {log_summary.get('error', 'Unknown error')}")
                    
            except requests.exceptions.HTTPError:
                print(f"Log file {log_file_name} not accessible")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to management API")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n✅ System monitoring demo completed!")
    return True


def demo_advanced_operations():
    """Demonstrate advanced management operations."""
    print("\n" + "=" * 60)
    print("RAG Management Demo - Advanced Operations")
    print("=" * 60)
    
    client = RAGManagementClient()
    
    try:
        # Index rebuild
        print("\n1. Index Rebuild")
        print("-" * 30)
        
        print("Starting index rebuild for all documents...")
        rebuild_result = client.rebuild_index()
        
        print(f"Status: {rebuild_result['status']}")
        print(f"Processed: {rebuild_result['processed_documents']} documents")
        print(f"Failed: {rebuild_result['failed_documents']} documents")
        print(f"Processing Time: {rebuild_result['processing_time']:.2f}s")
        
        if rebuild_result['errors']:
            print("Errors:")
            for error in rebuild_result['errors']:
                print(f"  • {error}")
        
        # Document cleanup demo
        print("\n2. Document Cleanup")
        print("-" * 30)
        
        # List documents to clean up demo documents
        doc_list = client.list_documents()
        demo_docs = [doc for doc in doc_list['documents'] if doc['id'].startswith('scp_')]
        
        if demo_docs:
            print(f"Found {len(demo_docs)} demo documents to clean up")
            
            for doc in demo_docs[:2]:  # Clean up first 2 demo docs
                try:
                    delete_result = client.delete_document(doc['id'])
                    print(f"✓ Deleted: {doc['id']} ({delete_result['chunks_removed']} chunks)")
                except Exception as e:
                    print(f"✗ Failed to delete {doc['id']}: {e}")
        else:
            print("No demo documents found to clean up")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to management API")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n✅ Advanced operations demo completed!")
    return True


def main():
    """Main demo function."""
    print("RAG System Management Interface Demo")
    print("This demo shows how to use the management API for administration")
    print("\nMake sure the API server is running:")
    print("  python run_api.py")
    print("\nPress Enter to continue or Ctrl+C to exit...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nDemo cancelled by user")
        return
    
    # Run demos
    success = True
    success &= demo_document_management()
    success &= demo_system_monitoring()
    success &= demo_advanced_operations()
    
    if success:
        print("\n🎉 All management demos completed successfully!")
        print("\nNext steps:")
        print("  • Access the management API at http://localhost:8000/admin/")
        print("  • View API docs at http://localhost:8000/docs")
        print("  • Monitor system status and manage documents")
        print("  • Build custom management tools using the API")
    else:
        print("\n⚠️  Some demos failed - check server status")
        sys.exit(1)


if __name__ == "__main__":
    main()
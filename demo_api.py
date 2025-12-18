#!/usr/bin/env python3
"""
Demo script for the RAG API.

This script demonstrates how to use the REST API endpoints
for question-answering and document processing.
"""

import requests
import json
import time
import sys
from typing import Dict, Any


class RAGAPIClient:
    """Simple client for the RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API client."""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def ask_question(self, question: str, max_results: int = 5, min_confidence: float = 0.0) -> Dict[str, Any]:
        """Ask a question to the RAG system."""
        data = {
            "question": question,
            "max_results": max_results,
            "min_confidence": min_confidence
        }
        response = self.session.post(f"{self.base_url}/ask", json=data)
        response.raise_for_status()
        return response.json()
    
    def process_documents(self, url: str = None, force_refresh: bool = False) -> Dict[str, Any]:
        """Trigger document processing."""
        data = {
            "force_refresh": force_refresh
        }
        if url:
            data["url"] = url
        
        response = self.session.post(f"{self.base_url}/documents/process", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_document_status(self) -> Dict[str, Any]:
        """Get document processing status."""
        response = self.session.get(f"{self.base_url}/documents/status")
        response.raise_for_status()
        return response.json()


def demo_basic_functionality():
    """Demonstrate basic API functionality."""
    print("=" * 60)
    print("RAG API Demo - Basic Functionality")
    print("=" * 60)
    
    client = RAGAPIClient()
    
    try:
        # Health check
        print("\n1. Health Check")
        print("-" * 30)
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Services: {json.dumps(health['services'], indent=2)}")
        
        # Ask questions
        print("\n2. Question Answering")
        print("-" * 30)
        
        questions = [
            "九州大学のスーパーコンピュータの使い方を教えてください",
            "アカウント申請の方法は？",
            "ジョブの実行手順について",
            "エラーが発生した場合の対処法"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}: {question}")
            try:
                answer = client.ask_question(question, max_results=3, min_confidence=0.1)
                print(f"Answer: {answer['answer'][:100]}...")
                print(f"Confidence: {answer['confidence']:.2f}")
                print(f"Sources: {len(answer['sources'])}")
                print(f"Processing time: {answer['processing_time']:.2f}s")
            except requests.exceptions.HTTPError as e:
                print(f"Error: {e}")
                if e.response.status_code == 422:
                    print("Low confidence answer - try adjusting min_confidence parameter")
        
        # Document processing
        print("\n3. Document Processing")
        print("-" * 30)
        
        print("Triggering document processing...")
        process_result = client.process_documents()
        print(f"Status: {process_result['status']}")
        print(f"Message: {process_result['message']}")
        
        # Document status
        print("\nChecking document status...")
        status = client.get_document_status()
        print(f"Status: {status['status']}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        print("Make sure the server is running with: python run_api.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n✅ Demo completed successfully!")
    return True


def demo_error_handling():
    """Demonstrate error handling."""
    print("\n" + "=" * 60)
    print("RAG API Demo - Error Handling")
    print("=" * 60)
    
    client = RAGAPIClient()
    
    try:
        # Test invalid requests
        print("\n1. Invalid Question (Empty)")
        print("-" * 30)
        try:
            client.ask_question("")
        except requests.exceptions.HTTPError as e:
            print(f"Expected error: {e.response.status_code} - {e.response.json()['detail'][0]['msg']}")
        
        print("\n2. Invalid Question (Too Long)")
        print("-" * 30)
        try:
            client.ask_question("x" * 1001)
        except requests.exceptions.HTTPError as e:
            print(f"Expected error: {e.response.status_code} - {e.response.json()['detail'][0]['msg']}")
        
        print("\n3. Invalid Parameters")
        print("-" * 30)
        try:
            client.ask_question("テスト", max_results=0)
        except requests.exceptions.HTTPError as e:
            print(f"Expected error: {e.response.status_code} - {e.response.json()['detail'][0]['msg']}")
        
        print("\n4. High Confidence Threshold")
        print("-" * 30)
        try:
            answer = client.ask_question("不明な質問", min_confidence=0.9)
            print(f"Unexpected success: {answer}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 422:
                print(f"Expected low confidence error: {e.response.json()['detail']}")
            else:
                print(f"Other error: {e}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    print("\n✅ Error handling demo completed!")
    return True


def demo_api_documentation():
    """Show API documentation endpoints."""
    print("\n" + "=" * 60)
    print("RAG API Demo - Documentation")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    print(f"\nAPI Documentation is available at:")
    print(f"  • Interactive docs: {base_url}/docs")
    print(f"  • ReDoc: {base_url}/redoc")
    print(f"  • OpenAPI schema: {base_url}/openapi.json")
    
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            info = response.json()
            print(f"\nAPI Info:")
            print(f"  • Name: {info['message']}")
            print(f"  • Version: {info['version']}")
    except:
        print("\n❌ Could not fetch API info - server may not be running")


def main():
    """Main demo function."""
    print("RAG System API Demo")
    print("This demo shows how to interact with the RAG API")
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
    success &= demo_basic_functionality()
    success &= demo_error_handling()
    demo_api_documentation()
    
    if success:
        print("\n🎉 All demos completed successfully!")
        print("\nNext steps:")
        print("  • Try the interactive docs at http://localhost:8000/docs")
        print("  • Explore the API endpoints")
        print("  • Build your own client applications")
    else:
        print("\n⚠️  Some demos failed - check server status")
        sys.exit(1)


if __name__ == "__main__":
    main()
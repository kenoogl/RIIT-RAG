"""
Search logging service for comprehensive search analytics and monitoring.

This module provides detailed logging and analytics for search operations,
including query analysis, result tracking, and performance monitoring.
"""

import logging
import json
import time
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque

from ..models.interfaces import LoggingInterface
from ..models.core import SearchResult, Answer
from ..utils.config import get_config


class SearchLoggingService(LoggingInterface):
    """
    Comprehensive search logging and analytics service.
    
    This service provides detailed logging for search operations, including:
    - Search query logging with metadata
    - Result quality tracking
    - Performance monitoring
    - Search analytics and reporting
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the search logging service.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        logging_config = self.config.get("logging", {})
        
        # Configure logging settings
        self.log_level = logging_config.get("level", "INFO")
        self.log_file_path = logging_config.get("file_path", "./logs/search.log")
        self.enable_file_logging = logging_config.get("enable_file", True)
        self.enable_console_logging = logging_config.get("enable_console", True)
        
        # Analytics settings
        performance_config = self.config.get("performance", {})
        self.enable_analytics = performance_config.get("enable_metrics", True)
        self.max_log_entries = performance_config.get("max_log_entries", 10000)
        
        # Initialize logging
        self.logger = self._setup_logger()
        
        # Analytics storage
        self.search_logs = deque(maxlen=self.max_log_entries)
        self.answer_logs = deque(maxlen=self.max_log_entries)
        self.document_logs = deque(maxlen=self.max_log_entries)
        
        # Performance metrics
        self.metrics = {
            "search_count": 0,
            "answer_count": 0,
            "document_processing_count": 0,
            "total_search_time": 0.0,
            "total_answer_time": 0.0,
            "avg_search_time": 0.0,
            "avg_answer_time": 0.0,
            "last_reset": datetime.now()
        }
        
        # Query analytics
        self.query_analytics = {
            "popular_terms": defaultdict(int),
            "query_types": defaultdict(int),
            "result_counts": defaultdict(int),
            "processing_times": []
        }
        
        self.logger.info("Search logging service initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup and configure the logger."""
        logger = logging.getLogger(f"{__name__}.SearchLogging")
        logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if self.enable_file_logging:
            # Ensure log directory exists
            log_path = Path(self.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(self.log_file_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def log_search(self, query: str, results: List[SearchResult], processing_time: float) -> None:
        """
        Log search operation details.
        
        Args:
            query: Search query
            results: Search results
            processing_time: Time taken for search
        """
        try:
            timestamp = datetime.now()
            
            # Create search log entry
            log_entry = {
                "timestamp": timestamp.isoformat(),
                "query": query,
                "query_length": len(query),
                "num_results": len(results),
                "processing_time": processing_time,
                "top_score": results[0].score if results else 0.0,
                "avg_score": sum(r.score for r in results) / len(results) if results else 0.0,
                "result_scores": [r.score for r in results[:5]],  # Top 5 scores
                "chunk_ids": [r.chunk.id for r in results[:5]],  # Top 5 chunk IDs
                "document_ids": [r.document.id if hasattr(r, 'document') and r.document else None for r in results[:5]]
            }
            
            # Store in analytics
            if self.enable_analytics:
                self.search_logs.append(log_entry)
                self._update_search_metrics(query, results, processing_time)
            
            # Log to file/console
            self.logger.info(
                f"SEARCH | Query: '{query[:50]}...' | Results: {len(results)} | "
                f"Time: {processing_time:.3f}s | Top Score: {log_entry['top_score']:.3f}"
            )
            
            # Detailed debug logging
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Search details: {json.dumps(log_entry, ensure_ascii=False, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Failed to log search operation: {e}")
    
    def log_document_processing(self, document_id: str, status: str, error: Optional[str] = None) -> None:
        """
        Log document processing operation.
        
        Args:
            document_id: ID of processed document
            status: Processing status (success/error)
            error: Error message if processing failed
        """
        try:
            timestamp = datetime.now()
            
            # Create document processing log entry
            log_entry = {
                "timestamp": timestamp.isoformat(),
                "document_id": document_id,
                "status": status,
                "error": error
            }
            
            # Store in analytics
            if self.enable_analytics:
                self.document_logs.append(log_entry)
                self.metrics["document_processing_count"] += 1
            
            # Log to file/console
            if status == "success":
                self.logger.info(f"DOCUMENT | ID: {document_id} | Status: {status}")
            else:
                self.logger.error(f"DOCUMENT | ID: {document_id} | Status: {status} | Error: {error}")
            
            # Detailed debug logging
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Document processing details: {json.dumps(log_entry, ensure_ascii=False, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Failed to log document processing: {e}")
    
    def log_answer_generation(self, query: str, answer: Answer, chunks_used: List[str]) -> None:
        """
        Log answer generation details.
        
        Args:
            query: User query
            answer: Generated answer
            chunks_used: List of chunk IDs used for generation
        """
        try:
            timestamp = datetime.now()
            
            # Create answer generation log entry
            log_entry = {
                "timestamp": timestamp.isoformat(),
                "query": query,
                "answer_length": len(answer.text),
                "confidence": answer.confidence,
                "processing_time": answer.processing_time,
                "num_sources": len(answer.sources),
                "chunks_used": chunks_used,
                "source_info": answer.sources[:3]  # Top 3 sources
            }
            
            # Store in analytics
            if self.enable_analytics:
                self.answer_logs.append(log_entry)
                self._update_answer_metrics(answer)
            
            # Log to file/console
            self.logger.info(
                f"ANSWER | Query: '{query[:50]}...' | Length: {len(answer.text)} | "
                f"Confidence: {answer.confidence:.3f} | Time: {answer.processing_time:.3f}s | "
                f"Sources: {len(answer.sources)}"
            )
            
            # Detailed debug logging
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Answer generation details: {json.dumps(log_entry, ensure_ascii=False, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Failed to log answer generation: {e}")
    
    def _update_search_metrics(self, query: str, results: List[SearchResult], processing_time: float) -> None:
        """Update search-related metrics and analytics."""
        # Update basic metrics
        self.metrics["search_count"] += 1
        self.metrics["total_search_time"] += processing_time
        self.metrics["avg_search_time"] = self.metrics["total_search_time"] / self.metrics["search_count"]
        
        # Update query analytics - extract meaningful terms from Japanese/English text
        import re
        
        # For Japanese text, we need a different approach since there are no spaces
        # Extract individual kanji/katakana words and common terms
        query_lower = query.lower()
        
        # First, try to extract English words (space-separated)
        english_words = re.findall(r'[a-zA-Z]+', query_lower)
        
        # For Japanese, extract meaningful character sequences
        # Extract kanji sequences (Chinese characters)
        kanji_words = re.findall(r'[\u4E00-\u9FFF]+', query_lower)
        
        # Extract katakana sequences (often used for technical terms)
        katakana_words = re.findall(r'[\u30A0-\u30FF]+', query_lower)
        
        # Combine all extracted words
        all_words = english_words + kanji_words + katakana_words
        
        # Also add the full query as a term for exact matching
        if len(query_lower.strip()) > 0:
            all_words.append(query_lower.strip())
        
        for word in all_words:
            if len(word) > 1:  # Only count meaningful words
                self.query_analytics["popular_terms"][word] += 1
        
        # Track result counts
        result_count_bucket = self._get_result_count_bucket(len(results))
        self.query_analytics["result_counts"][result_count_bucket] += 1
        
        # Track processing times
        self.query_analytics["processing_times"].append(processing_time)
        # Keep only recent processing times
        if len(self.query_analytics["processing_times"]) > 1000:
            self.query_analytics["processing_times"] = self.query_analytics["processing_times"][-1000:]
    
    def _update_answer_metrics(self, answer: Answer) -> None:
        """Update answer generation metrics."""
        self.metrics["answer_count"] += 1
        self.metrics["total_answer_time"] += answer.processing_time
        self.metrics["avg_answer_time"] = self.metrics["total_answer_time"] / self.metrics["answer_count"]
    
    def _get_result_count_bucket(self, count: int) -> str:
        """Get result count bucket for analytics."""
        if count == 0:
            return "0"
        elif count <= 5:
            return "1-5"
        elif count <= 10:
            return "6-10"
        elif count <= 20:
            return "11-20"
        else:
            return "20+"
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive search statistics.
        
        Returns:
            Dictionary containing search statistics
        """
        if not self.enable_analytics:
            return {"analytics_disabled": True}
        
        # Calculate time-based statistics
        now = datetime.now()
        last_hour_searches = [
            log for log in self.search_logs
            if datetime.fromisoformat(log["timestamp"]) > now - timedelta(hours=1)
        ]
        last_day_searches = [
            log for log in self.search_logs
            if datetime.fromisoformat(log["timestamp"]) > now - timedelta(days=1)
        ]
        
        # Calculate performance statistics
        processing_times = self.query_analytics["processing_times"]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        
        return {
            "overview": {
                "total_searches": self.metrics["search_count"],
                "total_answers": self.metrics["answer_count"],
                "total_documents_processed": self.metrics["document_processing_count"],
                "avg_search_time": self.metrics["avg_search_time"],
                "avg_answer_time": self.metrics["avg_answer_time"],
                "last_reset": self.metrics["last_reset"].isoformat()
            },
            "recent_activity": {
                "searches_last_hour": len(last_hour_searches),
                "searches_last_day": len(last_day_searches),
                "avg_processing_time": avg_processing_time
            },
            "query_analytics": {
                "popular_terms": dict(sorted(
                    self.query_analytics["popular_terms"].items(),
                    key=lambda x: x[1], reverse=True
                )[:10]),
                "result_distribution": dict(self.query_analytics["result_counts"])
            },
            "performance": {
                "min_processing_time": min(processing_times) if processing_times else 0.0,
                "max_processing_time": max(processing_times) if processing_times else 0.0,
                "avg_processing_time": avg_processing_time,
                "total_log_entries": len(self.search_logs)
            }
        }
    
    def get_recent_searches(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent search logs.
        
        Args:
            limit: Maximum number of recent searches to return
            
        Returns:
            List of recent search log entries
        """
        if not self.enable_analytics:
            return []
        
        return list(self.search_logs)[-limit:] if self.search_logs else []
    
    def get_search_trends(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get search trends over the specified time period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary containing trend analysis
        """
        if not self.enable_analytics:
            return {"analytics_disabled": True}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_searches = [
            log for log in self.search_logs
            if datetime.fromisoformat(log["timestamp"]) > cutoff_time
        ]
        
        if not recent_searches:
            return {"no_data": True, "period_hours": hours}
        
        # Analyze trends
        hourly_counts = defaultdict(int)
        query_lengths = []
        result_counts = []
        processing_times = []
        
        for log in recent_searches:
            timestamp = datetime.fromisoformat(log["timestamp"])
            hour_key = timestamp.strftime("%Y-%m-%d %H:00")
            hourly_counts[hour_key] += 1
            
            query_lengths.append(log["query_length"])
            result_counts.append(log["num_results"])
            processing_times.append(log["processing_time"])
        
        return {
            "period_hours": hours,
            "total_searches": len(recent_searches),
            "hourly_distribution": dict(hourly_counts),
            "averages": {
                "query_length": sum(query_lengths) / len(query_lengths),
                "result_count": sum(result_counts) / len(result_counts),
                "processing_time": sum(processing_times) / len(processing_times)
            },
            "ranges": {
                "min_query_length": min(query_lengths),
                "max_query_length": max(query_lengths),
                "min_results": min(result_counts),
                "max_results": max(result_counts),
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times)
            }
        }
    
    def export_logs(self, output_path: str, format: str = "json") -> bool:
        """
        Export logs to file.
        
        Args:
            output_path: Path to output file
            format: Export format ("json" or "csv")
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "json":
                export_data = {
                    "metadata": {
                        "export_timestamp": datetime.now().isoformat(),
                        "total_searches": len(self.search_logs),
                        "total_answers": len(self.answer_logs),
                        "total_documents": len(self.document_logs)
                    },
                    "search_logs": list(self.search_logs),
                    "answer_logs": list(self.answer_logs),
                    "document_logs": list(self.document_logs),
                    "metrics": self.metrics,
                    "analytics": self.query_analytics
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
            elif format.lower() == "csv":
                import csv
                
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Write search logs
                    writer.writerow(["Type", "Timestamp", "Query", "Results", "Processing Time", "Top Score"])
                    for log in self.search_logs:
                        writer.writerow([
                            "SEARCH",
                            log["timestamp"],
                            log["query"],
                            log["num_results"],
                            log["processing_time"],
                            log["top_score"]
                        ])
            
            self.logger.info(f"Logs exported to {output_path} in {format} format")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export logs: {e}")
            return False
    
    def reset_metrics(self) -> None:
        """Reset all metrics and analytics."""
        self.search_logs.clear()
        self.answer_logs.clear()
        self.document_logs.clear()
        
        self.metrics = {
            "search_count": 0,
            "answer_count": 0,
            "document_processing_count": 0,
            "total_search_time": 0.0,
            "total_answer_time": 0.0,
            "avg_search_time": 0.0,
            "avg_answer_time": 0.0,
            "last_reset": datetime.now()
        }
        
        self.query_analytics = {
            "popular_terms": defaultdict(int),
            "query_types": defaultdict(int),
            "result_counts": defaultdict(int),
            "processing_times": []
        }
        
        self.logger.info("Search metrics and analytics reset")


def create_search_logging_service(config_path: Optional[str] = None) -> SearchLoggingService:
    """
    Factory function to create a search logging service.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured search logging service instance
    """
    return SearchLoggingService(config_path)
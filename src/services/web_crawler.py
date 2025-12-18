"""
Web crawler service for the RAG system.

This module provides web crawling functionality to automatically collect
documents from specified websites, with robust error handling and
Japanese content filtering.
"""

import os
import time
import logging
import requests
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
from typing import List, Set, Optional, Dict, Any, Tuple
from pathlib import Path
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from bs4 import BeautifulSoup, Comment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..models.interfaces import WebCrawlerInterface
from ..models.core import Document
from ..models.exceptions import CrawlingError, ExtractionError
from ..utils.config import get_config


@dataclass
class CrawlResult:
    """Result of a crawling operation."""
    url: str
    title: str
    content: str
    status_code: int
    content_type: str
    last_modified: Optional[datetime]
    content_hash: str
    links: List[str]
    error: Optional[str] = None


class RobustWebCrawler(WebCrawlerInterface):
    """
    Robust web crawler implementation with error handling and rate limiting.
    
    This class provides comprehensive web crawling functionality with support for:
    - Respectful crawling with delays and robots.txt compliance
    - Robust error handling and retry mechanisms
    - Content change detection
    - Japanese content filtering
    """
    
    def __init__(
        self,
        target_url: Optional[str] = None,
        max_depth: int = 3,
        delay_between_requests: float = 1.0,
        timeout: int = 30,
        user_agent: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize the web crawler.
        
        Args:
            target_url: Base URL to crawl
            max_depth: Maximum crawling depth
            delay_between_requests: Delay between requests in seconds
            timeout: Request timeout in seconds
            user_agent: User agent string
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retry attempts
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        config = get_config()
        crawler_config = config.get('crawler', {})
        
        self.target_url = target_url or crawler_config.get('target_url', 'https://www.cc.kyushu-u.ac.jp/scp/')
        self.max_depth = max_depth or crawler_config.get('max_depth', 3)
        self.delay_between_requests = delay_between_requests or crawler_config.get('delay_between_requests', 1.0)
        self.timeout = timeout or crawler_config.get('timeout', 30)
        self.user_agent = user_agent or crawler_config.get('user_agent', 'RAG-System-Crawler/1.0')
        self.retry_attempts = retry_attempts or crawler_config.get('retry_attempts', 3)
        self.retry_delay = retry_delay or crawler_config.get('retry_delay', 2.0)
        
        # Initialize session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.retry_attempts,
            backoff_factor=self.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ja,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # Crawling state
        self.visited_urls: Set[str] = set()
        self.crawled_documents: List[Document] = []
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.last_request_time = 0.0
        
        # Content change tracking
        self.content_hashes: Dict[str, str] = {}
        self.load_content_hashes()
        
        self.logger.info(f"Initialized web crawler for {self.target_url}")
    
    def crawl_website(self, url: Optional[str] = None) -> List[Document]:
        """
        Crawl a website and extract documents.
        
        Args:
            url: The URL to crawl (uses target_url if not provided)
            
        Returns:
            List of extracted documents
            
        Raises:
            CrawlingError: If crawling fails
        """
        start_url = url or self.target_url
        
        try:
            self.logger.info(f"Starting crawl of {start_url}")
            
            # Reset state
            self.visited_urls.clear()
            self.crawled_documents.clear()
            
            # Start crawling
            self._crawl_recursive(start_url, depth=0)
            
            self.logger.info(f"Crawling completed. Found {len(self.crawled_documents)} documents")
            
            # Save content hashes for change detection
            self.save_content_hashes()
            
            return self.crawled_documents
            
        except Exception as e:
            self.logger.error(f"Crawling failed: {e}")
            raise CrawlingError(f"Failed to crawl website {start_url}: {e}")
    
    def _crawl_recursive(self, url: str, depth: int) -> None:
        """
        Recursively crawl a website.
        
        Args:
            url: URL to crawl
            depth: Current crawling depth
        """
        if depth > self.max_depth:
            return
        
        if url in self.visited_urls:
            return
        
        # Normalize URL
        normalized_url = self._normalize_url(url)
        if not normalized_url or not self._is_valid_url(normalized_url):
            return
        
        self.visited_urls.add(normalized_url)
        
        try:
            # Check robots.txt
            if not self._can_fetch(normalized_url):
                self.logger.info(f"Robots.txt disallows crawling: {normalized_url}")
                return
            
            # Rate limiting
            self._respect_rate_limit()
            
            # Fetch page
            result = self._fetch_page(normalized_url)
            if result.error:
                self.logger.warning(f"Failed to fetch {normalized_url}: {result.error}")
                return
            
            # Check for content changes
            if self._has_content_changed(normalized_url, result.content_hash):
                # Extract document
                document = self._create_document_from_result(result)
                if document:
                    self.crawled_documents.append(document)
                    self.logger.info(f"Extracted document: {document.title}")
                
                # Update content hash
                self.content_hashes[normalized_url] = result.content_hash
            else:
                self.logger.debug(f"Content unchanged, skipping: {normalized_url}")
            
            # Extract and crawl links
            if depth < self.max_depth:
                for link in result.links:
                    if self._should_crawl_link(link, normalized_url):
                        self._crawl_recursive(link, depth + 1)
        
        except Exception as e:
            self.logger.error(f"Error crawling {normalized_url}: {e}")
    
    def _fetch_page(self, url: str) -> CrawlResult:
        """
        Fetch a single web page.
        
        Args:
            url: URL to fetch
            
        Returns:
            CrawlResult object
        """
        try:
            self.logger.debug(f"Fetching: {url}")
            
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup, url)
            
            # Extract main content
            content = self.extract_text_from_html(response.text)
            
            # Calculate content hash
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            # Extract links
            links = self._extract_links(soup, url)
            
            # Get last modified date
            last_modified = self._parse_last_modified(response.headers.get('Last-Modified'))
            
            return CrawlResult(
                url=url,
                title=title,
                content=content,
                status_code=response.status_code,
                content_type=response.headers.get('Content-Type', ''),
                last_modified=last_modified,
                content_hash=content_hash,
                links=links
            )
            
        except requests.RequestException as e:
            return CrawlResult(
                url=url,
                title="",
                content="",
                status_code=0,
                content_type="",
                last_modified=None,
                content_hash="",
                links=[],
                error=str(e)
            )
        except Exception as e:
            return CrawlResult(
                url=url,
                title="",
                content="",
                status_code=0,
                content_type="",
                last_modified=None,
                content_hash="",
                links=[],
                error=f"Unexpected error: {e}"
            )
    
    def extract_text_from_html(self, html: str) -> str:
        """
        Extract text content from HTML.
        
        Args:
            html: Raw HTML content
            
        Returns:
            Extracted text content
            
        Raises:
            ExtractionError: If text extraction fails
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Find main content areas (prioritize main content)
            main_content = None
            
            # Try to find main content containers
            for selector in ['main', 'article', '.content', '#content', '.main', '#main']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # If no main content found, use body
            if not main_content:
                main_content = soup.find('body') or soup
            
            # Extract text
            text = main_content.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            text = ' '.join(text.split())
            
            return text
            
        except Exception as e:
            self.logger.error(f"Failed to extract text from HTML: {e}")
            raise ExtractionError(f"Failed to extract text from HTML: {e}")
    
    def filter_japanese_content(self, text: str) -> str:
        """
        Filter and extract Japanese content from text.
        
        Args:
            text: Input text content
            
        Returns:
            Filtered Japanese content
        """
        if not text:
            return ""
        
        # Import text processor for Japanese filtering
        try:
            from .text_processor import JapaneseTextProcessor
            processor = JapaneseTextProcessor()
            return processor.extract_japanese_content(text)
        except ImportError:
            # Fallback: simple Japanese character detection
            import re
            
            # Japanese character ranges
            japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+')
            
            # Split into sentences and filter
            sentences = re.split(r'[。！？\n]', text)
            japanese_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and japanese_pattern.search(sentence):
                    japanese_sentences.append(sentence)
            
            return '。'.join(japanese_sentences) + ('。' if japanese_sentences else '')
    
    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Extract page title."""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Fallback to h1
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        # Fallback to URL path
        parsed_url = urlparse(url)
        return parsed_url.path.split('/')[-1] or parsed_url.netloc
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from the page."""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            normalized_url = self._normalize_url(absolute_url)
            
            if normalized_url and self._is_valid_url(normalized_url):
                links.append(normalized_url)
        
        return list(set(links))  # Remove duplicates
    
    def _normalize_url(self, url: str) -> Optional[str]:
        """Normalize URL by removing fragments and query parameters."""
        try:
            parsed = urlparse(url)
            # Remove fragment and some query parameters
            normalized = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                '',  # Remove query for now
                ''   # Remove fragment
            ))
            return normalized
        except Exception:
            return None
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid for crawling."""
        try:
            parsed = urlparse(url)
            
            # Must be HTTP/HTTPS
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Must be same domain as target
            target_domain = urlparse(self.target_url).netloc
            if parsed.netloc != target_domain:
                return False
            
            # Skip certain file types
            skip_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
                             '.zip', '.tar', '.gz', '.jpg', '.jpeg', '.png', '.gif', '.svg']
            
            path_lower = parsed.path.lower()
            if any(path_lower.endswith(ext) for ext in skip_extensions):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _should_crawl_link(self, link: str, current_url: str) -> bool:
        """Determine if a link should be crawled."""
        if link in self.visited_urls:
            return False
        
        if not self._is_valid_url(link):
            return False
        
        # Skip if too many URLs already queued
        if len(self.visited_urls) > 1000:
            return False
        
        return True
    
    def _can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            if base_url not in self.robots_cache:
                robots_url = urljoin(base_url, '/robots.txt')
                rp = RobotFileParser()
                rp.set_url(robots_url)
                try:
                    rp.read()
                    self.robots_cache[base_url] = rp
                except Exception:
                    # If robots.txt can't be read, assume crawling is allowed
                    return True
            
            rp = self.robots_cache[base_url]
            return rp.can_fetch(self.user_agent, url)
            
        except Exception:
            # If there's any error, assume crawling is allowed
            return True
    
    def _respect_rate_limit(self) -> None:
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.delay_between_requests:
            sleep_time = self.delay_between_requests - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _has_content_changed(self, url: str, new_hash: str) -> bool:
        """Check if content has changed since last crawl."""
        old_hash = self.content_hashes.get(url)
        return old_hash != new_hash
    
    def _parse_last_modified(self, last_modified_header: Optional[str]) -> Optional[datetime]:
        """Parse Last-Modified header."""
        if not last_modified_header:
            return None
        
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(last_modified_header)
        except Exception:
            return None
    
    def _create_document_from_result(self, result: CrawlResult) -> Optional[Document]:
        """Create a Document object from crawl result."""
        try:
            # Filter Japanese content
            japanese_content = self.filter_japanese_content(result.content)
            
            if not japanese_content or len(japanese_content.strip()) < 100:
                self.logger.debug(f"Insufficient Japanese content in {result.url}")
                return None
            
            # Create document
            document = Document(
                id=self._generate_document_id(result.url),
                url=result.url,
                title=result.title,
                content=japanese_content,
                language='ja',
                created_at=datetime.now(),
                updated_at=result.last_modified or datetime.now()
            )
            
            return document
            
        except Exception as e:
            self.logger.error(f"Failed to create document from {result.url}: {e}")
            return None
    
    def _generate_document_id(self, url: str) -> str:
        """Generate a unique document ID from URL."""
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    def load_content_hashes(self) -> None:
        """Load content hashes from storage."""
        try:
            hash_file = Path("data/crawler_hashes.json")
            if hash_file.exists():
                with open(hash_file, 'r', encoding='utf-8') as f:
                    self.content_hashes = json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load content hashes: {e}")
            self.content_hashes = {}
    
    def save_content_hashes(self) -> None:
        """Save content hashes to storage."""
        try:
            hash_file = Path("data/crawler_hashes.json")
            hash_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(hash_file, 'w', encoding='utf-8') as f:
                json.dump(self.content_hashes, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save content hashes: {e}")
    
    def get_crawl_stats(self) -> Dict[str, Any]:
        """Get crawling statistics."""
        return {
            'target_url': self.target_url,
            'visited_urls': len(self.visited_urls),
            'documents_found': len(self.crawled_documents),
            'content_hashes_tracked': len(self.content_hashes),
            'max_depth': self.max_depth,
            'delay_between_requests': self.delay_between_requests
        }


class WebsiteChangeDetector:
    """
    Detects changes in website structure and content.
    
    This class monitors website changes to adapt crawling strategies
    and detect when content has been updated.
    """
    
    def __init__(self, storage_path: str = "data/website_structure"):
        """
        Initialize the change detector.
        
        Args:
            storage_path: Path to store website structure data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def detect_structure_changes(self, url: str, current_links: List[str]) -> Dict[str, Any]:
        """
        Detect changes in website structure.
        
        Args:
            url: Base URL of the website
            current_links: Current list of links found
            
        Returns:
            Dictionary containing change information
        """
        try:
            # Load previous structure
            structure_file = self.storage_path / f"{self._url_to_filename(url)}_structure.json"
            previous_links = []
            
            if structure_file.exists():
                with open(structure_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    previous_links = data.get('links', [])
            
            # Compare structures
            current_set = set(current_links)
            previous_set = set(previous_links)
            
            added_links = current_set - previous_set
            removed_links = previous_set - current_set
            
            changes = {
                'url': url,
                'timestamp': datetime.now().isoformat(),
                'total_links': len(current_links),
                'previous_links': len(previous_links),
                'added_links': list(added_links),
                'removed_links': list(removed_links),
                'has_changes': len(added_links) > 0 or len(removed_links) > 0
            }
            
            # Save current structure
            structure_data = {
                'url': url,
                'timestamp': datetime.now().isoformat(),
                'links': current_links
            }
            
            with open(structure_file, 'w', encoding='utf-8') as f:
                json.dump(structure_data, f, ensure_ascii=False, indent=2)
            
            if changes['has_changes']:
                self.logger.info(f"Structure changes detected for {url}: "
                               f"+{len(added_links)} -{len(removed_links)} links")
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Failed to detect structure changes: {e}")
            return {
                'url': url,
                'timestamp': datetime.now().isoformat(),
                'has_changes': False,
                'error': str(e)
            }
    
    def _url_to_filename(self, url: str) -> str:
        """Convert URL to safe filename."""
        return hashlib.md5(url.encode('utf-8')).hexdigest()


# Convenience functions
def create_web_crawler(**kwargs) -> RobustWebCrawler:
    """Create a web crawler instance with configuration."""
    return RobustWebCrawler(**kwargs)


def crawl_kyushu_university_scp() -> List[Document]:
    """Crawl the Kyushu University SCP website."""
    crawler = RobustWebCrawler()
    return crawler.crawl_website()
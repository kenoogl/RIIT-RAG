"""
Text processing service for the RAG system.

This module provides text preprocessing, chunking, and Japanese text
normalization functionality for document processing.
"""

import re
import logging
from typing import List, Optional, Dict, Any, Tuple
import unicodedata
from pathlib import Path

from ..models.core import Document, Chunk
from ..models.exceptions import ProcessingError
from ..utils.config import get_config


class JapaneseTextProcessor:
    """
    Japanese text processing utilities.
    
    This class provides specialized text processing functions for Japanese content,
    including normalization, cleaning, and language detection.
    """
    
    def __init__(self):
        """Initialize the Japanese text processor."""
        self.logger = logging.getLogger(__name__)
        
        # Japanese character ranges (Unicode)
        self.hiragana_pattern = re.compile(r'[\u3040-\u309F]')
        self.katakana_pattern = re.compile(r'[\u30A0-\u30FF]')
        self.kanji_pattern = re.compile(r'[\u4E00-\u9FAF]')
        self.japanese_punctuation = re.compile(r'[\u3000-\u303F]')
        
        # Common Japanese particles and words for language detection
        self.japanese_indicators = [
            'の', 'は', 'が', 'を', 'に', 'で', 'と', 'から', 'まで', 'より',
            'です', 'である', 'だ', 'である', 'します', 'ます', 'した', 'でした',
            'について', 'により', 'として', 'において', 'に関して'
        ]
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Japanese text.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Unicode normalization (NFKC for Japanese)
        text = unicodedata.normalize('NFKC', text)
        
        # Convert full-width ASCII to half-width
        text = text.translate(str.maketrans(
            '０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ',
            '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        ))
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters and formatting.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove HTML tags if any remain
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://[^\s]+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[。！？]{2,}', '。', text)
        text = re.sub(r'[、，]{2,}', '、', text)
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def is_japanese_text(self, text: str, threshold: float = 0.3) -> bool:
        """
        Determine if text is primarily Japanese.
        
        Args:
            text: Input text
            threshold: Minimum ratio of Japanese characters required
            
        Returns:
            True if text is primarily Japanese, False otherwise
        """
        if not text:
            return False
        
        # Count Japanese characters
        japanese_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isspace() or char in '.,!?;:()[]{}':
                continue
            
            total_chars += 1
            
            if (self.hiragana_pattern.match(char) or 
                self.katakana_pattern.match(char) or 
                self.kanji_pattern.match(char) or
                self.japanese_punctuation.match(char)):
                japanese_chars += 1
        
        if total_chars == 0:
            return False
        
        japanese_ratio = japanese_chars / total_chars
        
        # Also check for common Japanese words
        has_japanese_words = any(indicator in text for indicator in self.japanese_indicators)
        
        return japanese_ratio >= threshold or has_japanese_words
    
    def extract_japanese_content(self, text: str) -> str:
        """
        Extract Japanese content from mixed-language text.
        
        Args:
            text: Input text
            
        Returns:
            Extracted Japanese content
        """
        if not text:
            return ""
        
        # Split text into sentences
        sentences = re.split(r'[。！？\n]', text)
        
        japanese_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and self.is_japanese_text(sentence, threshold=0.2):
                japanese_sentences.append(sentence)
        
        return '。'.join(japanese_sentences) + ('。' if japanese_sentences else '')


class TextChunker:
    """
    Text chunking utilities for splitting documents into manageable pieces.
    
    This class provides intelligent text chunking that respects sentence boundaries
    and maintains context for Japanese text.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size for a chunk to be valid
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.logger = logging.getLogger(__name__)
        
        # Japanese sentence ending patterns
        self.sentence_endings = re.compile(r'[。！？]')
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, respecting Japanese punctuation.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Split on Japanese sentence endings
        sentences = self.sentence_endings.split(text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence:
                # Add back the punctuation (except for the last sentence)
                if i < len(sentences) - 1:
                    # Find the original punctuation
                    original_pos = text.find(sentence) + len(sentence)
                    if original_pos < len(text):
                        punct = text[original_pos]
                        if punct in '。！？':
                            sentence += punct
                
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def create_chunks(self, text: str) -> List[str]:
        """
        Create text chunks from input text.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        if len(text) <= self.chunk_size:
            return [text]
        
        sentences = self.split_into_sentences(text)
        if not sentences:
            # Fallback to character-based chunking
            return self._create_character_chunks(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk
                if len(sentence) > self.chunk_size:
                    # Split long sentence
                    sentence_chunks = self._create_character_chunks(sentence)
                    chunks.extend(sentence_chunks[:-1])  # Add all but last
                    current_chunk = sentence_chunks[-1] if sentence_chunks else ""
                else:
                    current_chunk = sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        # Add overlap between chunks
        if len(chunks) > 1:
            chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _create_character_chunks(self, text: str) -> List[str]:
        """
        Create chunks based on character count (fallback method).
        
        Args:
            text: Input text
            
        Returns:
            List of character-based chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Look for space or Japanese punctuation near the end
                for i in range(end, max(start + self.min_chunk_size, end - 50), -1):
                    if text[i] in ' 、。！？':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Add overlap between consecutive chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            List of chunks with overlap
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]
            
            # Get overlap from previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            
            # Add overlap to current chunk
            overlapped_chunk = overlap_text + " " + current_chunk
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks


class DocumentProcessor:
    """
    High-level document processor that combines text processing and chunking.
    
    This class provides a complete document processing pipeline for Japanese text.
    """
    
    def __init__(self):
        """Initialize the document processor."""
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        config = get_config()
        doc_config = config.get('document_processing', {})
        
        self.text_processor = JapaneseTextProcessor()
        self.chunker = TextChunker(
            chunk_size=doc_config.get('chunk_size', 512),
            chunk_overlap=doc_config.get('chunk_overlap', 50),
            min_chunk_size=doc_config.get('min_text_length', 100)
        )
        
        self.language_filter = doc_config.get('language_filter', 'ja')
        self.min_text_length = doc_config.get('min_text_length', 100)
        self.max_text_length = doc_config.get('max_text_length', 10000)
    
    def process_document(self, document: Document) -> List[Chunk]:
        """
        Process a document into chunks.
        
        Args:
            document: Document to process
            
        Returns:
            List of processed chunks
            
        Raises:
            ProcessingError: If document processing fails
        """
        try:
            self.logger.info(f"Processing document: {document.id}")
            
            # Validate document
            if not document.content:
                self.logger.warning(f"Document {document.id} has no content")
                return []
            
            # Filter by language if specified
            if self.language_filter == 'ja':
                if not self.text_processor.is_japanese_text(document.content):
                    self.logger.info(f"Document {document.id} is not Japanese, skipping")
                    return []
                
                # Extract Japanese content
                content = self.text_processor.extract_japanese_content(document.content)
            else:
                content = document.content
            
            # Clean and normalize text
            content = self.text_processor.clean_text(content)
            content = self.text_processor.normalize_text(content)
            
            # Check content length
            if len(content) < self.min_text_length:
                self.logger.warning(f"Document {document.id} content too short after processing")
                return []
            
            if len(content) > self.max_text_length:
                self.logger.warning(f"Document {document.id} content too long, truncating")
                content = content[:self.max_text_length]
            
            # Create chunks
            chunk_texts = self.chunker.create_chunks(content)
            
            # Create Chunk objects
            chunks = []
            for i, chunk_text in enumerate(chunk_texts):
                if len(chunk_text.strip()) >= self.min_text_length:
                    chunk = Chunk(
                        id=f"{document.id}_chunk_{i}",
                        document_id=document.id,
                        content=chunk_text.strip(),
                        position=i
                    )
                    chunks.append(chunk)
            
            self.logger.info(f"Created {len(chunks)} chunks for document {document.id}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to process document {document.id}: {e}")
            raise ProcessingError(f"Failed to process document {document.id}: {e}")
    
    def validate_text_quality(self, text: str) -> Tuple[bool, str]:
        """
        Validate text quality for processing.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not text or not text.strip():
            return False, "Empty text"
        
        if len(text) < self.min_text_length:
            return False, f"Text too short (minimum {self.min_text_length} characters)"
        
        if self.language_filter == 'ja' and not self.text_processor.is_japanese_text(text):
            return False, "Text is not Japanese"
        
        # Check for excessive repetition
        words = text.split()
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:
                return False, "Text has excessive repetition"
        
        return True, "Valid"
    
    def get_processing_stats(self, document: Document) -> Dict[str, Any]:
        """
        Get processing statistics for a document.
        
        Args:
            document: Document to analyze
            
        Returns:
            Dictionary containing processing statistics
        """
        stats = {
            'document_id': document.id,
            'original_length': len(document.content),
            'is_japanese': self.text_processor.is_japanese_text(document.content),
            'estimated_chunks': 0,
            'processing_time': 0
        }
        
        try:
            # Estimate number of chunks
            if document.content:
                cleaned_content = self.text_processor.clean_text(document.content)
                estimated_chunks = max(1, len(cleaned_content) // self.chunker.chunk_size)
                stats['estimated_chunks'] = estimated_chunks
        
        except Exception as e:
            self.logger.error(f"Failed to get processing stats: {e}")
        
        return stats
"""
Configuration management for the RAG system.
Loads and validates configuration from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application configuration."""
    name: str
    version: str
    debug: bool
    host: str
    port: int


@dataclass
class CrawlerConfig:
    """Web crawler configuration."""
    target_url: str
    max_depth: int
    delay_between_requests: float
    timeout: int
    user_agent: str
    retry_attempts: int
    retry_delay: float


@dataclass
class DocumentProcessingConfig:
    """Document processing configuration."""
    chunk_size: int
    chunk_overlap: int
    language_filter: str
    min_text_length: int
    max_text_length: int


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str
    model_path: str
    batch_size: int
    max_seq_length: int
    device: str


@dataclass
class VectorDBConfig:
    """Vector database configuration."""
    storage_path: str
    index_type: str
    similarity_metric: str
    top_k_results: int


@dataclass
class GenerationConfig:
    """Generation model configuration."""
    model_name: str
    model_path: str
    max_length: int
    temperature: float
    top_p: float
    device: str


@dataclass
class SearchConfig:
    """Search configuration."""
    min_similarity_score: float
    max_results: int
    enable_reranking: bool
    enable_query_expansion: bool
    enable_query_normalization: bool
    max_query_length: int
    enable_search_analytics: bool


@dataclass
class StorageConfig:
    """Storage configuration."""
    data_dir: str
    documents_dir: str
    logs_dir: str
    models_dir: str


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    format: str
    file_path: str
    max_file_size: str
    backup_count: int
    enable_console: bool
    enable_file: bool


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    max_concurrent_requests: int
    request_timeout: int
    cache_size: int
    enable_metrics: bool


@dataclass
class Config:
    """Main configuration class."""
    app: AppConfig
    crawler: CrawlerConfig
    document_processing: DocumentProcessingConfig
    embedding: EmbeddingConfig
    vector_db: VectorDBConfig
    generation: GenerationConfig
    search: SearchConfig
    storage: StorageConfig
    logging: LoggingConfig
    performance: PerformanceConfig


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default path.
        
    Returns:
        Configuration object
        
    Raises:
        FileNotFoundError: If configuration file is not found
        yaml.YAMLError: If configuration file is invalid
    """
    if config_path is None:
        config_path = "config.yaml"
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # Create configuration objects
    app_config = AppConfig(**config_data['app'])
    crawler_config = CrawlerConfig(**config_data['crawler'])
    doc_processing_config = DocumentProcessingConfig(**config_data['document_processing'])
    embedding_config = EmbeddingConfig(**config_data['embedding'])
    vector_db_config = VectorDBConfig(**config_data['vector_db'])
    generation_config = GenerationConfig(**config_data['generation'])
    search_config = SearchConfig(**config_data['search'])
    storage_config = StorageConfig(**config_data['storage'])
    logging_config = LoggingConfig(**config_data['logging'])
    performance_config = PerformanceConfig(**config_data['performance'])
    
    return Config(
        app=app_config,
        crawler=crawler_config,
        document_processing=doc_processing_config,
        embedding=embedding_config,
        vector_db=vector_db_config,
        generation=generation_config,
        search=search_config,
        storage=storage_config,
        logging=logging_config,
        performance=performance_config
    )


def create_directories(config: Config) -> None:
    """
    Create necessary directories based on configuration.
    
    Args:
        config: Configuration object
    """
    directories = [
        config.storage.data_dir,
        config.storage.documents_dir,
        config.storage.logs_dir,
        config.storage.models_dir,
        config.vector_db.storage_path,
        config.embedding.model_path,
        config.generation.model_path,
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration cache
_config_cache: Optional[Dict[str, Any]] = None


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration as dictionary (simplified interface).
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    global _config_cache
    
    if _config_cache is None:
        if config_path is None:
            config_path = "config.yaml"
        
        config_file = Path(config_path)
        if not config_file.exists():
            # Return default configuration if file doesn't exist
            return {
                'embedding': {
                    'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                    'model_path': './models/embedding',
                    'batch_size': 32,
                    'max_seq_length': 512,
                    'device': 'cpu'
                }
            }
        
        with open(config_file, 'r', encoding='utf-8') as f:
            _config_cache = yaml.safe_load(f)
    
    return _config_cache
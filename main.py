"""
Main entry point for the RAG system.
Initializes configuration, logging, and starts the application.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import load_config, create_directories
from src.utils.logging_config import setup_logging, get_logger


def main():
    """Main application entry point."""
    try:
        # Load configuration
        config = load_config()
        
        # Set up logging
        setup_logging(config.logging.__dict__)
        logger = get_logger(__name__)
        
        logger.info("Starting RAG System", version=config.app.version)
        
        # Create necessary directories
        create_directories(config)
        logger.info("Created necessary directories")
        
        # Log configuration summary
        logger.info("Configuration loaded successfully",
                   app_name=config.app.name,
                   debug_mode=config.app.debug,
                   target_url=config.crawler.target_url)
        
        logger.info("RAG System initialized successfully")
        
    except Exception as e:
        print(f"Failed to initialize RAG System: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
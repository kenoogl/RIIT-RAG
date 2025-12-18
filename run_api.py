#!/usr/bin/env python3
"""
CLI script to run the RAG API server.

This script provides a simple way to start the FastAPI server
with proper configuration and logging.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.api.main import run_server
from src.utils.config import get_config
from src.utils.logging_config import setup_logging


def main():
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(description="Run the RAG API server")
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to (default: from config or 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: from config or 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging_config = {
        "level": args.log_level,
        "enable_console": True,
        "enable_file": True,
        "logs_dir": "./logs"
    }
    setup_logging(logging_config)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = get_config()
        app_config = config.get("app", {})
        
        # Determine host and port
        host = args.host or app_config.get("host", "0.0.0.0")
        port = args.port or app_config.get("port", 8000)
        reload = args.reload or args.debug or app_config.get("debug", False)
        
        logger.info(f"Starting RAG API server on {host}:{port}")
        logger.info(f"Debug mode: {args.debug}")
        logger.info(f"Auto-reload: {reload}")
        
        # Run the server
        run_server(host=host, port=port, reload=reload)
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
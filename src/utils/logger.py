"""
Logging utilities for AI Banking Microservice.
"""
import logging
import structlog
from typing import Any, Dict
from src.config.settings import get_settings

settings = get_settings()


def setup_logging() -> None:
    """Configure structured logging for the application."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.log_format == "json" 
            else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("motor").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def log_request(request_data: Dict[str, Any], logger: structlog.BoundLogger) -> None:
    """Log request data for audit purposes."""
    logger.info(
        "Request received",
        method=request_data.get("method"),
        path=request_data.get("path"),
        client_id=request_data.get("client_id"),
        timestamp=request_data.get("timestamp")
    )


def log_response(response_data: Dict[str, Any], logger: structlog.BoundLogger) -> None:
    """Log response data for audit purposes."""
    logger.info(
        "Response sent",
        status_code=response_data.get("status_code"),
        latency_ms=response_data.get("latency_ms"),
        client_id=response_data.get("client_id"),
        timestamp=response_data.get("timestamp")
    )

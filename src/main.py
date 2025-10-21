"""
AI Banking Microservice - Main Application
"""
import uvicorn
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from src.config.settings import get_settings
from src.database.connection import db_manager
from src.middleware.auth_middleware import APIKeyAuthMiddleware
from src.routes.health import router as health_router
from src.routes.fraud import router as fraud_router
from src.services.fraud_detection import fraud_service
from src.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Get settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting AI Banking Microservice")
    try:
        await db_manager.connect()
        logger.info("Database connected successfully")
        
        # Initialize fraud detection service
        await fraud_service.initialize()
        logger.info("Fraud detection service initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Banking Microservice")
    await db_manager.disconnect()
    logger.info("Database disconnected")


# Create FastAPI application
app = FastAPI(
    title="AI Banking Microservice",
    description="Fraud detection and transaction processing microservice",
    version=settings.service_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment == "development" else None,
    redoc_url="/redoc" if settings.environment == "development" else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allowed_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Add API key authentication middleware
app.add_middleware(APIKeyAuthMiddleware)

# Include routers
app.include_router(health_router, tags=["Health"])
app.include_router(fraud_router, tags=["Fraud Detection"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "An unexpected error occurred",
            "service": settings.service_name
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
            "service": settings.service_name
        }
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.environment
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.service_host,
        port=settings.service_port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower()
    )

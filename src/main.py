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
from src.routes.graph import router as graph_router
from src.routes.aml import router as aml_router
from src.routes.credit import router as credit_router
from src.routes.models import router as models_router
from src.routes.monitoring import router as monitoring_router
from src.services.fraud_detection import fraud_service
from src.services.gnn_fraud import gnn_service
from src.services.aml import aml_service
from src.services.credit_scoring import credit_service
from src.services.retraining_pipeline import retraining_pipeline
from src.services.deployment_service import deployment_service
from src.services.metrics_service import metrics_service
from src.services.alerting_service import alerting_service
from src.services.tracing_service import tracing_service
from src.services.kafka_service import kafka_producer, kafka_consumer
from src.feature_store.feast_client import feature_store_client
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
        
        # Initialize feature store
        await feature_store_client.initialize()
        logger.info("Feature Store client initialized")

        # Initialize GNN service
        await gnn_service.initialize()
        logger.info("GNN Fraud Service initialized")

        # Initialize AML service
        await aml_service.initialize()
        logger.info("AML Detection Service initialized")

        # Initialize Credit Scoring service
        await credit_service.initialize()
        logger.info("Credit Scoring Service initialized")

        # Initialize Model Retraining Pipeline
        await retraining_pipeline.initialize()
        logger.info("Model Retraining Pipeline initialized")

        # Initialize Model Deployment Service
        await deployment_service.initialize()
        logger.info("Model Deployment Service initialized")

        # Initialize Monitoring Services
        await metrics_service.initialize()
        logger.info("Prometheus Metrics Service initialized")

        await alerting_service.initialize()
        logger.info("Alerting Service initialized")

        await tracing_service.initialize()
        logger.info("Distributed Tracing Service initialized")

        # Initialize Kafka services
        await kafka_producer.initialize()
        logger.info("Kafka producer initialized")
        
        # Start Kafka consumer in background
        asyncio.create_task(kafka_consumer.initialize())
        asyncio.create_task(kafka_consumer.start_consuming())
        logger.info("Kafka consumer started")

        # Initialize fraud detection service
        await fraud_service.initialize()
        logger.info("Fraud detection service initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Banking Microservice")
    
    # Shutdown services
    await kafka_consumer.shutdown()
    await kafka_producer.shutdown()
    await db_manager.disconnect()
    logger.info("All services shutdown successfully")


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
app.include_router(graph_router, tags=["Graph Analysis"])
app.include_router(aml_router, tags=["AML Analysis"])
app.include_router(credit_router, tags=["Credit Scoring"])
app.include_router(models_router, tags=["Model Management"])
app.include_router(monitoring_router, tags=["Monitoring"])


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

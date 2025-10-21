"""
Health check endpoints for monitoring and readiness probes.
"""
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse
from datetime import datetime
import logging

from src.config.settings import get_settings
from src.database.connection import db_manager
from src.feature_store.feast_client import feature_store_client
from src.services.gnn_fraud import gnn_service
from src.services.aml import aml_service
from src.services.credit_scoring import credit_service
from src.services.kafka_service import kafka_producer

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.service_name,
        "version": settings.service_version,
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.environment
    }


@router.get("/health/db")
async def database_health_check():
    """Database connectivity health check."""
    try:
        is_healthy = await db_manager.health_check()
        if is_healthy:
            return {
                "status": "healthy",
                "database": "connected",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database connection failed"
            )
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database health check failed: {str(e)}"
        )


@router.get("/health/ready")
async def readiness_check():
    """Readiness probe for Kubernetes."""
    try:
        # Check database connectivity
        db_healthy = await db_manager.health_check()
        # Check feature store connectivity
        fs_healthy = await feature_store_client.health_check()
        # Check GNN service health
        gnn_healthy = gnn_service.is_initialized
        # Check AML service health
        aml_healthy = aml_service.is_initialized
        # Check Credit service health
        credit_healthy = credit_service.is_initialized
        # Check Kafka producer health
        kafka_healthy = kafka_producer.is_connected
        
        if not all([db_healthy, fs_healthy, gnn_healthy, aml_healthy, credit_healthy, kafka_healthy]):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready - dependency unavailable"
            )
        
        return {
            "status": "ready",
            "service": settings.service_name,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "database": "ok" if db_healthy else "unavailable",
                "feature_store": "ok" if fs_healthy else "unavailable",
                "gnn_service": "ok" if gnn_healthy else "unavailable",
                "aml_service": "ok" if aml_healthy else "unavailable",
                "credit_service": "ok" if credit_healthy else "unavailable",
                "kafka_producer": "ok" if kafka_healthy else "unavailable",
                "configuration": "ok"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}"
        )


@router.get("/health/live")
async def liveness_check():
    """Liveness probe for Kubernetes."""
    return {
        "status": "alive",
        "service": settings.service_name,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/metrics")
async def metrics_endpoint():
    """Basic metrics endpoint (placeholder for Prometheus)."""
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "uptime": "placeholder",  # TODO: Implement actual uptime calculation
        "requests_total": "placeholder",  # TODO: Implement request counter
        "timestamp": datetime.utcnow().isoformat()
    }

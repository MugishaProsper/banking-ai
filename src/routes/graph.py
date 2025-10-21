"""
Graph Neural Network API routes for fraud detection and relationship analysis.
"""
from fastapi import APIRouter, Request, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from src.services.gnn_fraud import gnn_service
from src.middleware.auth_middleware import check_permission, get_client_id
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/v1/graph", tags=["Graph Analysis"])


class GraphAnalysisRequest(BaseModel):
    """Request model for graph analysis."""
    
    account_id: str = Field(..., description="Account ID to analyze")
    max_depth: int = Field(default=3, ge=1, le=5, description="Maximum analysis depth")
    include_paths: bool = Field(default=True, description="Include suspicious paths")
    
    class Config:
        schema_extra = {
            "example": {
                "account_id": "acc_123456",
                "max_depth": 3,
                "include_paths": True
            }
        }


class GraphAnalysisResponse(BaseModel):
    """Response model for graph analysis."""
    
    account_id: str
    risk_score: float = Field(..., ge=0, le=1)
    neighbor_count: int
    suspicious_paths: List[Dict[str, Any]]
    graph_statistics: Dict[str, Any]
    timestamp: str
    
    class Config:
        schema_extra = {
            "example": {
                "account_id": "acc_123456",
                "risk_score": 0.75,
                "neighbor_count": 5,
                "suspicious_paths": [
                    {
                        "path": ["acc_123456", "acc_789012", "acc_345678"],
                        "suspiciousness": 0.8,
                        "node_types": ["account", "account", "account"]
                    }
                ],
                "graph_statistics": {
                    "total_nodes": 100,
                    "total_edges": 200
                },
                "timestamp": "2024-01-15T14:30:00Z"
            }
        }


@router.post("/analyze", response_model=GraphAnalysisResponse)
async def analyze_account_graph(
    request: GraphAnalysisRequest,
    req: Request
):
    """
    Analyze account relationships and suspicious patterns in the transaction graph.
    
    This endpoint provides deep graph analysis for fraud detection,
    including suspicious path detection and relationship risk assessment.
    """
    try:
        # Check permissions
        if not check_permission(req, "graph_analyze"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for graph analysis"
            )
        
        # Perform graph analysis
        analysis_data = {
            "sender_account": request.account_id,
            "receiver_account": "dummy",  # Not needed for single account analysis
            "amount": 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        gnn_result = await gnn_service.analyze_transaction(analysis_data)
        
        # Get suspicious paths if requested
        suspicious_paths = []
        if request.include_paths:
            suspicious_paths = gnn_service.graph.get_suspicious_paths(
                request.account_id, 
                max_depth=request.max_depth
            )
        
        # Get graph statistics
        graph_stats = await gnn_service.get_graph_statistics()
        
        result = {
            "account_id": request.account_id,
            "risk_score": gnn_result.get("sender_risk", 0.5),
            "neighbor_count": gnn_result.get("neighbor_count", 0),
            "suspicious_paths": suspicious_paths,
            "graph_statistics": graph_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Log the analysis request
        client_id = get_client_id(req)
        logger.info(
            "Graph analysis completed",
            account_id=request.account_id,
            risk_score=result["risk_score"],
            suspicious_paths_count=len(suspicious_paths),
            client_id=client_id
        )
        
        return GraphAnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing account graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze account graph"
        )


@router.get("/statistics")
async def get_graph_statistics(req: Request):
    """
    Get comprehensive graph statistics and health metrics.
    
    Returns information about the transaction graph including node counts,
    edge counts, risk distributions, and system health.
    """
    try:
        # Check permissions
        if not check_permission(req, "graph_stats"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for graph statistics access"
            )
        
        # Get graph statistics
        stats = await gnn_service.get_graph_statistics()
        
        # Add service health information
        health_info = {
            "service_status": "healthy" if gnn_service.is_initialized else "unhealthy",
            "last_update": gnn_service.last_update.isoformat() if gnn_service.last_update else None,
            "max_graph_size": gnn_service.max_graph_size,
            "current_size": len(gnn_service.graph.nodes)
        }
        
        result = {
            "status": "success",
            "graph_statistics": stats,
            "health": health_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting graph statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get graph statistics"
        )


@router.post("/update")
async def update_graph_with_transaction(
    transaction_data: Dict[str, Any],
    req: Request
):
    """
    Update the transaction graph with a new transaction.
    
    This endpoint allows manual graph updates for testing or
    real-time graph maintenance.
    """
    try:
        # Check permissions
        if not check_permission(req, "graph_update"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for graph updates"
            )
        
        # Validate required fields
        required_fields = ["sender_account", "receiver_account", "amount"]
        for field in required_fields:
            if field not in transaction_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {field}"
                )
        
        # Update graph
        await gnn_service.update_graph_with_transaction(transaction_data)
        
        # Log the update
        client_id = get_client_id(req)
        logger.info(
            "Graph updated with transaction",
            transaction_id=transaction_data.get("transaction_id"),
            sender=transaction_data.get("sender_account"),
            receiver=transaction_data.get("receiver_account"),
            client_id=client_id
        )
        
        return {
            "status": "success",
            "message": "Graph updated successfully",
            "transaction_id": transaction_data.get("transaction_id"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update graph"
        )


@router.post("/cleanup")
async def cleanup_graph_data(req: Request):
    """
    Clean up old graph data to prevent memory issues.
    
    This endpoint removes old edges and nodes from the graph
    to maintain optimal performance.
    """
    try:
        # Check permissions
        if not check_permission(req, "graph_admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for graph cleanup"
            )
        
        # Perform cleanup
        await gnn_service.cleanup_old_data()
        
        # Log the cleanup
        client_id = get_client_id(req)
        logger.info(
            "Graph cleanup completed",
            client_id=client_id
        )
        
        return {
            "status": "success",
            "message": "Graph cleanup completed successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cleanup graph"
        )


@router.get("/health")
async def graph_health_check(req: Request):
    """
    Check the health of the GNN service and graph.
    
    Returns detailed health information including initialization status,
    graph size, and performance metrics.
    """
    try:
        # Check permissions
        if not check_permission(req, "graph_health"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for graph health check"
            )
        
        # Get health information
        is_healthy = gnn_service.is_initialized
        graph_stats = await gnn_service.get_graph_statistics()
        
        health_info = {
            "service_status": "healthy" if is_healthy else "unhealthy",
            "initialized": gnn_service.is_initialized,
            "graph_size": len(gnn_service.graph.nodes),
            "edge_count": len(gnn_service.graph.edges),
            "last_update": gnn_service.last_update.isoformat() if gnn_service.last_update else None,
            "max_size": gnn_service.max_graph_size,
            "utilization": len(gnn_service.graph.nodes) / gnn_service.max_graph_size
        }
        
        if not is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GNN service is not healthy"
            )
        
        return {
            "status": "healthy",
            "health_info": health_info,
            "graph_statistics": graph_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking graph health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check graph health"
        )

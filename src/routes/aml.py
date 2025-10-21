"""
AML (Anti-Money Laundering) API routes for transaction analysis and compliance.
"""
from fastapi import APIRouter, Request, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from src.services.aml import aml_service, AMLPatternType, AMLRiskLevel
from src.middleware.auth_middleware import check_permission, get_client_id
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/v1/aml", tags=["AML Analysis"])


class AMLAnalysisRequest(BaseModel):
    """Request model for AML analysis."""
    
    transaction_id: str = Field(..., description="Transaction ID to analyze")
    sender_account: str = Field(..., description="Sender account ID")
    receiver_account: str = Field(..., description="Receiver account ID")
    amount: float = Field(..., gt=0, description="Transaction amount")
    timestamp: str = Field(..., description="Transaction timestamp")
    currency: str = Field(default="USD", description="Currency code")
    channel: str = Field(..., description="Transaction channel")
    description: Optional[str] = Field(default=None, description="Transaction description")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError('Invalid timestamp format. Use ISO format.')
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "txn_aml_001",
                "sender_account": "acc_123456",
                "receiver_account": "acc_789012",
                "amount": 9500.0,
                "timestamp": "2024-01-15T14:30:00Z",
                "currency": "USD",
                "channel": "mobile_app",
                "description": "Payment for services"
            }
        }


class AMLAnalysisResponse(BaseModel):
    """Response model for AML analysis."""
    
    transaction_id: str
    aml_score: float = Field(..., ge=0, le=1)
    risk_level: str
    detected_patterns: List[Dict[str, Any]]
    flags: List[str]
    explanation: str
    model_version: str
    timestamp: str
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "txn_aml_001",
                "aml_score": 0.75,
                "risk_level": "high",
                "detected_patterns": [
                    {
                        "pattern_type": "structuring",
                        "confidence": 0.8,
                        "description": "Potential structuring pattern detected"
                    }
                ],
                "flags": ["STRUCTURING_SUSPECTED", "HIGH_RISK_TRANSACTION"],
                "explanation": "AML score 0.750. Detected patterns: Potential structuring pattern detected",
                "model_version": "aml_v1.0",
                "timestamp": "2024-01-15T14:30:00Z"
            }
        }


class BatchAMLAnalysisRequest(BaseModel):
    """Request model for batch AML analysis."""
    
    transactions: List[AMLAnalysisRequest] = Field(..., description="List of transactions to analyze")
    analysis_options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Analysis options")
    
    class Config:
        schema_extra = {
            "example": {
                "transactions": [
                    {
                        "transaction_id": "txn_001",
                        "sender_account": "acc_123456",
                        "receiver_account": "acc_789012",
                        "amount": 5000.0,
                        "timestamp": "2024-01-15T14:30:00Z",
                        "channel": "mobile_app"
                    }
                ],
                "analysis_options": {
                    "include_pattern_details": True,
                    "risk_threshold": 0.5
                }
            }
        }


@router.post("/analyze", response_model=AMLAnalysisResponse)
async def analyze_transaction_aml(
    request: AMLAnalysisRequest,
    req: Request
):
    """
    Analyze a single transaction for AML patterns and compliance.
    
    This endpoint detects suspicious patterns like structuring, layering,
    smurfing, and other money laundering indicators.
    """
    try:
        # Check permissions
        if not check_permission(req, "aml_analyze"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for AML analysis"
            )
        
        # Convert request to transaction data
        transaction_data = request.dict()
        
        # Perform AML analysis
        aml_result = await aml_service.analyze_transaction(transaction_data)
        
        # Convert to response format
        response_data = {
            "transaction_id": aml_result.transaction_id,
            "aml_score": aml_result.aml_score,
            "risk_level": aml_result.risk_level.value,
            "detected_patterns": [
                {
                    "pattern_type": pattern.pattern_type.value,
                    "confidence": pattern.confidence,
                    "description": pattern.description,
                    "risk_level": pattern.risk_level.value,
                    "metadata": pattern.metadata
                }
                for pattern in aml_result.detected_patterns
            ],
            "flags": aml_result.flags,
            "explanation": aml_result.explanation,
            "model_version": aml_result.model_version,
            "timestamp": aml_result.timestamp.isoformat()
        }
        
        # Log the analysis request
        client_id = get_client_id(req)
        logger.info(
            "AML analysis completed",
            transaction_id=aml_result.transaction_id,
            aml_score=aml_result.aml_score,
            risk_level=aml_result.risk_level.value,
            patterns_detected=len(aml_result.detected_patterns),
            client_id=client_id
        )
        
        return AMLAnalysisResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing transaction for AML: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze transaction for AML"
        )


@router.post("/analyze/batch")
async def analyze_transactions_batch(
    request: BatchAMLAnalysisRequest,
    req: Request
):
    """
    Analyze multiple transactions for AML patterns in batch.
    
    This endpoint provides efficient batch processing for AML analysis
    of multiple transactions with configurable options.
    """
    try:
        # Check permissions
        if not check_permission(req, "aml_batch"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for batch AML analysis"
            )
        
        results = []
        risk_threshold = request.analysis_options.get("risk_threshold", 0.5)
        
        # Process each transaction
        for transaction_request in request.transactions:
            try:
                transaction_data = transaction_request.dict()
                aml_result = await aml_service.analyze_transaction(transaction_data)
                
                # Filter by risk threshold if specified
                if aml_result.aml_score >= risk_threshold:
                    result = {
                        "transaction_id": aml_result.transaction_id,
                        "aml_score": aml_result.aml_score,
                        "risk_level": aml_result.risk_level.value,
                        "flags": aml_result.flags,
                        "patterns_count": len(aml_result.detected_patterns)
                    }
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error processing transaction {transaction_request.transaction_id}: {e}")
                continue
        
        # Log batch analysis
        client_id = get_client_id(req)
        logger.info(
            "Batch AML analysis completed",
            total_transactions=len(request.transactions),
            high_risk_transactions=len(results),
            client_id=client_id
        )
        
        return {
            "status": "success",
            "total_transactions": len(request.transactions),
            "analyzed_transactions": len(results),
            "high_risk_transactions": results,
            "analysis_options": request.analysis_options,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch AML analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform batch AML analysis"
        )


@router.get("/patterns")
async def get_aml_patterns(req: Request):
    """
    Get available AML pattern types and their descriptions.
    
    Returns information about all detectable AML patterns
    and their risk characteristics.
    """
    try:
        # Check permissions
        if not check_permission(req, "aml_patterns"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for AML patterns access"
            )
        
        patterns = []
        for pattern_type in AMLPatternType:
            patterns.append({
                "pattern_type": pattern_type.value,
                "description": pattern_type.value.replace("_", " ").title(),
                "threshold": aml_service.pattern_thresholds.get(pattern_type, 0.5)
            })
        
        return {
            "status": "success",
            "patterns": patterns,
            "total_patterns": len(patterns),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting AML patterns: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get AML patterns"
        )


@router.get("/statistics")
async def get_aml_statistics(req: Request):
    """
    Get AML service statistics and performance metrics.
    
    Returns comprehensive statistics about AML analysis performance,
    pattern detection rates, and service health.
    """
    try:
        # Check permissions
        if not check_permission(req, "aml_stats"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for AML statistics access"
            )
        
        stats = await aml_service.get_aml_statistics()
        
        return {
            "status": "success",
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting AML statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get AML statistics"
        )


@router.post("/thresholds/update")
async def update_aml_thresholds(
    thresholds: Dict[str, float],
    req: Request
):
    """
    Update AML pattern detection thresholds.
    
    This endpoint allows authorized users to adjust AML pattern
    detection sensitivity for different compliance requirements.
    """
    try:
        # Check permissions
        if not check_permission(req, "aml_admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for AML threshold updates"
            )
        
        # Validate threshold values
        for pattern_str, threshold in thresholds.items():
            try:
                pattern_type = AMLPatternType(pattern_str)
                if not 0.0 <= threshold <= 1.0:
                    raise ValueError(f"Threshold must be between 0.0 and 1.0")
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid threshold for {pattern_str}: {str(e)}"
                )
        
        # Update thresholds
        threshold_dict = {AMLPatternType(k): v for k, v in thresholds.items()}
        await aml_service.update_pattern_thresholds(threshold_dict)
        
        # Log threshold update
        client_id = get_client_id(req)
        logger.info(
            "AML thresholds updated",
            thresholds=thresholds,
            client_id=client_id
        )
        
        return {
            "status": "success",
            "message": "AML thresholds updated successfully",
            "updated_thresholds": thresholds,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating AML thresholds: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update AML thresholds"
        )


@router.get("/health")
async def aml_health_check(req: Request):
    """
    Check the health of the AML service.
    
    Returns detailed health information including initialization status,
    pattern detection capabilities, and performance metrics.
    """
    try:
        # Check permissions
        if not check_permission(req, "aml_health"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for AML health check"
            )
        
        # Get health information
        is_healthy = aml_service.is_initialized
        stats = await aml_service.get_aml_statistics()
        
        health_info = {
            "service_status": "healthy" if is_healthy else "unhealthy",
            "initialized": aml_service.is_initialized,
            "model_version": aml_service.model_version,
            "pattern_types": len(AMLPatternType),
            "risk_levels": len(AMLRiskLevel),
            "thresholds_configured": len(aml_service.pattern_thresholds)
        }
        
        if not is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AML service is not healthy"
            )
        
        return {
            "status": "healthy",
            "health_info": health_info,
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking AML health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check AML health"
        )

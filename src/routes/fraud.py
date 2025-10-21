"""
Fraud Detection API Routes - Core ML scoring endpoints.
"""
from fastapi import APIRouter, Request, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from src.services.fraud_detection import fraud_service
from src.database.models import Transaction, TransactionType, TransactionStatus
from src.database.repositories import transaction_repo, fraud_score_repo
from src.middleware.auth_middleware import check_permission, get_client_id
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["Fraud Detection"])


class TransactionScoreRequest(BaseModel):
    """Request model for transaction scoring."""
    
    transaction_id: str = Field(..., description="Unique transaction identifier")
    timestamp: str = Field(..., description="Transaction timestamp in ISO format")
    amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field(default="USD", description="Currency code")
    sender_account: str = Field(..., description="Sender account ID")
    receiver_account: str = Field(..., description="Receiver account ID")
    channel: str = Field(..., description="Transaction channel")
    transaction_type: TransactionType = Field(..., description="Type of transaction")
    description: Optional[str] = Field(default=None, description="Transaction description")
    reference: Optional[str] = Field(default=None, description="Transaction reference")
    features: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Precomputed features")
    
    @field_validator('timestamp')
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError('Invalid timestamp format. Use ISO format.')
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "txn_1234567890",
                "timestamp": "2024-01-15T14:30:00Z",
                "amount": 150.75,
                "currency": "USD",
                "sender_account": "acc_123456",
                "receiver_account": "acc_789012",
                "channel": "mobile_app",
                "transaction_type": "transfer",
                "description": "Payment for services",
                "features": {
                    "device_risk_score": 0.2,
                    "location_risk_score": 0.1
                }
            }
        }
    }


class TransactionScoreResponse(BaseModel):
    """Response model for transaction scoring."""
    
    transaction_id: str
    model_version: str
    probability: float = Field(..., ge=0, le=1)
    decision: str = Field(..., pattern="^(ALLOW|HOLD|BLOCK)$")
    explanation: Dict[str, Any]
    latency_ms: int
    timestamp: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "txn_1234567890",
                "model_version": "tabular_v2024-01-15",
                "probability": 0.87,
                "decision": "HOLD",
                "explanation": {
                    "top_features": [
                        {"feature": "amount_log", "impact": 0.45, "value": 2.18},
                        {"feature": "hour_of_day", "impact": 0.32, "value": 14}
                    ],
                    "graph_evidence": None,
                    "summary": "Transaction scored 0.870 based on 10 features"
                },
                "latency_ms": 38,
                "timestamp": "2024-01-15T14:30:00Z"
            }
        }
    }


class FeedbackRequest(BaseModel):
    """Request model for analyst feedback."""
    
    transaction_id: str = Field(..., description="Transaction ID")
    final_label: int = Field(..., ge=0, le=1, description="Final fraud label (1=fraud, 0=not fraud)")
    analyst_id: str = Field(..., description="Analyst identifier")
    notes: Optional[str] = Field(default=None, description="Analyst notes")
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "txn_1234567890",
                "final_label": 1,
                "analyst_id": "ANL42",
                "notes": "Confirmed chargeback"
            }
        }


@router.post("/score", response_model=TransactionScoreResponse)
async def score_transaction(
    request: TransactionScoreRequest,
    req: Request
):
    """
    Score a transaction for fraud risk.
    
    This endpoint provides real-time fraud scoring for individual transactions.
    Returns a fraud probability, decision (ALLOW/HOLD/BLOCK), and explanation.
    """
    try:
        # Check permissions
        if not check_permission(req, "fraud_score"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for fraud scoring"
            )
        
        # Convert request to dict for processing
        transaction_data = request.dict()
        
        # Score the transaction
        result = await fraud_service.score_transaction(transaction_data)
        
        # Store transaction in database
        await _store_transaction(transaction_data, result)
        
        # Log the scoring request
        client_id = get_client_id(req)
        logger.info(
            "Transaction scored",
            transaction_id=result["transaction_id"],
            decision=result["decision"],
            probability=result["probability"],
            client_id=client_id
        )
        
        return TransactionScoreResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scoring transaction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to score transaction"
        )


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    req: Request
):
    """
    Submit analyst feedback for model training.
    
    This endpoint accepts human analyst feedback and confirmed outcomes
    for improving fraud detection models.
    """
    try:
        # Check permissions
        if not check_permission(req, "fraud_feedback"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for feedback submission"
            )
        
        # Store feedback
        feedback_data = {
            "transaction_id": request.transaction_id,
            "final_label": request.final_label,
            "analyst_id": request.analyst_id,
            "notes": request.notes,
            "created_at": datetime.utcnow(),
            "client_id": get_client_id(req)
        }
        
        # TODO: Store in feedback collection and publish to Kafka
        logger.info(
            "Feedback received",
            transaction_id=request.transaction_id,
            final_label=request.final_label,
            analyst_id=request.analyst_id
        )
        
        return {
            "status": "success",
            "message": "Feedback submitted successfully",
            "transaction_id": request.transaction_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )


@router.get("/explain/{transaction_id}")
async def explain_transaction(
    transaction_id: str,
    req: Request
):
    """
    Get full explanation for a previously scored transaction.
    
    Returns detailed explanation and graph trace for audit purposes.
    """
    try:
        # Check permissions
        if not check_permission(req, "fraud_explain"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for explanation access"
            )
        
        # Get fraud scores for transaction
        fraud_scores = await fraud_score_repo.get_scores_by_transaction(transaction_id)
        
        if not fraud_scores:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No fraud scores found for transaction"
            )
        
        # Get transaction details
        transaction = await transaction_repo.find_by_transaction_id(transaction_id)
        
        return {
            "transaction_id": transaction_id,
            "transaction": transaction.dict() if transaction else None,
            "fraud_scores": [score.dict() for score in fraud_scores],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining transaction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get transaction explanation"
        )


@router.get("/model/status")
async def get_model_status(req: Request):
    """
    Get status of fraud detection models.
    
    Returns information about loaded models, versions, and performance.
    """
    try:
        # Check permissions
        if not check_permission(req, "model_status"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for model status access"
            )
        
        status_info = await fraud_service.get_model_status()
        
        return {
            "status": "success",
            "data": status_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model status"
        )


async def _store_transaction(transaction_data: Dict[str, Any], fraud_result: Dict[str, Any]) -> None:
    """Store transaction with fraud scoring results."""
    try:
        # Create transaction record
        transaction_record = {
            "transaction_id": transaction_data["transaction_id"],
            "sender_account": transaction_data["sender_account"],
            "receiver_account": transaction_data["receiver_account"],
            "amount": transaction_data["amount"],
            "currency": transaction_data["currency"],
            "transaction_type": transaction_data["transaction_type"],
            "status": TransactionStatus.PENDING,
            "channel": transaction_data["channel"],
            "timestamp": datetime.fromisoformat(transaction_data["timestamp"].replace('Z', '+00:00')),
            "description": transaction_data.get("description"),
            "reference": transaction_data.get("reference"),
            "fraud_score": fraud_result["probability"],
            "fraud_decision": fraud_result["decision"],
            "fraud_model_version": fraud_result["model_version"],
            "fraud_explanation": fraud_result["explanation"],
            "metadata": transaction_data.get("features", {})
        }
        
        # Store in database
        await transaction_repo.create_transaction(transaction_record)
        
    except Exception as e:
        logger.error(f"Error storing transaction: {e}")
        # Don't raise exception as this shouldn't fail the scoring request

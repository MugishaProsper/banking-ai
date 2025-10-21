"""
Credit Scoring API routes for risk assessment and loan decisions.
"""
from fastapi import APIRouter, Request, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from src.services.credit_scoring import credit_service, CreditApplication, CreditRiskLevel, CreditDecision
from src.middleware.auth_middleware import check_permission, get_client_id
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/v1/credit", tags=["Credit Scoring"])


class CreditApplicationRequest(BaseModel):
    """Request model for credit application."""
    
    account_id: str = Field(..., description="Account ID")
    requested_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_purpose: str = Field(..., description="Purpose of the loan")
    term_months: int = Field(..., ge=1, le=360, description="Loan term in months")
    monthly_income: float = Field(..., gt=0, description="Monthly income")
    existing_debt: float = Field(default=0, ge=0, description="Existing debt amount")
    employment_status: str = Field(..., description="Employment status")
    credit_history_months: int = Field(..., ge=0, description="Credit history length in months")
    
    @validator('employment_status')
    def validate_employment_status(cls, v):
        valid_statuses = ['employed', 'self_employed', 'unemployed', 'retired', 'student']
        if v.lower() not in valid_statuses:
            raise ValueError(f'Employment status must be one of: {valid_statuses}')
        return v.lower()
    
    class Config:
        schema_extra = {
            "example": {
                "account_id": "acc_123456",
                "requested_amount": 25000.0,
                "loan_purpose": "home_purchase",
                "term_months": 60,
                "monthly_income": 5000.0,
                "existing_debt": 5000.0,
                "employment_status": "employed",
                "credit_history_months": 36
            }
        }


class CreditScoreResponse(BaseModel):
    """Response model for credit scoring."""
    
    account_id: str
    credit_score: int = Field(..., ge=300, le=850)
    risk_level: str
    decision: str
    confidence: float = Field(..., ge=0, le=1)
    factors: Dict[str, Any]
    explanation: str
    model_version: str
    timestamp: str
    
    class Config:
        schema_extra = {
            "example": {
                "account_id": "acc_123456",
                "credit_score": 720,
                "risk_level": "good",
                "decision": "approve",
                "confidence": 0.85,
                "factors": {
                    "payment_history": 0.95,
                    "credit_utilization": 0.75,
                    "credit_length": 0.60,
                    "credit_mix": 0.70,
                    "new_credit": 0.90
                },
                "explanation": "Credit score: 720. Strong payment history. Credit approved",
                "model_version": "credit_v1.0",
                "timestamp": "2024-01-15T14:30:00Z"
            }
        }


class CreditPreApprovalRequest(BaseModel):
    """Request model for credit pre-approval."""
    
    account_id: str = Field(..., description="Account ID")
    monthly_income: float = Field(..., gt=0, description="Monthly income")
    existing_debt: float = Field(default=0, ge=0, description="Existing debt amount")
    employment_status: str = Field(..., description="Employment status")
    credit_history_months: int = Field(..., ge=0, description="Credit history length")
    
    @validator('employment_status')
    def validate_employment_status(cls, v):
        valid_statuses = ['employed', 'self_employed', 'unemployed', 'retired', 'student']
        if v.lower() not in valid_statuses:
            raise ValueError(f'Employment status must be one of: {valid_statuses}')
        return v.lower()


@router.post("/score", response_model=CreditScoreResponse)
async def score_credit_application(
    request: CreditApplicationRequest,
    req: Request
):
    """
    Score a credit application and make lending decision.
    
    This endpoint analyzes creditworthiness based on payment history,
    credit utilization, credit length, and other factors to determine
    loan approval and terms.
    """
    try:
        # Check permissions
        if not check_permission(req, "credit_score"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for credit scoring"
            )
        
        # Convert request to credit application
        application = CreditApplication(
            account_id=request.account_id,
            requested_amount=request.requested_amount,
            loan_purpose=request.loan_purpose,
            term_months=request.term_months,
            monthly_income=request.monthly_income,
            existing_debt=request.existing_debt,
            employment_status=request.employment_status,
            credit_history_months=request.credit_history_months
        )
        
        # Perform credit scoring
        credit_result = await credit_service.score_credit_application(application)
        
        # Convert to response format
        response_data = {
            "account_id": credit_result.account_id,
            "credit_score": credit_result.credit_score,
            "risk_level": credit_result.risk_level.value,
            "decision": credit_result.decision.value,
            "confidence": credit_result.confidence,
            "factors": credit_result.factors,
            "explanation": credit_result.explanation,
            "model_version": credit_result.model_version,
            "timestamp": credit_result.timestamp.isoformat()
        }
        
        # Log the credit scoring request
        client_id = get_client_id(req)
        logger.info(
            "Credit scoring completed",
            account_id=credit_result.account_id,
            credit_score=credit_result.credit_score,
            decision=credit_result.decision.value,
            confidence=credit_result.confidence,
            client_id=client_id
        )
        
        return CreditScoreResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scoring credit application: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to score credit application"
        )


@router.post("/pre-approval")
async def get_credit_pre_approval(
    request: CreditPreApprovalRequest,
    req: Request
):
    """
    Get credit pre-approval estimate without full application.
    
    This endpoint provides a quick estimate of creditworthiness
    and potential loan amounts without requiring a full application.
    """
    try:
        # Check permissions
        if not check_permission(req, "credit_preapproval"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for credit pre-approval"
            )
        
        # Create a simplified application for pre-approval
        application = CreditApplication(
            account_id=request.account_id,
            requested_amount=50000,  # Default amount for pre-approval
            loan_purpose="pre_approval",
            term_months=60,  # Default term
            monthly_income=request.monthly_income,
            existing_debt=request.existing_debt,
            employment_status=request.employment_status,
            credit_history_months=request.credit_history_months
        )
        
        # Perform credit scoring
        credit_result = await credit_service.score_credit_application(application)
        
        # Calculate pre-approval amount based on income and debt
        debt_to_income = request.existing_debt / max(request.monthly_income * 12, 1)
        max_loan_amount = request.monthly_income * 12 * 0.3  # 30% of annual income
        
        if debt_to_income > 0.4:  # High debt-to-income ratio
            max_loan_amount *= 0.5
        elif debt_to_income > 0.2:  # Moderate debt-to-income ratio
            max_loan_amount *= 0.8
        
        # Adjust based on credit score
        if credit_result.credit_score >= 750:
            max_loan_amount *= 1.2
        elif credit_result.credit_score >= 700:
            max_loan_amount *= 1.1
        elif credit_result.credit_score < 600:
            max_loan_amount *= 0.5
        
        # Log pre-approval request
        client_id = get_client_id(req)
        logger.info(
            "Credit pre-approval completed",
            account_id=request.account_id,
            credit_score=credit_result.credit_score,
            max_loan_amount=max_loan_amount,
            client_id=client_id
        )
        
        return {
            "status": "success",
            "account_id": request.account_id,
            "credit_score": credit_result.credit_score,
            "risk_level": credit_result.risk_level.value,
            "pre_approved": credit_result.decision in [CreditDecision.APPROVE, CreditDecision.APPROVE_WITH_CONDITIONS],
            "max_loan_amount": round(max_loan_amount, 2),
            "estimated_rate_range": {
                "min_rate": 3.5 if credit_result.credit_score >= 750 else 5.0,
                "max_rate": 8.0 if credit_result.credit_score >= 700 else 12.0
            },
            "confidence": credit_result.confidence,
            "explanation": credit_result.explanation,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting credit pre-approval: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get credit pre-approval"
        )


@router.get("/factors")
async def get_credit_factors(req: Request):
    """
    Get credit scoring factors and their weights.
    
    Returns information about the factors used in credit scoring
    and their relative importance in the decision process.
    """
    try:
        # Check permissions
        if not check_permission(req, "credit_factors"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for credit factors access"
            )
        
        factors = []
        for factor, weight in credit_service.scoring_factors.items():
            factors.append({
                "factor": factor,
                "weight": weight,
                "description": factor.replace("_", " ").title(),
                "impact": "High" if weight >= 0.3 else "Medium" if weight >= 0.15 else "Low"
            })
        
        return {
            "status": "success",
            "scoring_factors": factors,
            "total_factors": len(factors),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting credit factors: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get credit factors"
        )


@router.get("/score-ranges")
async def get_credit_score_ranges(req: Request):
    """
    Get credit score ranges and risk levels.
    
    Returns information about credit score ranges and their
    corresponding risk levels and typical lending terms.
    """
    try:
        # Check permissions
        if not check_permission(req, "credit_ranges"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for credit score ranges access"
            )
        
        ranges = []
        for risk_level, (min_score, max_score) in credit_service.score_ranges.items():
            ranges.append({
                "risk_level": risk_level.value,
                "score_range": f"{min_score}-{max_score}",
                "min_score": min_score,
                "max_score": max_score,
                "typical_rate_range": {
                    "excellent": "3.5-4.5%",
                    "good": "4.5-6.0%",
                    "fair": "6.0-8.0%",
                    "poor": "8.0-12.0%",
                    "very_poor": "12.0%+"
                }.get(risk_level.value, "Varies")
            })
        
        return {
            "status": "success",
            "score_ranges": ranges,
            "total_ranges": len(ranges),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting credit score ranges: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get credit score ranges"
        )


@router.get("/statistics")
async def get_credit_statistics(req: Request):
    """
    Get credit scoring service statistics and performance metrics.
    
    Returns comprehensive statistics about credit scoring performance,
    decision distribution, and service health.
    """
    try:
        # Check permissions
        if not check_permission(req, "credit_stats"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for credit statistics access"
            )
        
        stats = await credit_service.get_credit_statistics()
        
        return {
            "status": "success",
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting credit statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get credit statistics"
        )


@router.get("/health")
async def credit_health_check(req: Request):
    """
    Check the health of the credit scoring service.
    
    Returns detailed health information including initialization status,
    scoring capabilities, and performance metrics.
    """
    try:
        # Check permissions
        if not check_permission(req, "credit_health"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for credit health check"
            )
        
        # Get health information
        is_healthy = credit_service.is_initialized
        stats = await credit_service.get_credit_statistics()
        
        health_info = {
            "service_status": "healthy" if is_healthy else "unhealthy",
            "initialized": credit_service.is_initialized,
            "model_version": credit_service.model_version,
            "scoring_factors": len(credit_service.scoring_factors),
            "risk_levels": len(CreditRiskLevel),
            "decision_types": len(CreditDecision),
            "score_range": credit_service.score_ranges
        }
        
        if not is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Credit scoring service is not healthy"
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
        logger.error(f"Error checking credit health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check credit health"
        )

"""
Model Retraining Pipeline API routes for automated ML model updates.
"""
from fastapi import APIRouter, Request, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from src.services.retraining_pipeline import retraining_pipeline, ModelType, RetrainingStatus
from src.services.deployment_service import deployment_service, DeploymentStrategy
from src.middleware.auth_middleware import check_permission, get_client_id
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/v1/models", tags=["Model Management"])


class RetrainingRequest(BaseModel):
    """Request model for manual retraining trigger."""
    
    model_type: str = Field(..., description="Type of model to retrain")
    trigger_reason: str = Field(default="manual", description="Reason for retraining")
    force_retrain: bool = Field(default=False, description="Force retraining even if not needed")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        valid_types = [mt.value for mt in ModelType]
        if v not in valid_types:
            raise ValueError(f'Model type must be one of: {valid_types}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "fraud_tabular",
                "trigger_reason": "performance_degradation",
                "force_retrain": False
            }
        }


class DeploymentRequest(BaseModel):
    """Request model for model deployment."""
    
    model_type: str = Field(..., description="Type of model to deploy")
    version: str = Field(..., description="Model version to deploy")
    strategy: str = Field(default="canary", description="Deployment strategy")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        valid_types = [mt.value for mt in ModelType]
        if v not in valid_types:
            raise ValueError(f'Model type must be one of: {valid_types}')
        return v
    
    @validator('strategy')
    def validate_strategy(cls, v):
        valid_strategies = [s.value for s in DeploymentStrategy]
        if v not in valid_strategies:
            raise ValueError(f'Strategy must be one of: {valid_strategies}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "fraud_tabular",
                "version": "fraud_tabular_v20240115_143000",
                "strategy": "canary"
            }
        }


class ABTestRequest(BaseModel):
    """Request model for A/B test."""
    
    model_type: str = Field(..., description="Type of model for A/B test")
    control_version: str = Field(..., description="Control version")
    treatment_version: str = Field(..., description="Treatment version")
    traffic_split: float = Field(default=0.5, ge=0.1, le=0.9, description="Traffic split to treatment")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        valid_types = [mt.value for mt in ModelType]
        if v not in valid_types:
            raise ValueError(f'Model type must be one of: {valid_types}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "fraud_tabular",
                "control_version": "fraud_tabular_v20240115_143000",
                "treatment_version": "fraud_tabular_v20240115_150000",
                "traffic_split": 0.5
            }
        }


@router.post("/retrain")
async def trigger_retraining(
    request: RetrainingRequest,
    req: Request
):
    """
    Trigger manual retraining for a specific model type.
    
    This endpoint allows authorized users to manually trigger
    model retraining based on performance issues or new data availability.
    """
    try:
        # Check permissions
        if not check_permission(req, "model_retrain"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for model retraining"
            )
        
        model_type = ModelType(request.model_type)
        
        # Trigger retraining
        await retraining_pipeline._trigger_retraining(model_type, request.trigger_reason)
        
        # Log the retraining request
        client_id = get_client_id(req)
        logger.info(
            "Manual retraining triggered",
            model_type=model_type.value,
            trigger_reason=request.trigger_reason,
            force_retrain=request.force_retrain,
            client_id=client_id
        )
        
        return {
            "status": "success",
            "message": f"Retraining triggered for {model_type.value}",
            "model_type": model_type.value,
            "trigger_reason": request.trigger_reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to trigger retraining"
        )


@router.post("/deploy")
async def deploy_model(
    request: DeploymentRequest,
    req: Request
):
    """
    Deploy a specific model version using the specified strategy.
    
    This endpoint allows deployment of model versions with different
    strategies including immediate, canary, and blue-green deployments.
    """
    try:
        # Check permissions
        if not check_permission(req, "model_deploy"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for model deployment"
            )
        
        model_type = ModelType(request.model_type)
        strategy = DeploymentStrategy(request.strategy)
        
        # Deploy model
        deployment_id = await deployment_service.deploy_model(
            model_type=model_type,
            version=request.version,
            strategy=strategy
        )
        
        # Log the deployment request
        client_id = get_client_id(req)
        logger.info(
            "Model deployment initiated",
            deployment_id=deployment_id,
            model_type=model_type.value,
            version=request.version,
            strategy=strategy.value,
            client_id=client_id
        )
        
        return {
            "status": "success",
            "message": f"Deployment initiated for {model_type.value}",
            "deployment_id": deployment_id,
            "model_type": model_type.value,
            "version": request.version,
            "strategy": strategy.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deploy model"
        )


@router.post("/ab-test")
async def start_ab_test(
    request: ABTestRequest,
    req: Request
):
    """
    Start an A/B test between two model versions.
    
    This endpoint allows comparison of model versions through
    controlled traffic splitting and performance evaluation.
    """
    try:
        # Check permissions
        if not check_permission(req, "model_abtest"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for A/B testing"
            )
        
        model_type = ModelType(request.model_type)
        
        # Start A/B test
        test_id = await deployment_service.start_ab_test(
            model_type=model_type,
            control_version=request.control_version,
            treatment_version=request.treatment_version,
            traffic_split=request.traffic_split
        )
        
        # Log the A/B test request
        client_id = get_client_id(req)
        logger.info(
            "A/B test started",
            test_id=test_id,
            model_type=model_type.value,
            control_version=request.control_version,
            treatment_version=request.treatment_version,
            traffic_split=request.traffic_split,
            client_id=client_id
        )
        
        return {
            "status": "success",
            "message": f"A/B test started for {model_type.value}",
            "test_id": test_id,
            "model_type": model_type.value,
            "control_version": request.control_version,
            "treatment_version": request.treatment_version,
            "traffic_split": request.traffic_split,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting A/B test: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start A/B test"
        )


@router.get("/versions")
async def get_model_versions(
    req: Request,
    model_type: Optional[str] = None
):
    """
    Get model versions information.
    
    Returns information about available model versions,
    their performance metrics, and deployment status.
    """
    try:
        # Check permissions
        if not check_permission(req, "model_versions"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for model versions access"
            )
        
        if model_type:
            try:
                mt = ModelType(model_type)
                versions_info = await retraining_pipeline.get_model_versions(mt)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid model type: {model_type}"
                )
        else:
            versions_info = await retraining_pipeline.get_model_versions()
        
        return {
            "status": "success",
            "model_versions": versions_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model versions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model versions"
        )


@router.get("/deployments")
async def get_active_deployments(req: Request):
    """
    Get active model deployments.
    
    Returns information about currently active deployments,
    their status, and traffic distribution.
    """
    try:
        # Check permissions
        if not check_permission(req, "model_deployments"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for deployments access"
            )
        
        deployments = await deployment_service.get_active_deployments()
        
        return {
            "status": "success",
            "active_deployments": deployments,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting active deployments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get active deployments"
        )


@router.get("/ab-tests")
async def get_ab_tests(req: Request):
    """
    Get A/B tests information.
    
    Returns information about active and completed A/B tests,
    including test results and winner versions.
    """
    try:
        # Check permissions
        if not check_permission(req, "model_abtests"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for A/B tests access"
            )
        
        ab_tests = await deployment_service.get_ab_tests()
        
        return {
            "status": "success",
            "ab_tests": ab_tests,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting A/B tests: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get A/B tests"
        )


@router.get("/retraining/status")
async def get_retraining_status(req: Request):
    """
    Get retraining pipeline status.
    
    Returns information about retraining jobs, their status,
    and pipeline health metrics.
    """
    try:
        # Check permissions
        if not check_permission(req, "model_status"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for retraining status access"
            )
        
        status_info = await retraining_pipeline.get_retraining_status()
        
        return {
            "status": "success",
            "retraining_status": status_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting retraining status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get retraining status"
        )


@router.get("/deployment/status")
async def get_deployment_status(req: Request):
    """
    Get deployment service status.
    
    Returns information about deployment service health,
    active deployments, and A/B tests.
    """
    try:
        # Check permissions
        if not check_permission(req, "model_status"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for deployment status access"
            )
        
        status_info = await deployment_service.get_deployment_status()
        
        return {
            "status": "success",
            "deployment_status": status_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting deployment status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get deployment status"
        )


@router.post("/rollback/{model_type}")
async def rollback_model(
    model_type: str,
    req: Request
):
    """
    Rollback a model to its previous version.
    
    This endpoint allows emergency rollback of model deployments
    when performance issues are detected.
    """
    try:
        # Check permissions
        if not check_permission(req, "model_rollback"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for model rollback"
            )
        
        try:
            mt = ModelType(model_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model type: {model_type}"
            )
        
        # Get current deployment
        current_deployment = deployment_service.active_deployments.get(mt)
        if not current_deployment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active deployment found for {model_type}"
            )
        
        if not current_deployment.rollback_version:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No rollback version available for {model_type}"
            )
        
        # Create rollback deployment
        rollback_deployment_id = await deployment_service.deploy_model(
            model_type=mt,
            version=current_deployment.rollback_version,
            strategy=DeploymentStrategy.IMMEDIATE
        )
        
        # Log the rollback request
        client_id = get_client_id(req)
        logger.info(
            "Model rollback initiated",
            model_type=model_type,
            from_version=current_deployment.version,
            to_version=current_deployment.rollback_version,
            client_id=client_id
        )
        
        return {
            "status": "success",
            "message": f"Rollback initiated for {model_type}",
            "rollback_deployment_id": rollback_deployment_id,
            "from_version": current_deployment.version,
            "to_version": current_deployment.rollback_version,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rolling back model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rollback model"
        )

"""
Model Deployment and A/B Testing service for gradual rollout and rollback.
"""
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import joblib
import shutil

from src.config.settings import get_settings
from src.services.retraining_pipeline import ModelType, DeploymentStatus, ModelVersion
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ABTestStatus(str, Enum):
    """A/B test status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class DeploymentStrategy(str, Enum):
    """Deployment strategies."""
    IMMEDIATE = "immediate"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"


@dataclass
class ABTest:
    """A/B test configuration."""
    test_id: str
    model_type: ModelType
    control_version: str
    treatment_version: str
    traffic_split: float  # Percentage of traffic to treatment
    status: ABTestStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    success_metrics: Optional[Dict[str, float]] = None
    winner_version: Optional[str] = None


@dataclass
class Deployment:
    """Model deployment information."""
    deployment_id: str
    model_type: ModelType
    version: str
    strategy: DeploymentStrategy
    status: DeploymentStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    traffic_percentage: float = 100.0
    rollback_version: Optional[str] = None


class ModelDeploymentService:
    """Model deployment and A/B testing service."""
    
    def __init__(self):
        self.is_initialized = False
        self.models_dir = Path("models")
        self.active_deployments: Dict[ModelType, Deployment] = {}
        self.ab_tests: Dict[str, ABTest] = {}
        self.deployment_history: List[Deployment] = []
        
        # Deployment configuration
        self.deployment_config = {
            "canary_duration_hours": 24,
            "canary_traffic_percentages": [5, 25, 50, 75, 100],
            "success_metrics_threshold": 0.95,
            "rollback_threshold": 0.85,
            "max_concurrent_tests": 3
        }
        
    async def initialize(self) -> None:
        """Initialize the deployment service."""
        try:
            # Create models directory
            self.models_dir.mkdir(exist_ok=True)
            
            # Load active deployments
            await self._load_active_deployments()
            
            # Start deployment monitoring
            asyncio.create_task(self._deployment_monitor())
            
            self.is_initialized = True
            logger.info("Model Deployment Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize deployment service: {e}")
            raise
    
    async def _load_active_deployments(self) -> None:
        """Load active deployments from disk."""
        try:
            deployment_file = self.models_dir / "active_deployments.json"
            if deployment_file.exists():
                with open(deployment_file, 'r') as f:
                    deployments_data = json.load(f)
                
                for deployment_data in deployments_data:
                    deployment = Deployment(
                        deployment_id=deployment_data["deployment_id"],
                        model_type=ModelType(deployment_data["model_type"]),
                        version=deployment_data["version"],
                        strategy=DeploymentStrategy(deployment_data["strategy"]),
                        status=DeploymentStatus(deployment_data["status"]),
                        created_at=datetime.fromisoformat(deployment_data["created_at"]),
                        started_at=datetime.fromisoformat(deployment_data["started_at"]) if deployment_data.get("started_at") else None,
                        traffic_percentage=deployment_data.get("traffic_percentage", 100.0),
                        rollback_version=deployment_data.get("rollback_version")
                    )
                    
                    self.active_deployments[deployment.model_type] = deployment
            
            logger.info(f"Loaded {len(self.active_deployments)} active deployments")
            
        except Exception as e:
            logger.error(f"Error loading active deployments: {e}")
    
    async def _deployment_monitor(self) -> None:
        """Monitor active deployments and A/B tests."""
        while True:
            try:
                await self._monitor_canary_deployments()
                await self._monitor_ab_tests()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in deployment monitor: {e}")
                await asyncio.sleep(300)
    
    async def _monitor_canary_deployments(self) -> None:
        """Monitor canary deployments and progress traffic."""
        try:
            for model_type, deployment in self.active_deployments.items():
                if (deployment.strategy == DeploymentStrategy.CANARY and 
                    deployment.status == DeploymentStatus.CANARY):
                    
                    # Check if canary duration has passed
                    if deployment.started_at:
                        duration_hours = (datetime.utcnow() - deployment.started_at).total_seconds() / 3600
                        
                        if duration_hours >= self.deployment_config["canary_duration_hours"]:
                            # Check performance metrics
                            if await self._check_deployment_performance(deployment):
                                # Promote to production
                                await self._promote_to_production(deployment)
                            else:
                                # Rollback
                                await self._rollback_deployment(deployment)
        except Exception as e:
            logger.error(f"Error monitoring canary deployments: {e}")
    
    async def _monitor_ab_tests(self) -> None:
        """Monitor A/B tests and determine winners."""
        try:
            for test_id, test in self.ab_tests.items():
                if test.status == ABTestStatus.RUNNING:
                    # Check if test duration has passed
                    if test.started_at:
                        duration_hours = (datetime.utcnow() - test.started_at).total_seconds() / 3600
                        
                        if duration_hours >= 48:  # 48-hour test duration
                            # Evaluate test results
                            await self._evaluate_ab_test(test)
        except Exception as e:
            logger.error(f"Error monitoring A/B tests: {e}")
    
    async def _check_deployment_performance(self, deployment: Deployment) -> bool:
        """Check if deployment is performing well."""
        try:
            # Get performance metrics for the deployed version
            metrics = await self._get_deployment_metrics(deployment)
            
            # Check against success threshold
            success_threshold = self.deployment_config["success_metrics_threshold"]
            
            for metric, value in metrics.items():
                if value < success_threshold:
                    logger.warning(f"Deployment {deployment.deployment_id} failed metric {metric}: {value}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking deployment performance: {e}")
            return False
    
    async def _get_deployment_metrics(self, deployment: Deployment) -> Dict[str, float]:
        """Get performance metrics for a deployment."""
        try:
            # This would query actual metrics from monitoring systems
            # For now, simulate metrics based on deployment age
            
            base_metrics = {
                "accuracy": 0.85,
                "latency_p99": 150.0,
                "throughput": 1000.0,
                "error_rate": 0.01
            }
            
            # Add some variation based on deployment age
            age_hours = (datetime.utcnow() - deployment.started_at).total_seconds() / 3600 if deployment.started_at else 0
            
            # Simulate performance degradation over time (for testing rollback)
            if age_hours > 12:
                degradation_factor = min(age_hours / 24, 0.2)  # Up to 20% degradation
                base_metrics["accuracy"] *= (1 - degradation_factor)
                base_metrics["latency_p99"] *= (1 + degradation_factor)
                base_metrics["error_rate"] *= (1 + degradation_factor * 2)
            
            return base_metrics
            
        except Exception as e:
            logger.error(f"Error getting deployment metrics: {e}")
            return {"error": 0.0}
    
    async def _promote_to_production(self, deployment: Deployment) -> None:
        """Promote canary deployment to production."""
        try:
            deployment.status = DeploymentStatus.PRODUCTION
            deployment.traffic_percentage = 100.0
            deployment.completed_at = datetime.utcnow()
            
            # Update active deployments
            self.active_deployments[deployment.model_type] = deployment
            
            # Save to disk
            await self._save_active_deployments()
            
            logger.info(f"Promoted deployment {deployment.deployment_id} to production")
            
        except Exception as e:
            logger.error(f"Error promoting deployment to production: {e}")
    
    async def _rollback_deployment(self, deployment: Deployment) -> None:
        """Rollback deployment to previous version."""
        try:
            if deployment.rollback_version:
                # Create rollback deployment
                rollback_deployment = Deployment(
                    deployment_id=f"rollback_{deployment.deployment_id}",
                    model_type=deployment.model_type,
                    version=deployment.rollback_version,
                    strategy=DeploymentStrategy.IMMEDIATE,
                    status=DeploymentStatus.ROLLBACK,
                    created_at=datetime.utcnow(),
                    started_at=datetime.utcnow(),
                    traffic_percentage=100.0
                )
                
                # Update active deployment
                self.active_deployments[deployment.model_type] = rollback_deployment
                
                # Save to disk
                await self._save_active_deployments()
                
                logger.info(f"Rolled back deployment {deployment.deployment_id} to version {deployment.rollback_version}")
            
        except Exception as e:
            logger.error(f"Error rolling back deployment: {e}")
    
    async def _evaluate_ab_test(self, test: ABTest) -> None:
        """Evaluate A/B test results and determine winner."""
        try:
            # Get metrics for both versions
            control_metrics = await self._get_version_metrics(test.control_version)
            treatment_metrics = await self._get_version_metrics(test.treatment_version)
            
            # Calculate improvement
            improvements = {}
            for metric in control_metrics:
                if metric in treatment_metrics:
                    improvement = (treatment_metrics[metric] - control_metrics[metric]) / control_metrics[metric]
                    improvements[metric] = improvement
            
            # Determine winner (simplified logic)
            overall_improvement = np.mean(list(improvements.values()))
            
            if overall_improvement > 0.05:  # 5% improvement threshold
                test.winner_version = test.treatment_version
                test.success_metrics = improvements
                logger.info(f"A/B test {test.test_id} winner: treatment version")
            else:
                test.winner_version = test.control_version
                test.success_metrics = improvements
                logger.info(f"A/B test {test.test_id} winner: control version")
            
            test.status = ABTestStatus.COMPLETED
            test.ended_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error evaluating A/B test: {e}")
            test.status = ABTestStatus.FAILED
    
    async def _get_version_metrics(self, version: str) -> Dict[str, float]:
        """Get performance metrics for a specific version."""
        try:
            # This would query actual metrics from monitoring systems
            # For now, simulate metrics
            
            base_metrics = {
                "accuracy": np.random.uniform(0.80, 0.90),
                "precision": np.random.uniform(0.75, 0.85),
                "recall": np.random.uniform(0.70, 0.80),
                "latency_p99": np.random.uniform(100, 200),
                "throughput": np.random.uniform(800, 1200)
            }
            
            return base_metrics
            
        except Exception as e:
            logger.error(f"Error getting version metrics: {e}")
            return {"error": 0.0}
    
    async def deploy_model(self, model_type: ModelType, version: str, 
                          strategy: DeploymentStrategy = DeploymentStrategy.CANARY) -> str:
        """Deploy a model version."""
        try:
            deployment_id = f"{model_type.value}_{version}_{int(time.time())}"
            
            # Get current deployment for rollback version
            current_deployment = self.active_deployments.get(model_type)
            rollback_version = current_deployment.version if current_deployment else None
            
            # Create deployment
            deployment = Deployment(
                deployment_id=deployment_id,
                model_type=model_type,
                version=version,
                strategy=strategy,
                status=DeploymentStatus.STAGING,
                created_at=datetime.utcnow(),
                rollback_version=rollback_version
            )
            
            # Start deployment
            await self._start_deployment(deployment)
            
            logger.info(f"Started deployment {deployment_id} for {model_type.value} version {version}")
            
            return deployment_id
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            raise
    
    async def _start_deployment(self, deployment: Deployment) -> None:
        """Start a deployment."""
        try:
            deployment.status = DeploymentStatus.STAGING
            deployment.started_at = datetime.utcnow()
            
            if deployment.strategy == DeploymentStrategy.IMMEDIATE:
                # Deploy immediately to production
                deployment.status = DeploymentStatus.PRODUCTION
                deployment.traffic_percentage = 100.0
                deployment.completed_at = datetime.utcnow()
                
            elif deployment.strategy == DeploymentStrategy.CANARY:
                # Start canary deployment
                deployment.status = DeploymentStatus.CANARY
                deployment.traffic_percentage = self.deployment_config["canary_traffic_percentages"][0]
                
            # Update active deployments
            self.active_deployments[deployment.model_type] = deployment
            
            # Add to history
            self.deployment_history.append(deployment)
            
            # Save to disk
            await self._save_active_deployments()
            
        except Exception as e:
            logger.error(f"Error starting deployment: {e}")
            raise
    
    async def start_ab_test(self, model_type: ModelType, control_version: str, 
                           treatment_version: str, traffic_split: float = 0.5) -> str:
        """Start an A/B test between two model versions."""
        try:
            test_id = f"ab_test_{model_type.value}_{int(time.time())}"
            
            # Check if we can start a new test
            active_tests = len([t for t in self.ab_tests.values() if t.status == ABTestStatus.RUNNING])
            if active_tests >= self.deployment_config["max_concurrent_tests"]:
                raise ValueError("Maximum concurrent A/B tests reached")
            
            # Create A/B test
            ab_test = ABTest(
                test_id=test_id,
                model_type=model_type,
                control_version=control_version,
                treatment_version=treatment_version,
                traffic_split=traffic_split,
                status=ABTestStatus.PENDING,
                created_at=datetime.utcnow()
            )
            
            # Start the test
            ab_test.status = ABTestStatus.RUNNING
            ab_test.started_at = datetime.utcnow()
            
            self.ab_tests[test_id] = ab_test
            
            logger.info(f"Started A/B test {test_id} for {model_type.value}")
            
            return test_id
            
        except Exception as e:
            logger.error(f"Error starting A/B test: {e}")
            raise
    
    async def _save_active_deployments(self) -> None:
        """Save active deployments to disk."""
        try:
            deployments_data = []
            for deployment in self.active_deployments.values():
                deployments_data.append({
                    "deployment_id": deployment.deployment_id,
                    "model_type": deployment.model_type.value,
                    "version": deployment.version,
                    "strategy": deployment.strategy.value,
                    "status": deployment.status.value,
                    "created_at": deployment.created_at.isoformat(),
                    "started_at": deployment.started_at.isoformat() if deployment.started_at else None,
                    "traffic_percentage": deployment.traffic_percentage,
                    "rollback_version": deployment.rollback_version
                })
            
            deployment_file = self.models_dir / "active_deployments.json"
            with open(deployment_file, 'w') as f:
                json.dump(deployments_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving active deployments: {e}")
    
    async def get_deployment_status(self) -> Dict[str, Any]:
        """Get deployment service status."""
        return {
            "is_initialized": self.is_initialized,
            "active_deployments": len(self.active_deployments),
            "active_ab_tests": len([t for t in self.ab_tests.values() if t.status == ABTestStatus.RUNNING]),
            "total_deployments": len(self.deployment_history),
            "deployment_strategies": [strategy.value for strategy in DeploymentStrategy],
            "last_update": datetime.utcnow().isoformat()
        }
    
    async def get_active_deployments(self) -> Dict[str, Any]:
        """Get active deployments information."""
        result = {}
        for model_type, deployment in self.active_deployments.items():
            result[model_type.value] = {
                "deployment_id": deployment.deployment_id,
                "version": deployment.version,
                "strategy": deployment.strategy.value,
                "status": deployment.status.value,
                "traffic_percentage": deployment.traffic_percentage,
                "created_at": deployment.created_at.isoformat(),
                "started_at": deployment.started_at.isoformat() if deployment.started_at else None,
                "rollback_version": deployment.rollback_version
            }
        return result
    
    async def get_ab_tests(self) -> Dict[str, Any]:
        """Get A/B tests information."""
        result = {}
        for test_id, test in self.ab_tests.items():
            result[test_id] = {
                "model_type": test.model_type.value,
                "control_version": test.control_version,
                "treatment_version": test.treatment_version,
                "traffic_split": test.traffic_split,
                "status": test.status.value,
                "created_at": test.created_at.isoformat(),
                "started_at": test.started_at.isoformat() if test.started_at else None,
                "ended_at": test.ended_at.isoformat() if test.ended_at else None,
                "winner_version": test.winner_version,
                "success_metrics": test.success_metrics
            }
        return result


# Global deployment service instance
deployment_service = ModelDeploymentService()

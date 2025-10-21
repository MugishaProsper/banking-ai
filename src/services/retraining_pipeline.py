"""
Model Retraining Pipeline for automated ML model updates and deployment.
"""
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import shutil
import hashlib

from src.config.settings import get_settings
from src.database.repositories import transaction_repo, fraud_score_repo
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ModelType(str, Enum):
    """Model types for retraining."""
    FRAUD_TABULAR = "fraud_tabular"
    FRAUD_ANOMALY = "fraud_anomaly"
    AML_PATTERN = "aml_pattern"
    CREDIT_SCORE = "credit_score"


class RetrainingStatus(str, Enum):
    """Retraining job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeploymentStatus(str, Enum):
    """Model deployment status."""
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    ROLLBACK = "rollback"


@dataclass
class ModelVersion:
    """Model version information."""
    model_type: ModelType
    version: str
    created_at: datetime
    performance_metrics: Dict[str, float]
    training_data_size: int
    feature_count: int
    model_hash: str
    status: DeploymentStatus
    is_active: bool = False


@dataclass
class RetrainingJob:
    """Retraining job information."""
    job_id: str
    model_type: ModelType
    status: RetrainingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    new_version: Optional[str] = None


class ModelRetrainingPipeline:
    """Automated model retraining and deployment pipeline."""
    
    def __init__(self):
        self.is_initialized = False
        self.models_dir = Path("models")
        self.versions_dir = Path("model_versions")
        self.training_data_dir = Path("training_data")
        self.jobs: Dict[str, RetrainingJob] = {}
        self.model_versions: Dict[ModelType, List[ModelVersion]] = {
            model_type: [] for model_type in ModelType
        }
        
        # Retraining configuration
        self.retraining_config = {
            ModelType.FRAUD_TABULAR: {
                "retrain_interval_hours": 168,  # Weekly
                "min_data_points": 1000,
                "performance_threshold": 0.85,
                "drift_threshold": 0.1
            },
            ModelType.FRAUD_ANOMALY: {
                "retrain_interval_hours": 168,  # Weekly
                "min_data_points": 2000,
                "performance_threshold": 0.8,
                "drift_threshold": 0.15
            },
            ModelType.AML_PATTERN: {
                "retrain_interval_hours": 720,  # Monthly
                "min_data_points": 500,
                "performance_threshold": 0.9,
                "drift_threshold": 0.05
            },
            ModelType.CREDIT_SCORE: {
                "retrain_interval_hours": 720,  # Monthly
                "min_data_points": 2000,
                "performance_threshold": 0.88,
                "drift_threshold": 0.08
            }
        }
        
    async def initialize(self) -> None:
        """Initialize the retraining pipeline."""
        try:
            # Create directories
            self.models_dir.mkdir(exist_ok=True)
            self.versions_dir.mkdir(exist_ok=True)
            self.training_data_dir.mkdir(exist_ok=True)
            
            # Load existing model versions
            await self._load_model_versions()
            
            # Start background retraining scheduler
            asyncio.create_task(self._retraining_scheduler())
            
            self.is_initialized = True
            logger.info("Model Retraining Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize retraining pipeline: {e}")
            raise
    
    async def _load_model_versions(self) -> None:
        """Load existing model versions from disk."""
        try:
            for model_type in ModelType:
                version_file = self.versions_dir / f"{model_type.value}_versions.json"
                if version_file.exists():
                    with open(version_file, 'r') as f:
                        versions_data = json.load(f)
                    
                    versions = []
                    for version_data in versions_data:
                        version = ModelVersion(
                            model_type=ModelType(version_data["model_type"]),
                            version=version_data["version"],
                            created_at=datetime.fromisoformat(version_data["created_at"]),
                            performance_metrics=version_data["performance_metrics"],
                            training_data_size=version_data["training_data_size"],
                            feature_count=version_data["feature_count"],
                            model_hash=version_data["model_hash"],
                            status=DeploymentStatus(version_data["status"]),
                            is_active=version_data.get("is_active", False)
                        )
                        versions.append(version)
                    
                    self.model_versions[model_type] = sorted(versions, key=lambda v: v.created_at, reverse=True)
            
            logger.info(f"Loaded model versions for {len(ModelType)} model types")
            
        except Exception as e:
            logger.error(f"Error loading model versions: {e}")
    
    async def _retraining_scheduler(self) -> None:
        """Background scheduler for automatic retraining."""
        while True:
            try:
                await self._check_retraining_triggers()
                await asyncio.sleep(3600)  # Check every hour
            except Exception as e:
                logger.error(f"Error in retraining scheduler: {e}")
                await asyncio.sleep(3600)
    
    async def _check_retraining_triggers(self) -> None:
        """Check if models need retraining based on triggers."""
        try:
            for model_type in ModelType:
                config = self.retraining_config[model_type]
                
                # Check time-based trigger
                last_retrain = await self._get_last_retrain_time(model_type)
                if last_retrain:
                    hours_since_retrain = (datetime.utcnow() - last_retrain).total_seconds() / 3600
                    if hours_since_retrain >= config["retrain_interval_hours"]:
                        await self._trigger_retraining(model_type, "scheduled")
                        continue
                
                # Check performance drift
                if await self._check_performance_drift(model_type, config["drift_threshold"]):
                    await self._trigger_retraining(model_type, "performance_drift")
                    continue
                
                # Check data availability
                if await self._check_data_availability(model_type, config["min_data_points"]):
                    await self._trigger_retraining(model_type, "new_data")
                    
        except Exception as e:
            logger.error(f"Error checking retraining triggers: {e}")
    
    async def _get_last_retrain_time(self, model_type: ModelType) -> Optional[datetime]:
        """Get the last retraining time for a model type."""
        versions = self.model_versions.get(model_type, [])
        if versions:
            return versions[0].created_at
        return None
    
    async def _check_performance_drift(self, model_type: ModelType, threshold: float) -> bool:
        """Check if model performance has drifted beyond threshold."""
        try:
            # Get recent performance metrics
            recent_metrics = await self._get_recent_performance_metrics(model_type)
            if not recent_metrics:
                return False
            
            # Compare with baseline performance
            baseline_metrics = await self._get_baseline_performance(model_type)
            if not baseline_metrics:
                return False
            
            # Check for significant drift
            for metric, recent_value in recent_metrics.items():
                if metric in baseline_metrics:
                    baseline_value = baseline_metrics[metric]
                    drift = abs(recent_value - baseline_value) / baseline_value
                    if drift > threshold:
                        logger.warning(f"Performance drift detected for {model_type}: {metric} drift={drift:.3f}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking performance drift: {e}")
            return False
    
    async def _check_data_availability(self, model_type: ModelType, min_points: int) -> bool:
        """Check if enough new data is available for retraining."""
        try:
            # Get data count since last retrain
            last_retrain = await self._get_last_retrain_time(model_type)
            if not last_retrain:
                return True  # First training
            
            # Count new data points (simplified - would query actual database)
            new_data_count = await self._count_new_data_points(model_type, last_retrain)
            
            return new_data_count >= min_points
            
        except Exception as e:
            logger.error(f"Error checking data availability: {e}")
            return False
    
    async def _count_new_data_points(self, model_type: ModelType, since: datetime) -> int:
        """Count new data points since last retrain."""
        try:
            # This would query the actual database for new transactions/feedback
            # For now, simulate based on model type
            if model_type == ModelType.FRAUD_TABULAR:
                return np.random.randint(500, 2000)
            elif model_type == ModelType.AML_PATTERN:
                return np.random.randint(200, 1000)
            else:
                return np.random.randint(300, 1500)
                
        except Exception as e:
            logger.error(f"Error counting new data points: {e}")
            return 0
    
    async def _trigger_retraining(self, model_type: ModelType, trigger: str) -> None:
        """Trigger retraining for a model type."""
        try:
            job_id = f"{model_type.value}_{int(time.time())}"
            
            job = RetrainingJob(
                job_id=job_id,
                model_type=model_type,
                status=RetrainingStatus.PENDING,
                created_at=datetime.utcnow()
            )
            
            self.jobs[job_id] = job
            
            # Start retraining in background
            asyncio.create_task(self._execute_retraining(job_id))
            
            logger.info(f"Triggered retraining for {model_type.value} (trigger: {trigger})")
            
        except Exception as e:
            logger.error(f"Error triggering retraining: {e}")
    
    async def _execute_retraining(self, job_id: str) -> None:
        """Execute retraining job."""
        try:
            job = self.jobs[job_id]
            job.status = RetrainingStatus.RUNNING
            job.started_at = datetime.utcnow()
            
            logger.info(f"Starting retraining job {job_id} for {job.model_type.value}")
            
            # Prepare training data
            training_data = await self._prepare_training_data(job.model_type)
            
            # Train new model
            new_model, metrics = await self._train_model(job.model_type, training_data)
            
            # Evaluate model performance
            performance_metrics = await self._evaluate_model(new_model, job.model_type)
            
            # Create new model version
            new_version = await self._create_model_version(
                job.model_type, new_model, performance_metrics, training_data
            )
            
            # Update job status
            job.status = RetrainingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.performance_metrics = performance_metrics
            job.new_version = new_version.version
            
            logger.info(f"Retraining job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Retraining job {job_id} failed: {e}")
            job.status = RetrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
    
    async def _prepare_training_data(self, model_type: ModelType) -> pd.DataFrame:
        """Prepare training data for model type."""
        try:
            # This would query the database for training data
            # For now, generate synthetic data based on model type
            
            if model_type == ModelType.FRAUD_TABULAR:
                n_samples = 5000
                features = np.random.randn(n_samples, 10)
                labels = np.random.randint(0, 2, n_samples)
                
            elif model_type == ModelType.FRAUD_ANOMALY:
                n_samples = 3000
                features = np.random.randn(n_samples, 8)
                labels = np.random.randint(0, 2, n_samples)
                
            elif model_type == ModelType.AML_PATTERN:
                n_samples = 2000
                features = np.random.randn(n_samples, 6)
                labels = np.random.randint(0, 2, n_samples)
                
            elif model_type == ModelType.CREDIT_SCORE:
                n_samples = 4000
                features = np.random.randn(n_samples, 12)
                labels = np.random.randint(300, 851, n_samples)
            
            # Create DataFrame
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]
            df = pd.DataFrame(features, columns=feature_names)
            df['target'] = labels
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    async def _train_model(self, model_type: ModelType, training_data: pd.DataFrame) -> Tuple[Any, Dict[str, float]]:
        """Train a new model."""
        try:
            from sklearn.ensemble import RandomForestClassifier, IsolationForest
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            X = training_data.drop('target', axis=1)
            y = training_data['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model based on type
            if model_type == ModelType.FRAUD_TABULAR:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Calculate metrics
                y_pred = model.predict(X_test)
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average='weighted'),
                    "recall": recall_score(y_test, y_pred, average='weighted'),
                    "f1_score": f1_score(y_test, y_pred, average='weighted')
                }
                
            elif model_type == ModelType.FRAUD_ANOMALY:
                model = IsolationForest(contamination=0.1, random_state=42)
                model.fit(X_train)
                
                # Calculate metrics
                y_pred = model.predict(X_test)
                metrics = {
                    "anomaly_detection_rate": np.mean(y_pred == -1),
                    "normal_detection_rate": np.mean(y_pred == 1)
                }
                
            else:
                # Default to RandomForest for other types
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average='weighted'),
                    "recall": recall_score(y_test, y_pred, average='weighted'),
                    "f1_score": f1_score(y_test, y_pred, average='weighted')
                }
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    async def _evaluate_model(self, model: Any, model_type: ModelType) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            # This would use proper evaluation datasets
            # For now, return simulated metrics
            
            if model_type == ModelType.FRAUD_TABULAR:
                return {
                    "accuracy": np.random.uniform(0.85, 0.95),
                    "precision": np.random.uniform(0.80, 0.90),
                    "recall": np.random.uniform(0.75, 0.85),
                    "f1_score": np.random.uniform(0.80, 0.88),
                    "auc": np.random.uniform(0.85, 0.95)
                }
            elif model_type == ModelType.FRAUD_ANOMALY:
                return {
                    "anomaly_detection_rate": np.random.uniform(0.80, 0.95),
                    "false_positive_rate": np.random.uniform(0.05, 0.15),
                    "precision": np.random.uniform(0.75, 0.90)
                }
            elif model_type == ModelType.AML_PATTERN:
                return {
                    "pattern_detection_rate": np.random.uniform(0.85, 0.95),
                    "false_positive_rate": np.random.uniform(0.02, 0.08),
                    "precision": np.random.uniform(0.90, 0.98)
                }
            elif model_type == ModelType.CREDIT_SCORE:
                return {
                    "mae": np.random.uniform(20, 50),
                    "rmse": np.random.uniform(30, 60),
                    "r2_score": np.random.uniform(0.80, 0.95)
                }
            
            return {"default_metric": 0.85}
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {"error": 0.0}
    
    async def _create_model_version(self, model_type: ModelType, model: Any, 
                                  performance_metrics: Dict[str, float], 
                                  training_data: pd.DataFrame) -> ModelVersion:
        """Create a new model version."""
        try:
            # Generate version string
            version = f"{model_type.value}_v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Calculate model hash
            model_bytes = joblib.dumps(model)
            model_hash = hashlib.md5(model_bytes).hexdigest()
            
            # Create model version
            model_version = ModelVersion(
                model_type=model_type,
                version=version,
                created_at=datetime.utcnow(),
                performance_metrics=performance_metrics,
                training_data_size=len(training_data),
                feature_count=training_data.shape[1] - 1,  # Exclude target
                model_hash=model_hash,
                status=DeploymentStatus.STAGING,
                is_active=False
            )
            
            # Save model and metadata
            await self._save_model_version(model_version, model)
            
            # Add to versions list
            self.model_versions[model_type].insert(0, model_version)
            
            # Save versions to disk
            await self._save_model_versions(model_type)
            
            logger.info(f"Created new model version {version} for {model_type.value}")
            
            return model_version
            
        except Exception as e:
            logger.error(f"Error creating model version: {e}")
            raise
    
    async def _save_model_version(self, model_version: ModelVersion, model: Any) -> None:
        """Save model version to disk."""
        try:
            # Create version directory
            version_dir = self.versions_dir / model_version.version
            version_dir.mkdir(exist_ok=True)
            
            # Save model
            model_path = version_dir / "model.pkl"
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata = {
                "model_type": model_version.model_type.value,
                "version": model_version.version,
                "created_at": model_version.created_at.isoformat(),
                "performance_metrics": model_version.performance_metrics,
                "training_data_size": model_version.training_data_size,
                "feature_count": model_version.feature_count,
                "model_hash": model_version.model_hash,
                "status": model_version.status.value,
                "is_active": model_version.is_active
            }
            
            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving model version: {e}")
            raise
    
    async def _save_model_versions(self, model_type: ModelType) -> None:
        """Save model versions to disk."""
        try:
            versions_data = []
            for version in self.model_versions[model_type]:
                versions_data.append({
                    "model_type": version.model_type.value,
                    "version": version.version,
                    "created_at": version.created_at.isoformat(),
                    "performance_metrics": version.performance_metrics,
                    "training_data_size": version.training_data_size,
                    "feature_count": version.feature_count,
                    "model_hash": version.model_hash,
                    "status": version.status.value,
                    "is_active": version.is_active
                })
            
            version_file = self.versions_dir / f"{model_type.value}_versions.json"
            with open(version_file, 'w') as f:
                json.dump(versions_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving model versions: {e}")
    
    async def _get_recent_performance_metrics(self, model_type: ModelType) -> Dict[str, float]:
        """Get recent performance metrics for drift detection."""
        # This would query actual performance metrics from the database
        # For now, return simulated metrics
        return {
            "accuracy": np.random.uniform(0.80, 0.90),
            "precision": np.random.uniform(0.75, 0.85),
            "recall": np.random.uniform(0.70, 0.80)
        }
    
    async def _get_baseline_performance(self, model_type: ModelType) -> Dict[str, float]:
        """Get baseline performance metrics."""
        # This would get the baseline from the current production model
        # For now, return simulated baseline metrics
        return {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.78
        }
    
    async def get_retraining_status(self) -> Dict[str, Any]:
        """Get retraining pipeline status."""
        return {
            "is_initialized": self.is_initialized,
            "active_jobs": len([job for job in self.jobs.values() if job.status == RetrainingStatus.RUNNING]),
            "pending_jobs": len([job for job in self.jobs.values() if job.status == RetrainingStatus.PENDING]),
            "total_jobs": len(self.jobs),
            "model_types": len(ModelType),
            "total_versions": sum(len(versions) for versions in self.model_versions.values()),
            "last_update": datetime.utcnow().isoformat()
        }
    
    async def get_model_versions(self, model_type: Optional[ModelType] = None) -> Dict[str, Any]:
        """Get model versions information."""
        if model_type:
            versions = self.model_versions.get(model_type, [])
            return {
                "model_type": model_type.value,
                "versions": [
                    {
                        "version": v.version,
                        "created_at": v.created_at.isoformat(),
                        "performance_metrics": v.performance_metrics,
                        "status": v.status.value,
                        "is_active": v.is_active
                    }
                    for v in versions
                ]
            }
        else:
            result = {}
            for mt in ModelType:
                versions = self.model_versions.get(mt, [])
                result[mt.value] = [
                    {
                        "version": v.version,
                        "created_at": v.created_at.isoformat(),
                        "performance_metrics": v.performance_metrics,
                        "status": v.status.value,
                        "is_active": v.is_active
                    }
                    for v in versions[:5]  # Latest 5 versions
                ]
            return result


# Global retraining pipeline instance
retraining_pipeline = ModelRetrainingPipeline()

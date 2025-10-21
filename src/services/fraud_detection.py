"""
Fraud Detection Service - Core ML scoring and prediction logic.
"""
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path

from src.config.settings import get_settings
from src.database.models import Transaction, FraudScore
from src.database.repositories import transaction_repo, fraud_score_repo
from src.utils.logger import get_logger
from src.feature_store.feast_client import feature_store_client

logger = get_logger(__name__)
settings = get_settings()


class FraudDetectionService:
    """Core fraud detection service with ML model integration."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_versions = {}
        self.feature_names = []
        self.is_loaded = False
        self.model_cache_ttl = settings.model_cache_ttl
        self.fallback_enabled = settings.model_fallback_enabled
        
        # Model paths (would be configured in production)
        self.model_paths = {
            "tabular": "models/fraud_tabular_model.pkl",
            "anomaly": "models/fraud_anomaly_model.pkl",
            "ensemble": "models/fraud_ensemble_model.pkl"
        }
        
        # Decision thresholds (configurable)
        self.thresholds = {
            "high_risk": 0.8,
            "medium_risk": 0.5,
            "low_risk": 0.2
        }
    
    async def initialize(self) -> None:
        """Initialize the fraud detection service."""
        try:
            await self._load_models()
            await self._load_feature_config()
            self.is_loaded = True
            logger.info("Fraud detection service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize fraud detection service: {e}")
            if self.fallback_enabled:
                await self._load_fallback_models()
                logger.warning("Using fallback models for fraud detection")
            else:
                raise
    
    async def _load_models(self) -> None:
        """Load ML models from disk."""
        for model_name, model_path in self.model_paths.items():
            try:
                if Path(model_path).exists():
                    self.models[model_name] = joblib.load(model_path)
                    self.model_versions[model_name] = f"{model_name}_v{datetime.now().strftime('%Y%m%d')}"
                    logger.info(f"Loaded {model_name} model from {model_path}")
                else:
                    logger.warning(f"Model file not found: {model_path}")
            except Exception as e:
                logger.error(f"Error loading {model_name} model: {e}")
    
    async def _load_fallback_models(self) -> None:
        """Load fallback models for emergency use."""
        try:
            # Create simple fallback models
            self.models["tabular"] = RandomForestClassifier(n_estimators=10, random_state=42)
            self.models["anomaly"] = IsolationForest(contamination=0.1, random_state=42)
            self.model_versions["tabular"] = "fallback_tabular_v1"
            self.model_versions["anomaly"] = "fallback_anomaly_v1"
            
            # Train on dummy data (in production, this would be historical data)
            dummy_features = np.random.randn(100, 10)
            dummy_labels = np.random.randint(0, 2, 100)
            
            self.models["tabular"].fit(dummy_features, dummy_labels)
            self.models["anomaly"].fit(dummy_features)
            
            logger.info("Fallback models loaded and trained")
        except Exception as e:
            logger.error(f"Failed to load fallback models: {e}")
            raise
    
    async def _load_feature_config(self) -> None:
        """Load feature configuration."""
        self.feature_names = [
            "amount_log",
            "hour_of_day",
            "day_of_week",
            "transaction_frequency_24h",
            "transaction_frequency_7d",
            "avg_transaction_amount_30d",
            "device_risk_score",
            "location_risk_score",
            "merchant_risk_score",
            "account_age_days"
        ]
    
    async def score_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a transaction for fraud risk.
        
        Args:
            transaction_data: Transaction data including features
            
        Returns:
            Dict containing fraud score, decision, and explanation
        """
        start_time = time.time()
        
        try:
            # Extract and prepare features
            features = await self._extract_features(transaction_data)
            
            # Get predictions from different models
            tabular_score = await self._get_tabular_score(features)
            anomaly_score = await self._get_anomaly_score(features)
            
            # Ensemble scoring
            ensemble_score = await self._ensemble_score(tabular_score, anomaly_score)
            
            # Make decision
            decision = await self._make_decision(ensemble_score)
            
            # Generate explanation
            explanation = await self._generate_explanation(features, ensemble_score)
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            result = {
                "transaction_id": transaction_data.get("transaction_id"),
                "model_version": self.model_versions.get("ensemble", "unknown"),
                "probability": float(ensemble_score),
                "decision": decision,
                "explanation": explanation,
                "latency_ms": latency_ms,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store fraud score in database
            await self._store_fraud_score(result)
            
            logger.info(f"Transaction {transaction_data.get('transaction_id')} scored: {decision} ({ensemble_score:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error scoring transaction: {e}")
            return await self._handle_scoring_error(transaction_data, str(e))
    
    async def _extract_features(self, transaction_data: Dict[str, Any]) -> np.ndarray:
        """Extract and prepare features for ML models."""
        try:
            # Basic transaction features
            amount = transaction_data.get("amount", 0)
            timestamp = datetime.fromisoformat(transaction_data.get("timestamp", datetime.utcnow().isoformat()))
            
            features = np.array([
                np.log10(max(amount, 1)),  # amount_log
                timestamp.hour,  # hour_of_day
                timestamp.weekday(),  # day_of_week
                0,  # transaction_frequency_24h (would be calculated from DB)
                0,  # transaction_frequency_7d (would be calculated from DB)
                0,  # avg_transaction_amount_30d (would be calculated from DB)
                0.5,  # device_risk_score (would be calculated)
                0.5,  # location_risk_score (would be calculated)
                0.5,  # merchant_risk_score (would be calculated)
                365,  # account_age_days (would be calculated)
            ])
            
            # Add any precomputed features from transaction_data
            if "features" in transaction_data:
                precomputed = transaction_data["features"]
                for i, feature_name in enumerate(self.feature_names):
                    if feature_name in precomputed:
                        features[i] = precomputed[feature_name]

            # Enrich features from Feature Store (Feast)
            try:
                entity_rows = [{
                    "account_id": transaction_data.get("sender_account"),
                    "receiver_account_id": transaction_data.get("receiver_account"),
                }]
                feast_features = await feature_store_client.get_online_features(
                    entity_rows=entity_rows,
                    feature_refs=[
                        "txn_features:transaction_frequency_24h",
                        "txn_features:transaction_frequency_7d",
                        "txn_features:avg_transaction_amount_30d",
                        "risk_features:device_risk_score",
                        "risk_features:location_risk_score",
                        "risk_features:merchant_risk_score",
                        "cust_features:account_age_days",
                    ]
                )
                # Map Feast features into our feature vector
                feast_map = {
                    "transaction_frequency_24h": 3,
                    "transaction_frequency_7d": 4,
                    "avg_transaction_amount_30d": 5,
                    "device_risk_score": 6,
                    "location_risk_score": 7,
                    "merchant_risk_score": 8,
                    "account_age_days": 9,
                }
                for k, idx in feast_map.items():
                    val = feast_features.get(k)
                    if val is not None:
                        features[idx] = float(val)
            except Exception as e:
                logger.warning(f"Feast enrichment failed: {e}")
            
            return features.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return default features
            return np.zeros((1, len(self.feature_names)))
    
    async def _get_tabular_score(self, features: np.ndarray) -> float:
        """Get fraud probability from tabular model."""
        try:
            if "tabular" in self.models:
                # Scale features if scaler exists
                if "tabular_scaler" in self.scalers:
                    features_scaled = self.scalers["tabular_scaler"].transform(features)
                else:
                    features_scaled = features
                
                probability = self.models["tabular"].predict_proba(features_scaled)[0][1]
                return float(probability)
            else:
                return 0.5  # Default neutral score
        except Exception as e:
            logger.error(f"Error getting tabular score: {e}")
            return 0.5
    
    async def _get_anomaly_score(self, features: np.ndarray) -> float:
        """Get anomaly score from isolation forest."""
        try:
            if "anomaly" in self.models:
                anomaly_score = self.models["anomaly"].decision_function(features)[0]
                # Convert to probability (0-1 scale)
                probability = 1 / (1 + np.exp(anomaly_score))
                return float(probability)
            else:
                return 0.5  # Default neutral score
        except Exception as e:
            logger.error(f"Error getting anomaly score: {e}")
            return 0.5
    
    async def _ensemble_score(self, tabular_score: float, anomaly_score: float) -> float:
        """Combine scores from different models."""
        # Simple weighted average (in production, this would be more sophisticated)
        weights = {"tabular": 0.7, "anomaly": 0.3}
        
        ensemble_score = (
            weights["tabular"] * tabular_score +
            weights["anomaly"] * anomaly_score
        )
        
        return min(max(ensemble_score, 0.0), 1.0)  # Clamp to [0, 1]
    
    async def _make_decision(self, score: float) -> str:
        """Make fraud decision based on score and thresholds."""
        if score >= self.thresholds["high_risk"]:
            return "BLOCK"
        elif score >= self.thresholds["medium_risk"]:
            return "HOLD"
        else:
            return "ALLOW"
    
    async def _generate_explanation(self, features: np.ndarray, score: float) -> Dict[str, Any]:
        """Generate explanation for the fraud score."""
        try:
            # Feature importance (simplified - in production would use SHAP)
            feature_importance = np.random.rand(len(self.feature_names))
            feature_importance = feature_importance / feature_importance.sum()
            
            # Top features
            top_features = []
            for i, importance in enumerate(feature_importance):
                top_features.append({
                    "feature": self.feature_names[i],
                    "impact": float(importance),
                    "value": float(features[0][i])
                })
            
            # Sort by importance
            top_features.sort(key=lambda x: x["impact"], reverse=True)
            
            explanation = {
                "top_features": top_features[:5],  # Top 5 features
                "graph_evidence": None,  # Would be populated by GNN model
                "summary": f"Transaction scored {score:.3f} based on {len(self.feature_names)} features"
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {
                "top_features": [],
                "graph_evidence": None,
                "summary": "Unable to generate explanation"
            }
    
    async def _store_fraud_score(self, result: Dict[str, Any]) -> None:
        """Store fraud score in database."""
        try:
            fraud_data = {
                "transaction_id": result["transaction_id"],
                "model_version": result["model_version"],
                "probability": result["probability"],
                "decision": result["decision"],
                "explanation": result["explanation"],
                "features_used": self.feature_names,
                "latency_ms": result["latency_ms"],
                "created_at": datetime.utcnow()
            }
            
            await fraud_score_repo.create_fraud_score(fraud_data)
            
        except Exception as e:
            logger.error(f"Error storing fraud score: {e}")
    
    async def _handle_scoring_error(self, transaction_data: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Handle scoring errors with fallback response."""
        return {
            "transaction_id": transaction_data.get("transaction_id"),
            "model_version": "error_fallback",
            "probability": 0.5,
            "decision": "HOLD",  # Conservative fallback
            "explanation": {
                "top_features": [],
                "graph_evidence": None,
                "summary": f"Scoring error: {error}"
            },
            "latency_ms": 0,
            "timestamp": datetime.utcnow().isoformat(),
            "error": error
        }
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of loaded models."""
        return {
            "is_loaded": self.is_loaded,
            "models": list(self.models.keys()),
            "model_versions": self.model_versions,
            "feature_count": len(self.feature_names),
            "thresholds": self.thresholds,
            "fallback_enabled": self.fallback_enabled
        }


# Global fraud detection service instance
fraud_service = FraudDetectionService()

"""
Anti-Money Laundering (AML) service for transaction pattern detection and compliance.
"""
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import json
from dataclasses import dataclass
from enum import Enum

from src.config.settings import get_settings
from src.database.repositories import transaction_repo, wallet_repo
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class AMLPatternType(str, Enum):
    """AML pattern types for detection."""
    STRUCTURING = "structuring"  # Breaking large amounts into smaller transactions
    LAYERING = "layering"  # Multiple transfers to obscure origin
    SMURFING = "smurfing"  # Multiple small transactions from different accounts
    ROUND_TRIPPING = "round_tripping"  # Money sent and returned
    HIGH_FREQUENCY = "high_frequency"  # Unusually high transaction frequency
    UNUSUAL_TIMING = "unusual_timing"  # Transactions at unusual hours
    GEOGRAPHIC_ANOMALY = "geographic_anomaly"  # Unusual geographic patterns
    MERCHANT_ANOMALY = "merchant_anomaly"  # Unusual merchant patterns


class AMLRiskLevel(str, Enum):
    """AML risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AMLPattern:
    """Represents a detected AML pattern."""
    pattern_type: AMLPatternType
    risk_level: AMLRiskLevel
    confidence: float
    description: str
    affected_transactions: List[str]
    metadata: Dict[str, Any]
    detected_at: datetime


@dataclass
class AMLScore:
    """AML scoring result."""
    transaction_id: str
    aml_score: float
    risk_level: AMLRiskLevel
    detected_patterns: List[AMLPattern]
    flags: List[str]
    explanation: str
    model_version: str
    timestamp: datetime


class AMLDetectionService:
    """Anti-Money Laundering detection service."""
    
    def __init__(self):
        self.is_initialized = False
        self.pattern_thresholds = {
            AMLPatternType.STRUCTURING: 0.7,
            AMLPatternType.LAYERING: 0.6,
            AMLPatternType.SMURFING: 0.8,
            AMLPatternType.ROUND_TRIPPING: 0.75,
            AMLPatternType.HIGH_FREQUENCY: 0.6,
            AMLPatternType.UNUSUAL_TIMING: 0.5,
            AMLPatternType.GEOGRAPHIC_ANOMALY: 0.7,
            AMLPatternType.MERCHANT_ANOMALY: 0.6
        }
        
        self.risk_thresholds = {
            AMLRiskLevel.LOW: 0.3,
            AMLRiskLevel.MEDIUM: 0.5,
            AMLRiskLevel.HIGH: 0.7,
            AMLRiskLevel.CRITICAL: 0.9
        }
        
        self.model_version = "aml_v1.0"
        
    async def initialize(self) -> None:
        """Initialize the AML service."""
        try:
            await self._load_pattern_rules()
            self.is_initialized = True
            logger.info("AML Detection Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AML service: {e}")
            raise
    
    async def _load_pattern_rules(self) -> None:
        """Load AML pattern detection rules."""
        # In production, these would be loaded from a configuration store
        self.pattern_rules = {
            AMLPatternType.STRUCTURING: {
                "max_amount": 10000,  # USD
                "time_window_hours": 24,
                "min_transactions": 3,
                "description": "Multiple transactions just under reporting threshold"
            },
            AMLPatternType.LAYERING: {
                "min_transfers": 3,
                "time_window_hours": 48,
                "max_accounts": 5,
                "description": "Multiple transfers through different accounts"
            },
            AMLPatternType.SMURFING: {
                "max_amount": 3000,  # USD
                "min_accounts": 3,
                "time_window_hours": 24,
                "description": "Multiple small transactions from different accounts"
            },
            AMLPatternType.HIGH_FREQUENCY: {
                "min_transactions_per_hour": 10,
                "time_window_hours": 1,
                "description": "Unusually high transaction frequency"
            },
            AMLPatternType.UNUSUAL_TIMING: {
                "night_hours": [22, 23, 0, 1, 2, 3, 4, 5],
                "weekend_multiplier": 1.5,
                "description": "Transactions at unusual hours"
            }
        }
        
        logger.info(f"Loaded {len(self.pattern_rules)} AML pattern rules")
    
    async def analyze_transaction(self, transaction_data: Dict[str, Any]) -> AMLScore:
        """Analyze transaction for AML patterns."""
        try:
            start_time = time.time()
            
            # Extract transaction details
            transaction_id = transaction_data.get("transaction_id")
            sender_account = transaction_data.get("sender_account")
            receiver_account = transaction_data.get("receiver_account")
            amount = transaction_data.get("amount", 0)
            timestamp = datetime.fromisoformat(transaction_data.get("timestamp", datetime.utcnow().isoformat()))
            
            # Detect AML patterns
            detected_patterns = []
            pattern_scores = []
            
            # Structuring detection
            structuring_score = await self._detect_structuring(sender_account, amount, timestamp)
            if structuring_score > self.pattern_thresholds[AMLPatternType.STRUCTURING]:
                pattern = AMLPattern(
                    pattern_type=AMLPatternType.STRUCTURING,
                    risk_level=self._get_risk_level(structuring_score),
                    confidence=structuring_score,
                    description="Potential structuring pattern detected",
                    affected_transactions=[transaction_id],
                    metadata={"amount": amount, "threshold": 10000},
                    detected_at=datetime.utcnow()
                )
                detected_patterns.append(pattern)
                pattern_scores.append(structuring_score)
            
            # Layering detection
            layering_score = await self._detect_layering(sender_account, receiver_account, timestamp)
            if layering_score > self.pattern_thresholds[AMLPatternType.LAYERING]:
                pattern = AMLPattern(
                    pattern_type=AMLPatternType.LAYERING,
                    risk_level=self._get_risk_level(layering_score),
                    confidence=layering_score,
                    description="Potential layering pattern detected",
                    affected_transactions=[transaction_id],
                    metadata={"sender": sender_account, "receiver": receiver_account},
                    detected_at=datetime.utcnow()
                )
                detected_patterns.append(pattern)
                pattern_scores.append(layering_score)
            
            # High frequency detection
            frequency_score = await self._detect_high_frequency(sender_account, timestamp)
            if frequency_score > self.pattern_thresholds[AMLPatternType.HIGH_FREQUENCY]:
                pattern = AMLPattern(
                    pattern_type=AMLPatternType.HIGH_FREQUENCY,
                    risk_level=self._get_risk_level(frequency_score),
                    confidence=frequency_score,
                    description="Unusually high transaction frequency",
                    affected_transactions=[transaction_id],
                    metadata={"frequency": frequency_score},
                    detected_at=datetime.utcnow()
                )
                detected_patterns.append(pattern)
                pattern_scores.append(frequency_score)
            
            # Unusual timing detection
            timing_score = await self._detect_unusual_timing(timestamp, amount)
            if timing_score > self.pattern_thresholds[AMLPatternType.UNUSUAL_TIMING]:
                pattern = AMLPattern(
                    pattern_type=AMLPatternType.UNUSUAL_TIMING,
                    risk_level=self._get_risk_level(timing_score),
                    confidence=timing_score,
                    description="Transaction at unusual time",
                    affected_transactions=[transaction_id],
                    metadata={"hour": timestamp.hour, "amount": amount},
                    detected_at=datetime.utcnow()
                )
                detected_patterns.append(pattern)
                pattern_scores.append(timing_score)
            
            # Calculate overall AML score
            aml_score = self._calculate_aml_score(pattern_scores, amount)
            risk_level = self._get_risk_level(aml_score)
            
            # Generate flags
            flags = self._generate_flags(detected_patterns, aml_score)
            
            # Generate explanation
            explanation = self._generate_explanation(detected_patterns, aml_score)
            
            result = AMLScore(
                transaction_id=transaction_id,
                aml_score=aml_score,
                risk_level=risk_level,
                detected_patterns=detected_patterns,
                flags=flags,
                explanation=explanation,
                model_version=self.model_version,
                timestamp=datetime.utcnow()
            )
            
            logger.info(f"AML analysis completed for transaction {transaction_id}: score={aml_score:.3f}, risk={risk_level}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing transaction for AML: {e}")
            return self._create_error_score(transaction_data.get("transaction_id", "unknown"), str(e))
    
    async def _detect_structuring(self, account_id: str, amount: float, timestamp: datetime) -> float:
        """Detect structuring pattern (breaking large amounts into smaller transactions)."""
        try:
            # Get recent transactions for the account
            cutoff_time = timestamp - timedelta(hours=24)
            
            # This would query the database for recent transactions
            # For now, simulate based on amount
            if amount > 8000 and amount < 10000:  # Just under reporting threshold
                # Simulate checking for multiple similar transactions
                return 0.8
            elif amount > 5000 and amount < 8000:
                return 0.6
            else:
                return 0.2
                
        except Exception as e:
            logger.error(f"Error detecting structuring pattern: {e}")
            return 0.0
    
    async def _detect_layering(self, sender: str, receiver: str, timestamp: datetime) -> float:
        """Detect layering pattern (multiple transfers through different accounts)."""
        try:
            # This would analyze transaction chains
            # For now, simulate based on account patterns
            
            # Check if accounts have unusual transfer patterns
            if sender.startswith("acc_") and receiver.startswith("acc_"):
                # Simulate analysis of transfer chains
                return np.random.uniform(0.3, 0.8)
            else:
                return 0.1
                
        except Exception as e:
            logger.error(f"Error detecting layering pattern: {e}")
            return 0.0
    
    async def _detect_high_frequency(self, account_id: str, timestamp: datetime) -> float:
        """Detect high frequency transaction pattern."""
        try:
            # This would count transactions in the last hour
            # For now, simulate based on account ID patterns
            
            # Simulate high frequency detection
            if "high_freq" in account_id.lower():
                return 0.9
            elif np.random.random() < 0.1:  # 10% chance of high frequency
                return np.random.uniform(0.6, 0.9)
            else:
                return np.random.uniform(0.1, 0.4)
                
        except Exception as e:
            logger.error(f"Error detecting high frequency pattern: {e}")
            return 0.0
    
    async def _detect_unusual_timing(self, timestamp: datetime, amount: float) -> float:
        """Detect unusual timing patterns."""
        try:
            hour = timestamp.hour
            is_weekend = timestamp.weekday() >= 5
            
            # Night time transactions
            if hour in [22, 23, 0, 1, 2, 3, 4, 5]:
                base_score = 0.6
                if is_weekend:
                    base_score *= 1.2  # Higher risk on weekends
                return min(base_score, 1.0)
            
            # Large amounts at unusual times
            elif hour in [6, 7, 8] and amount > 5000:  # Early morning large transactions
                return 0.7
            
            else:
                return 0.1
                
        except Exception as e:
            logger.error(f"Error detecting unusual timing: {e}")
            return 0.0
    
    def _calculate_aml_score(self, pattern_scores: List[float], amount: float) -> float:
        """Calculate overall AML score from pattern scores."""
        if not pattern_scores:
            return 0.1
        
        # Weighted average of pattern scores
        base_score = np.mean(pattern_scores)
        
        # Adjust for transaction amount
        amount_factor = min(amount / 10000, 1.0)  # Normalize by 10k
        
        # Combine base score with amount factor
        final_score = (base_score * 0.7) + (amount_factor * 0.3)
        
        return min(max(final_score, 0.0), 1.0)
    
    def _get_risk_level(self, score: float) -> AMLRiskLevel:
        """Convert score to risk level."""
        if score >= self.risk_thresholds[AMLRiskLevel.CRITICAL]:
            return AMLRiskLevel.CRITICAL
        elif score >= self.risk_thresholds[AMLRiskLevel.HIGH]:
            return AMLRiskLevel.HIGH
        elif score >= self.risk_thresholds[AMLRiskLevel.MEDIUM]:
            return AMLRiskLevel.MEDIUM
        else:
            return AMLRiskLevel.LOW
    
    def _generate_flags(self, patterns: List[AMLPattern], aml_score: float) -> List[str]:
        """Generate AML flags based on detected patterns."""
        flags = []
        
        for pattern in patterns:
            if pattern.pattern_type == AMLPatternType.STRUCTURING:
                flags.append("STRUCTURING_SUSPECTED")
            elif pattern.pattern_type == AMLPatternType.LAYERING:
                flags.append("LAYERING_SUSPECTED")
            elif pattern.pattern_type == AMLPatternType.HIGH_FREQUENCY:
                flags.append("HIGH_FREQUENCY")
            elif pattern.pattern_type == AMLPatternType.UNUSUAL_TIMING:
                flags.append("UNUSUAL_TIMING")
        
        # Add severity flags
        if aml_score >= 0.8:
            flags.append("HIGH_RISK_TRANSACTION")
        elif aml_score >= 0.6:
            flags.append("MEDIUM_RISK_TRANSACTION")
        
        return flags
    
    def _generate_explanation(self, patterns: List[AMLPattern], aml_score: float) -> str:
        """Generate human-readable explanation of AML analysis."""
        if not patterns:
            return f"Transaction analyzed with AML score {aml_score:.3f}. No suspicious patterns detected."
        
        pattern_descriptions = [pattern.description for pattern in patterns]
        explanation = f"AML score {aml_score:.3f}. Detected patterns: {', '.join(pattern_descriptions)}"
        
        return explanation
    
    def _create_error_score(self, transaction_id: str, error: str) -> AMLScore:
        """Create error AML score."""
        return AMLScore(
            transaction_id=transaction_id,
            aml_score=0.5,  # Neutral score on error
            risk_level=AMLRiskLevel.MEDIUM,
            detected_patterns=[],
            flags=["ANALYSIS_ERROR"],
            explanation=f"AML analysis failed: {error}",
            model_version=self.model_version,
            timestamp=datetime.utcnow()
        )
    
    async def get_aml_statistics(self) -> Dict[str, Any]:
        """Get AML service statistics."""
        return {
            "service_status": "healthy" if self.is_initialized else "unhealthy",
            "model_version": self.model_version,
            "pattern_types": len(AMLPatternType),
            "risk_levels": len(AMLRiskLevel),
            "thresholds": self.pattern_thresholds,
            "last_update": datetime.utcnow().isoformat()
        }
    
    async def update_pattern_thresholds(self, new_thresholds: Dict[AMLPatternType, float]) -> None:
        """Update AML pattern detection thresholds."""
        try:
            self.pattern_thresholds.update(new_thresholds)
            logger.info(f"Updated AML pattern thresholds: {new_thresholds}")
        except Exception as e:
            logger.error(f"Error updating AML thresholds: {e}")
            raise


# Global AML service instance
aml_service = AMLDetectionService()

"""
Credit Scoring service for risk assessment and loan decisions.
"""
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from src.config.settings import get_settings
from src.database.repositories import transaction_repo, wallet_repo
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class CreditRiskLevel(str, Enum):
    """Credit risk levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"


class CreditDecision(str, Enum):
    """Credit decision outcomes."""
    APPROVE = "approve"
    APPROVE_WITH_CONDITIONS = "approve_with_conditions"
    DECLINE = "decline"
    MANUAL_REVIEW = "manual_review"


@dataclass
class CreditScore:
    """Credit scoring result."""
    account_id: str
    credit_score: int
    risk_level: CreditRiskLevel
    decision: CreditDecision
    confidence: float
    factors: Dict[str, Any]
    explanation: str
    model_version: str
    timestamp: datetime


@dataclass
class CreditApplication:
    """Credit application data."""
    account_id: str
    requested_amount: float
    loan_purpose: str
    term_months: int
    monthly_income: float
    existing_debt: float
    employment_status: str
    credit_history_months: int


class CreditScoringService:
    """Credit scoring and risk assessment service."""
    
    def __init__(self):
        self.is_initialized = False
        self.model_version = "credit_v1.0"
        
        # Credit score ranges
        self.score_ranges = {
            CreditRiskLevel.EXCELLENT: (750, 850),
            CreditRiskLevel.GOOD: (700, 749),
            CreditRiskLevel.FAIR: (650, 699),
            CreditRiskLevel.POOR: (600, 649),
            CreditRiskLevel.VERY_POOR: (300, 599)
        }
        
        # Decision thresholds
        self.decision_thresholds = {
            CreditDecision.APPROVE: 0.8,
            CreditDecision.APPROVE_WITH_CONDITIONS: 0.6,
            CreditDecision.MANUAL_REVIEW: 0.4,
            CreditDecision.DECLINE: 0.0
        }
        
    async def initialize(self) -> None:
        """Initialize the credit scoring service."""
        try:
            await self._load_scoring_models()
            self.is_initialized = True
            logger.info("Credit Scoring Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize credit scoring service: {e}")
            raise
    
    async def _load_scoring_models(self) -> None:
        """Load credit scoring models and rules."""
        # In production, these would be loaded from ML model files
        self.scoring_factors = {
            "payment_history": 0.35,  # 35% weight
            "credit_utilization": 0.30,  # 30% weight
            "credit_length": 0.15,  # 15% weight
            "credit_mix": 0.10,  # 10% weight
            "new_credit": 0.10  # 10% weight
        }
        
        # Risk factors
        self.risk_factors = {
            "high_debt_to_income": 0.8,
            "recent_late_payments": 0.7,
            "high_credit_utilization": 0.6,
            "short_credit_history": 0.4,
            "frequent_applications": 0.5
        }
        
        logger.info(f"Loaded credit scoring models with {len(self.scoring_factors)} factors")
    
    async def score_credit_application(self, application: CreditApplication) -> CreditScore:
        """Score a credit application."""
        try:
            start_time = time.time()
            
            # Get account financial data
            account_data = await self._get_account_financial_data(application.account_id)
            
            # Calculate individual factor scores
            factor_scores = await self._calculate_factor_scores(application, account_data)
            
            # Calculate overall credit score
            credit_score = self._calculate_credit_score(factor_scores)
            
            # Determine risk level
            risk_level = self._get_risk_level(credit_score)
            
            # Make credit decision
            decision = await self._make_credit_decision(application, credit_score, factor_scores)
            
            # Calculate confidence
            confidence = self._calculate_confidence(factor_scores, application)
            
            # Generate explanation
            explanation = self._generate_explanation(factor_scores, credit_score, decision)
            
            result = CreditScore(
                account_id=application.account_id,
                credit_score=int(credit_score),
                risk_level=risk_level,
                decision=decision,
                confidence=confidence,
                factors=factor_scores,
                explanation=explanation,
                model_version=self.model_version,
                timestamp=datetime.utcnow()
            )
            
            logger.info(f"Credit scoring completed for account {application.account_id}: score={credit_score}, decision={decision}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error scoring credit application: {e}")
            return self._create_error_score(application.account_id, str(e))
    
    async def _get_account_financial_data(self, account_id: str) -> Dict[str, Any]:
        """Get financial data for account."""
        try:
            # This would query the database for account financial data
            # For now, simulate based on account ID
            
            # Simulate account data
            account_data = {
                "balance": np.random.uniform(1000, 50000),
                "monthly_income": np.random.uniform(3000, 15000),
                "transaction_count_30d": np.random.poisson(20),
                "avg_transaction_amount": np.random.uniform(100, 2000),
                "credit_history_months": np.random.uniform(12, 120),
                "late_payments_12m": np.random.poisson(1),
                "credit_applications_6m": np.random.poisson(2),
                "existing_loans": np.random.uniform(0, 3),
                "debt_to_income_ratio": np.random.uniform(0.1, 0.6)
            }
            
            return account_data
            
        except Exception as e:
            logger.error(f"Error getting account financial data: {e}")
            return {}
    
    async def _calculate_factor_scores(self, application: CreditApplication, account_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate individual factor scores."""
        try:
            factors = {}
            
            # Payment History (35%)
            late_payments = account_data.get("late_payments_12m", 0)
            if late_payments == 0:
                factors["payment_history"] = 0.95
            elif late_payments <= 2:
                factors["payment_history"] = 0.75
            elif late_payments <= 5:
                factors["payment_history"] = 0.50
            else:
                factors["payment_history"] = 0.20
            
            # Credit Utilization (30%)
            debt_to_income = account_data.get("debt_to_income_ratio", 0)
            if debt_to_income <= 0.2:
                factors["credit_utilization"] = 0.90
            elif debt_to_income <= 0.3:
                factors["credit_utilization"] = 0.75
            elif debt_to_income <= 0.4:
                factors["credit_utilization"] = 0.60
            elif debt_to_income <= 0.5:
                factors["credit_utilization"] = 0.40
            else:
                factors["credit_utilization"] = 0.20
            
            # Credit Length (15%)
            credit_history = account_data.get("credit_history_months", 0)
            if credit_history >= 84:  # 7+ years
                factors["credit_length"] = 0.90
            elif credit_history >= 60:  # 5+ years
                factors["credit_length"] = 0.75
            elif credit_history >= 36:  # 3+ years
                factors["credit_length"] = 0.60
            elif credit_history >= 24:  # 2+ years
                factors["credit_length"] = 0.45
            else:
                factors["credit_length"] = 0.25
            
            # Credit Mix (10%)
            existing_loans = account_data.get("existing_loans", 0)
            if existing_loans >= 3:
                factors["credit_mix"] = 0.80
            elif existing_loans >= 2:
                factors["credit_mix"] = 0.70
            elif existing_loans >= 1:
                factors["credit_mix"] = 0.60
            else:
                factors["credit_mix"] = 0.40
            
            # New Credit (10%)
            recent_applications = account_data.get("credit_applications_6m", 0)
            if recent_applications == 0:
                factors["new_credit"] = 0.90
            elif recent_applications <= 2:
                factors["new_credit"] = 0.70
            elif recent_applications <= 4:
                factors["new_credit"] = 0.50
            else:
                factors["new_credit"] = 0.30
            
            return factors
            
        except Exception as e:
            logger.error(f"Error calculating factor scores: {e}")
            return {}
    
    def _calculate_credit_score(self, factor_scores: Dict[str, Any]) -> float:
        """Calculate overall credit score from factor scores."""
        try:
            if not factor_scores:
                return 500  # Default middle score
            
            # Weighted average of factor scores
            weighted_score = 0
            total_weight = 0
            
            for factor, score in factor_scores.items():
                if factor in self.scoring_factors:
                    weight = self.scoring_factors[factor]
                    weighted_score += score * weight
                    total_weight += weight
            
            if total_weight > 0:
                # Convert to credit score range (300-850)
                normalized_score = weighted_score / total_weight
                credit_score = 300 + (normalized_score * 550)  # Scale to 300-850
            else:
                credit_score = 500
            
            return min(max(credit_score, 300), 850)
            
        except Exception as e:
            logger.error(f"Error calculating credit score: {e}")
            return 500
    
    def _get_risk_level(self, credit_score: float) -> CreditRiskLevel:
        """Convert credit score to risk level."""
        if credit_score >= 750:
            return CreditRiskLevel.EXCELLENT
        elif credit_score >= 700:
            return CreditRiskLevel.GOOD
        elif credit_score >= 650:
            return CreditRiskLevel.FAIR
        elif credit_score >= 600:
            return CreditRiskLevel.POOR
        else:
            return CreditRiskLevel.VERY_POOR
    
    async def _make_credit_decision(self, application: CreditApplication, credit_score: float, factor_scores: Dict[str, Any]) -> CreditDecision:
        """Make credit decision based on score and application."""
        try:
            # Calculate decision score (0-1)
            decision_score = credit_score / 850  # Normalize to 0-1
            
            # Adjust for application-specific factors
            debt_to_income = application.existing_debt / max(application.monthly_income, 1)
            if debt_to_income > 0.4:  # High debt-to-income
                decision_score *= 0.8
            
            # Adjust for requested amount
            income_ratio = application.requested_amount / max(application.monthly_income * 12, 1)
            if income_ratio > 0.3:  # Requesting more than 30% of annual income
                decision_score *= 0.9
            
            # Make decision based on thresholds
            if decision_score >= self.decision_thresholds[CreditDecision.APPROVE]:
                return CreditDecision.APPROVE
            elif decision_score >= self.decision_thresholds[CreditDecision.APPROVE_WITH_CONDITIONS]:
                return CreditDecision.APPROVE_WITH_CONDITIONS
            elif decision_score >= self.decision_thresholds[CreditDecision.MANUAL_REVIEW]:
                return CreditDecision.MANUAL_REVIEW
            else:
                return CreditDecision.DECLINE
                
        except Exception as e:
            logger.error(f"Error making credit decision: {e}")
            return CreditDecision.MANUAL_REVIEW
    
    def _calculate_confidence(self, factor_scores: Dict[str, Any], application: CreditApplication) -> float:
        """Calculate confidence in the credit decision."""
        try:
            # Base confidence on data completeness
            data_completeness = len(factor_scores) / len(self.scoring_factors)
            
            # Adjust for credit history length
            history_factor = min(application.credit_history_months / 60, 1.0)  # Normalize by 5 years
            
            # Adjust for income stability (simplified)
            income_stability = 0.8 if application.employment_status == "employed" else 0.6
            
            confidence = (data_completeness * 0.4) + (history_factor * 0.3) + (income_stability * 0.3)
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _generate_explanation(self, factor_scores: Dict[str, Any], credit_score: float, decision: CreditDecision) -> str:
        """Generate human-readable explanation of credit decision."""
        try:
            explanations = []
            
            # Add factor explanations
            for factor, score in factor_scores.items():
                if score < 0.5:
                    explanations.append(f"Low {factor.replace('_', ' ')} score")
                elif score > 0.8:
                    explanations.append(f"Strong {factor.replace('_', ' ')}")
            
            # Add decision explanation
            if decision == CreditDecision.APPROVE:
                explanations.append("Credit approved")
            elif decision == CreditDecision.APPROVE_WITH_CONDITIONS:
                explanations.append("Credit approved with conditions")
            elif decision == CreditDecision.MANUAL_REVIEW:
                explanations.append("Requires manual review")
            else:
                explanations.append("Credit declined")
            
            explanation = f"Credit score: {int(credit_score)}. " + ". ".join(explanations) + "."
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"Credit score: {int(credit_score)}. Decision: {decision.value}."
    
    def _create_error_score(self, account_id: str, error: str) -> CreditScore:
        """Create error credit score."""
        return CreditScore(
            account_id=account_id,
            credit_score=500,  # Neutral score on error
            risk_level=CreditRiskLevel.FAIR,
            decision=CreditDecision.MANUAL_REVIEW,
            confidence=0.0,
            factors={},
            explanation=f"Credit scoring failed: {error}",
            model_version=self.model_version,
            timestamp=datetime.utcnow()
        )
    
    async def get_credit_statistics(self) -> Dict[str, Any]:
        """Get credit scoring service statistics."""
        return {
            "service_status": "healthy" if self.is_initialized else "unhealthy",
            "model_version": self.model_version,
            "scoring_factors": len(self.scoring_factors),
            "risk_levels": len(CreditRiskLevel),
            "decision_types": len(CreditDecision),
            "score_range": (300, 850),
            "last_update": datetime.utcnow().isoformat()
        }


# Global credit scoring service instance
credit_service = CreditScoringService()

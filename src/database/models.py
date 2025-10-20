"""
Pydantic models for MongoDB collections.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId
from enum import Enum


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class APIKeyStatus(str, Enum):
    """API Key status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"


class TransactionStatus(str, Enum):
    """Transaction status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TransactionType(str, Enum):
    """Transaction type enumeration."""
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    PAYMENT = "payment"
    REFUND = "refund"


class APIKey(BaseModel):
    """API Key model for authentication."""
    
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    key: str = Field(..., description="The API key value")
    name: str = Field(..., description="Human-readable name for the key")
    client_id: str = Field(..., description="Client identifier")
    permissions: List[str] = Field(default=[], description="List of permissions")
    status: APIKeyStatus = Field(default=APIKeyStatus.ACTIVE)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(default=None)
    last_used_at: Optional[datetime] = Field(default=None)
    usage_count: int = Field(default=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "key": "ak_1234567890abcdef",
                "name": "Backend Service Key",
                "client_id": "backend_service",
                "permissions": ["read", "write", "admin"],
                "status": "active",
                "expires_at": "2024-12-31T23:59:59Z"
            }
        }


class Transaction(BaseModel):
    """Transaction model for banking operations."""
    
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    transaction_id: str = Field(..., description="Unique transaction identifier")
    sender_account: str = Field(..., description="Sender account ID")
    receiver_account: str = Field(..., description="Receiver account ID")
    amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field(default="USD", description="Currency code")
    transaction_type: TransactionType = Field(..., description="Type of transaction")
    status: TransactionStatus = Field(default=TransactionStatus.PENDING)
    channel: str = Field(..., description="Transaction channel (mobile, web, atm, etc.)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = Field(default=None)
    description: Optional[str] = Field(default=None)
    reference: Optional[str] = Field(default=None)
    
    # Fraud detection fields
    fraud_score: Optional[float] = Field(default=None, ge=0, le=1)
    fraud_decision: Optional[str] = Field(default=None)
    fraud_model_version: Optional[str] = Field(default=None)
    fraud_explanation: Optional[Dict[str, Any]] = Field(default=None)
    
    # AML fields
    aml_score: Optional[float] = Field(default=None, ge=0, le=1)
    aml_flags: List[str] = Field(default=[])
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "transaction_id": "txn_1234567890",
                "sender_account": "acc_123456",
                "receiver_account": "acc_789012",
                "amount": 150.75,
                "currency": "USD",
                "transaction_type": "transfer",
                "channel": "mobile_app",
                "description": "Payment for services"
            }
        }


class Wallet(BaseModel):
    """Wallet model for account balances."""
    
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    account_id: str = Field(..., description="Account identifier")
    balance: float = Field(default=0.0, ge=0, description="Current balance")
    currency: str = Field(default="USD", description="Currency code")
    available_balance: float = Field(default=0.0, ge=0, description="Available balance")
    frozen_balance: float = Field(default=0.0, ge=0, description="Frozen balance")
    
    # Risk assessment
    risk_level: str = Field(default="low", description="Risk level (low, medium, high)")
    risk_score: Optional[float] = Field(default=None, ge=0, le=1)
    
    # Transaction limits
    daily_limit: Optional[float] = Field(default=None)
    monthly_limit: Optional[float] = Field(default=None)
    daily_used: float = Field(default=0.0)
    monthly_used: float = Field(default=0.0)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_transaction_at: Optional[datetime] = Field(default=None)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "account_id": "acc_123456",
                "balance": 1000.50,
                "currency": "USD",
                "available_balance": 950.50,
                "frozen_balance": 50.0,
                "risk_level": "low"
            }
        }


class FraudScore(BaseModel):
    """Fraud score model for ML predictions."""
    
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    transaction_id: str = Field(..., description="Transaction ID")
    model_version: str = Field(..., description="Model version used")
    probability: float = Field(..., ge=0, le=1, description="Fraud probability")
    decision: str = Field(..., description="Decision (ALLOW, HOLD, BLOCK)")
    explanation: Dict[str, Any] = Field(default_factory=dict)
    features_used: List[str] = Field(default=[])
    latency_ms: int = Field(..., description="Processing latency in milliseconds")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

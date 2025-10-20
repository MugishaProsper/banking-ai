"""
Repository pattern for database operations with strict permissions.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.errors import DuplicateKeyError
import logging

from src.database.models import APIKey, Transaction, Wallet, FraudScore, APIKeyStatus, TransactionStatus
from src.database.connection import get_collection

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common database operations."""
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self._collection: Optional[AsyncIOMotorCollection] = None
    
    @property
    def collection(self) -> AsyncIOMotorCollection:
        """Get collection instance."""
        if not self._collection:
            self._collection = get_collection(self.collection_name)
        return self._collection
    
    async def find_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Find document by ID."""
        try:
            from bson import ObjectId
            result = await self.collection.find_one({"_id": ObjectId(document_id)})
            return result
        except Exception as e:
            logger.error(f"Error finding document by ID {document_id}: {e}")
            return None
    
    async def create(self, document: Dict[str, Any]) -> Optional[str]:
        """Create a new document."""
        try:
            document["created_at"] = datetime.utcnow()
            document["updated_at"] = datetime.utcnow()
            result = await self.collection.insert_one(document)
            return str(result.inserted_id)
        except DuplicateKeyError as e:
            logger.error(f"Duplicate key error creating document: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating document: {e}")
            return None
    
    async def update(self, document_id: str, update_data: Dict[str, Any]) -> bool:
        """Update document by ID."""
        try:
            from bson import ObjectId
            update_data["updated_at"] = datetime.utcnow()
            result = await self.collection.update_one(
                {"_id": ObjectId(document_id)},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            return False


class APIKeyRepository(BaseRepository):
    """Repository for API key operations."""
    
    def __init__(self):
        super().__init__("api_keys")
    
    async def find_by_key(self, api_key: str) -> Optional[APIKey]:
        """Find API key by key value."""
        try:
            result = await self.collection.find_one({"key": api_key})
            if result:
                return APIKey(**result)
            return None
        except Exception as e:
            logger.error(f"Error finding API key: {e}")
            return None
    
    async def validate_key(self, api_key: str) -> Optional[APIKey]:
        """Validate API key and check if it's active."""
        key_doc = await self.find_by_key(api_key)
        if not key_doc:
            return None
        
        # Check if key is active
        if key_doc.status != APIKeyStatus.ACTIVE:
            logger.warning(f"API key {api_key} is not active: {key_doc.status}")
            return None
        
        # Check expiration
        if key_doc.expires_at and key_doc.expires_at < datetime.utcnow():
            logger.warning(f"API key {api_key} has expired")
            return None
        
        # Update last used timestamp
        await self.update_last_used(api_key)
        return key_doc
    
    async def update_last_used(self, api_key: str) -> bool:
        """Update last used timestamp and increment usage count."""
        try:
            result = await self.collection.update_one(
                {"key": api_key},
                {
                    "$set": {"last_used_at": datetime.utcnow()},
                    "$inc": {"usage_count": 1}
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating API key usage: {e}")
            return False
    
    async def create_key(self, key_data: Dict[str, Any]) -> Optional[str]:
        """Create a new API key."""
        return await self.create(key_data)
    
    async def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        try:
            result = await self.collection.update_one(
                {"key": api_key},
                {"$set": {"status": APIKeyStatus.REVOKED, "updated_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error revoking API key: {e}")
            return False


class TransactionRepository(BaseRepository):
    """Repository for transaction operations."""
    
    def __init__(self):
        super().__init__("transactions")
    
    async def find_by_transaction_id(self, transaction_id: str) -> Optional[Transaction]:
        """Find transaction by transaction ID."""
        try:
            result = await self.collection.find_one({"transaction_id": transaction_id})
            if result:
                return Transaction(**result)
            return None
        except Exception as e:
            logger.error(f"Error finding transaction {transaction_id}: {e}")
            return None
    
    async def create_transaction(self, transaction_data: Dict[str, Any]) -> Optional[str]:
        """Create a new transaction."""
        return await self.create(transaction_data)
    
    async def update_status(self, transaction_id: str, status: TransactionStatus, 
                          processed_at: Optional[datetime] = None) -> bool:
        """Update transaction status."""
        update_data = {"status": status}
        if processed_at:
            update_data["processed_at"] = processed_at
        
        return await self.update(transaction_id, update_data)
    
    async def add_fraud_score(self, transaction_id: str, fraud_data: Dict[str, Any]) -> bool:
        """Add fraud detection results to transaction."""
        try:
            from bson import ObjectId
            result = await self.collection.update_one(
                {"transaction_id": transaction_id},
                {"$set": fraud_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error adding fraud score to transaction {transaction_id}: {e}")
            return False
    
    async def get_recent_transactions(self, account_id: str, limit: int = 10) -> List[Transaction]:
        """Get recent transactions for an account."""
        try:
            cursor = self.collection.find(
                {"$or": [{"sender_account": account_id}, {"receiver_account": account_id}]}
            ).sort("timestamp", -1).limit(limit)
            
            transactions = []
            async for doc in cursor:
                transactions.append(Transaction(**doc))
            return transactions
        except Exception as e:
            logger.error(f"Error getting recent transactions for {account_id}: {e}")
            return []


class WalletRepository(BaseRepository):
    """Repository for wallet operations."""
    
    def __init__(self):
        super().__init__("wallets")
    
    async def find_by_account_id(self, account_id: str) -> Optional[Wallet]:
        """Find wallet by account ID."""
        try:
            result = await self.collection.find_one({"account_id": account_id})
            if result:
                return Wallet(**result)
            return None
        except Exception as e:
            logger.error(f"Error finding wallet for account {account_id}: {e}")
            return None
    
    async def get_balance(self, account_id: str) -> Optional[float]:
        """Get account balance."""
        wallet = await self.find_by_account_id(account_id)
        return wallet.balance if wallet else None
    
    async def update_balance(self, account_id: str, new_balance: float) -> bool:
        """Update account balance."""
        try:
            result = await self.collection.update_one(
                {"account_id": account_id},
                {
                    "$set": {
                        "balance": new_balance,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating balance for account {account_id}: {e}")
            return False
    
    async def freeze_amount(self, account_id: str, amount: float) -> bool:
        """Freeze amount in account."""
        try:
            result = await self.collection.update_one(
                {"account_id": account_id},
                {
                    "$inc": {
                        "frozen_balance": amount,
                        "available_balance": -amount
                    },
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error freezing amount for account {account_id}: {e}")
            return False
    
    async def unfreeze_amount(self, account_id: str, amount: float) -> bool:
        """Unfreeze amount in account."""
        try:
            result = await self.collection.update_one(
                {"account_id": account_id},
                {
                    "$inc": {
                        "frozen_balance": -amount,
                        "available_balance": amount
                    },
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error unfreezing amount for account {account_id}: {e}")
            return False


class FraudScoreRepository(BaseRepository):
    """Repository for fraud score operations."""
    
    def __init__(self):
        super().__init__("fraud_scores")
    
    async def create_fraud_score(self, fraud_data: Dict[str, Any]) -> Optional[str]:
        """Create a new fraud score record."""
        return await self.create(fraud_data)
    
    async def get_scores_by_transaction(self, transaction_id: str) -> List[FraudScore]:
        """Get fraud scores for a transaction."""
        try:
            cursor = self.collection.find({"transaction_id": transaction_id})
            scores = []
            async for doc in cursor:
                scores.append(FraudScore(**doc))
            return scores
        except Exception as e:
            logger.error(f"Error getting fraud scores for transaction {transaction_id}: {e}")
            return []


# Repository instances
api_key_repo = APIKeyRepository()
transaction_repo = TransactionRepository()
wallet_repo = WalletRepository()
fraud_score_repo = FraudScoreRepository()

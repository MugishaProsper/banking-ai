"""
MongoDB database connection and lifecycle management.
"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure
import asyncio
from typing import Optional
import logging

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class DatabaseManager:
    """MongoDB database manager with connection lifecycle."""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self._connection_string = settings.mongo_uri
        self._database_name = settings.mongo_db_name
        
    async def connect(self) -> None:
        """Establish MongoDB connection."""
        try:
            self.client = AsyncIOMotorClient(
                self._connection_string,
                maxPoolSize=settings.mongo_max_pool_size,
                minPoolSize=settings.mongo_min_pool_size,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=20000,
            )
            
            # Test connection
            await self.client.admin.command('ping')
            self.database = self.client[self._database_name]
            
            logger.info(f"Connected to MongoDB database: {self._database_name}")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def health_check(self) -> bool:
        """Check database health."""
        try:
            if not self.client:
                return False
            await self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_collection(self, collection_name: str):
        """Get a collection from the database."""
        if not self.database:
            raise RuntimeError("Database not connected")
        return self.database[collection_name]


# Global database manager instance
db_manager = DatabaseManager()


async def get_database() -> AsyncIOMotorDatabase:
    """Get the database instance."""
    if not db_manager.database:
        await db_manager.connect()
    return db_manager.database


def get_collection(collection_name: str):
    """Get a collection from the database."""
    return db_manager.get_collection(collection_name)

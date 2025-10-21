"""
Feast Feature Store client wrapper for online feature retrieval.
"""
from typing import Dict, List, Any
import asyncio
import logging

from feast import FeatureStore

from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FeatureStoreClient:
    """Wrapper around Feast FeatureStore for async usage in FastAPI."""

    def __init__(self):
        self._store = None
        self._ready = False

    async def initialize(self) -> None:
        """Initialize the Feast FeatureStore client."""
        try:
            # Feast is synchronous; initialize in threadpool if needed
            self._store = FeatureStore(repo_path=settings.feature_store_repo_path)
            # Perform a lightweight operation to verify registry load
            await asyncio.to_thread(lambda: self._store.list_feature_views())
            self._ready = True
            logger.info("Feast FeatureStore initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Feast: {e}")
            self._ready = False

    async def get_online_features(self, *, entity_rows: List[Dict[str, Any]], feature_refs: List[str]) -> Dict[str, Any]:
        """Fetch online features for given entities.

        Returns a flat dict of feature name to value for the first entity row.
        """
        if not self._store:
            raise RuntimeError("FeatureStore not initialized")

        try:
            result = await asyncio.to_thread(
                lambda: self._store.get_online_features(
                    features=feature_refs,
                    entity_rows=entity_rows
                ).to_dict()
            )
            # Flatten: pick first entity's values
            flattened: Dict[str, Any] = {}
            for k, v in result.items():
                if isinstance(v, list) and len(v) > 0:
                    flattened[k.split(":")[-1]] = v[0]
                else:
                    flattened[k.split(":")[-1]] = v
            return flattened
        except Exception as e:
            logger.error(f"Feast get_online_features failed: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if FeatureStore client is ready."""
        if not self._ready:
            return False
        try:
            await asyncio.to_thread(lambda: self._store.list_feature_views())
            return True
        except Exception:
            return False


# Global singleton
feature_store_client = FeatureStoreClient()



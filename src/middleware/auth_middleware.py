"""
API Key authentication middleware for AI Banking Microservice.
"""
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging
from typing import Optional

from src.config.settings import get_settings
from src.database.repositories import api_key_repo
from src.database.models import APIKey

logger = logging.getLogger(__name__)
settings = get_settings()


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.api_key_header = settings.api_key_header_name
        self.excluded_paths = {
            "/health",
            "/health/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/metrics"
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request with API key validation."""
        
        # Skip authentication for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Extract API key from headers
        api_key = self._extract_api_key(request)
        if not api_key:
            return self._create_error_response(
                "API key is required",
                status.HTTP_401_UNAUTHORIZED
            )
        
        # Validate API key
        try:
            validated_key = await api_key_repo.validate_key(api_key)
            if not validated_key:
                return self._create_error_response(
                    "Invalid or expired API key",
                    status.HTTP_401_UNAUTHORIZED
                )
            
            # Attach client context to request state
            request.state.api_key = validated_key
            request.state.client_id = validated_key.client_id
            request.state.permissions = validated_key.permissions
            
            # Log successful authentication
            logger.info(f"API key validated for client: {validated_key.client_id}")
            
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return self._create_error_response(
                "Authentication error",
                status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        return response
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request headers."""
        # Try different header names
        api_key = request.headers.get(self.api_key_header)
        if not api_key:
            api_key = request.headers.get("Authorization")
            if api_key and api_key.startswith("Bearer "):
                api_key = api_key[7:]  # Remove "Bearer " prefix
        
        return api_key
    
    def _create_error_response(self, message: str, status_code: int) -> JSONResponse:
        """Create standardized error response."""
        return JSONResponse(
            status_code=status_code,
            content={
                "error": message,
                "status_code": status_code,
                "timestamp": self._get_timestamp(),
                "service": settings.service_name
            }
        )
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()


def check_permission(request: Request, required_permission: str) -> bool:
    """Check if request has required permission."""
    permissions = getattr(request.state, 'permissions', [])
    return required_permission in permissions


def get_client_id(request: Request) -> Optional[str]:
    """Get client ID from request state."""
    return getattr(request.state, 'client_id', None)


def get_api_key(request: Request) -> Optional[APIKey]:
    """Get API key object from request state."""
    return getattr(request.state, 'api_key', None)

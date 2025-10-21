"""
Configuration management for AI Banking Microservice.
"""
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database Configuration
    mongo_uri: str = Field(default="mongodb://localhost:27017", env="MONGO_URI")
    mongo_db_name: str = Field(default="banking_db", env="MONGO_DB_NAME")
    mongo_max_pool_size: int = Field(default=10, env="MONGO_MAX_POOL_SIZE")
    mongo_min_pool_size: int = Field(default=1, env="MONGO_MIN_POOL_SIZE")
    
    # Backend Service Configuration
    backend_url: str = Field(default="http://localhost:5000", env="BACKEND_URL")
    backend_timeout: int = Field(default=30, env="BACKEND_TIMEOUT")
    
    # Service Configuration
    service_name: str = Field(default="fraud-ai-microservice", env="SERVICE_NAME")
    service_version: str = Field(default="1.0.0", env="SERVICE_VERSION")
    service_host: str = Field(default="0.0.0.0", env="SERVICE_HOST")
    service_port: int = Field(default=8000, env="SERVICE_PORT")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Security Configuration
    api_key_secret: str = Field(default="your-secret-key-here", env="API_KEY_SECRET")
    jwt_secret_key: str = Field(default="your-jwt-secret-here", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Alerting Configuration
    email_smtp_server: str = Field(default="smtp.gmail.com", env="EMAIL_SMTP_SERVER")
    email_smtp_port: int = Field(default=587, env="EMAIL_SMTP_PORT")
    email_username: str = Field(default="", env="EMAIL_USERNAME")
    email_password: str = Field(default="", env="EMAIL_PASSWORD")
    email_from: str = Field(default="", env="EMAIL_FROM")
    email_to_emails: str = Field(default="", env="EMAIL_TO_EMAILS")
    
    slack_webhook_url: str = Field(default="", env="SLACK_WEBHOOK_URL")
    slack_channel: str = Field(default="#alerts", env="SLACK_CHANNEL")
    slack_username: str = Field(default="AI Banking Bot", env="SLACK_USERNAME")
    
    webhook_url: str = Field(default="", env="WEBHOOK_URL")
    webhook_headers: str = Field(default="{}", env="WEBHOOK_HEADERS")
    
    pagerduty_integration_key: str = Field(default="", env="PAGERDUTY_INTEGRATION_KEY")
    
    # Tracing Configuration
    jaeger_endpoint: str = Field(default="", env="JAEGER_ENDPOINT")
    jaeger_agent_host: str = Field(default="localhost", env="JAEGER_AGENT_HOST")
    jaeger_agent_port: int = Field(default=14268, env="JAEGER_AGENT_PORT")
    tracing_sampling_rate: float = Field(default=0.1, env="TRACING_SAMPLING_RATE")
    
    # Feature Store Configuration (Feast)
    feast_repo_path: str = Field(default="./feature_repo", env="FEAST_REPO_PATH")
    feast_provider: str = Field(default="local", env="FEAST_PROVIDER")
    feast_registry: str = Field(default="./feature_repo/registry.db", env="FEAST_REGISTRY")
    feast_project: str = Field(default="fraud_ai", env="FEAST_PROJECT")
    feast_online_host: str = Field(default="localhost", env="FEAST_ONLINE_HOST")
    feast_online_port: int = Field(default=6379, env="FEAST_ONLINE_PORT")
    feature_store_timeout: int = Field(default=10, env="FEATURE_STORE_TIMEOUT")
    
    # Kafka Configuration
    kafka_bootstrap_servers: str = Field(default="localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS")
    kafka_group_id: str = Field(default="ai-banking-service", env="KAFKA_GROUP_ID")
    kafka_auto_offset_reset: str = Field(default="earliest", env="KAFKA_AUTO_OFFSET_RESET")
    kafka_txn_topic: str = Field(default="transactions", env="KAFKA_TXN_TOPIC")
    kafka_fraud_score_topic: str = Field(default="fraud_scores", env="KAFKA_FRAUD_SCORE_TOPIC")
    kafka_fraud_alert_topic: str = Field(default="fraud_alerts", env="KAFKA_FRAUD_ALERT_TOPIC")
    kafka_fraud_feedback_topic: str = Field(default="fraud_feedback", env="KAFKA_FRAUD_FEEDBACK_TOPIC")
    kafka_model_metrics_topic: str = Field(default="model_metrics", env="KAFKA_MODEL_METRICS_TOPIC")
    
    # Model Configuration
    model_cache_ttl: int = Field(default=3600, env="MODEL_CACHE_TTL")
    model_fallback_enabled: bool = Field(default=True, env="MODEL_FALLBACK_ENABLED")
    
    # API Key Validation
    api_key_header_name: str = Field(default="X-API-Key", env="API_KEY_HEADER_NAME")
    
    # CORS Configuration
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "protected_namespaces": ("settings_",)
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings

"""
Prometheus metrics collection and export for comprehensive monitoring.
"""
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict, deque
import json

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info, 
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, REGISTRY
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily, HistogramMetricFamily

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class CustomMetric:
    """Custom metric definition."""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str]
    value: float
    timestamp: datetime


class PrometheusMetricsService:
    """Prometheus metrics collection and export service."""
    
    def __init__(self):
        self.is_initialized = False
        self.registry = CollectorRegistry()
        self._start_time = time.time()
        
        # Standard metrics
        self._init_standard_metrics()
        
        # Custom metrics storage
        self.custom_metrics: Dict[str, CustomMetric] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Business metrics
        self._init_business_metrics()
        
        # ML model metrics
        self._init_ml_metrics()
        
    def _init_standard_metrics(self) -> None:
        """Initialize standard application metrics."""
        # HTTP metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Database metrics
        self.db_connections_active = Gauge(
            'db_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.db_operations_total = Counter(
            'db_operations_total',
            'Total database operations',
            ['operation', 'collection', 'status'],
            registry=self.registry
        )
        
        self.db_operation_duration_seconds = Histogram(
            'db_operation_duration_seconds',
            'Database operation duration in seconds',
            ['operation', 'collection'],
            registry=self.registry
        )
        
        # Service metrics
        self.service_uptime_seconds = Gauge(
            'service_uptime_seconds',
            'Service uptime in seconds',
            registry=self.registry
        )
        
        self.service_memory_usage_bytes = Gauge(
            'service_memory_usage_bytes',
            'Service memory usage in bytes',
            registry=self.registry
        )
        
        self.service_cpu_usage_percent = Gauge(
            'service_cpu_usage_percent',
            'Service CPU usage percentage',
            registry=self.registry
        )
        
    def _init_business_metrics(self) -> None:
        """Initialize business-specific metrics."""
        # Transaction metrics
        self.transactions_total = Counter(
            'transactions_total',
            'Total transactions processed',
            ['status', 'type', 'channel'],
            registry=self.registry
        )
        
        self.transaction_amount_total = Counter(
            'transaction_amount_total',
            'Total transaction amount',
            ['currency', 'type'],
            registry=self.registry
        )
        
        self.transaction_processing_duration_seconds = Histogram(
            'transaction_processing_duration_seconds',
            'Transaction processing duration in seconds',
            ['type', 'status'],
            registry=self.registry
        )
        
        # Fraud detection metrics
        self.fraud_detections_total = Counter(
            'fraud_detections_total',
            'Total fraud detections',
            ['model_type', 'decision', 'risk_level'],
            registry=self.registry
        )
        
        self.fraud_score_distribution = Histogram(
            'fraud_score_distribution',
            'Distribution of fraud scores',
            ['model_type'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        self.fraud_false_positives_total = Counter(
            'fraud_false_positives_total',
            'Total false positives',
            ['model_type'],
            registry=self.registry
        )
        
        self.fraud_false_negatives_total = Counter(
            'fraud_false_negatives_total',
            'Total false negatives',
            ['model_type'],
            registry=self.registry
        )
        
        # AML metrics
        self.aml_patterns_detected_total = Counter(
            'aml_patterns_detected_total',
            'Total AML patterns detected',
            ['pattern_type', 'risk_level'],
            registry=self.registry
        )
        
        self.aml_analysis_duration_seconds = Histogram(
            'aml_analysis_duration_seconds',
            'AML analysis duration in seconds',
            ['pattern_type'],
            registry=self.registry
        )
        
        # Credit scoring metrics
        self.credit_applications_total = Counter(
            'credit_applications_total',
            'Total credit applications',
            ['decision', 'risk_level'],
            registry=self.registry
        )
        
        self.credit_score_distribution = Histogram(
            'credit_score_distribution',
            'Distribution of credit scores',
            buckets=[300, 400, 500, 600, 700, 750, 800, 850],
            registry=self.registry
        )
        
        self.credit_processing_duration_seconds = Histogram(
            'credit_processing_duration_seconds',
            'Credit processing duration in seconds',
            registry=self.registry
        )
        
    def _init_ml_metrics(self) -> None:
        """Initialize ML model-specific metrics."""
        # Model performance metrics
        self.model_predictions_total = Counter(
            'model_predictions_total',
            'Total model predictions',
            ['model_type', 'model_version', 'status'],
            registry=self.registry
        )
        
        self.model_prediction_duration_seconds = Histogram(
            'model_prediction_duration_seconds',
            'Model prediction duration in seconds',
            ['model_type', 'model_version'],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'model_accuracy',
            'Model accuracy',
            ['model_type', 'model_version'],
            registry=self.registry
        )
        
        self.model_precision = Gauge(
            'model_precision',
            'Model precision',
            ['model_type', 'model_version'],
            registry=self.registry
        )
        
        self.model_recall = Gauge(
            'model_recall',
            'Model recall',
            ['model_type', 'model_version'],
            registry=self.registry
        )
        
        self.model_f1_score = Gauge(
            'model_f1_score',
            'Model F1 score',
            ['model_type', 'model_version'],
            registry=self.registry
        )
        
        # Model retraining metrics
        self.model_retraining_total = Counter(
            'model_retraining_total',
            'Total model retraining events',
            ['model_type', 'trigger', 'status'],
            registry=self.registry
        )
        
        self.model_retraining_duration_seconds = Histogram(
            'model_retraining_duration_seconds',
            'Model retraining duration in seconds',
            ['model_type'],
            registry=self.registry
        )
        
        self.model_deployments_total = Counter(
            'model_deployments_total',
            'Total model deployments',
            ['model_type', 'strategy', 'status'],
            registry=self.registry
        )
        
        # Feature store metrics
        self.feature_requests_total = Counter(
            'feature_requests_total',
            'Total feature requests',
            ['feature_name', 'status'],
            registry=self.registry
        )
        
        self.feature_request_duration_seconds = Histogram(
            'feature_request_duration_seconds',
            'Feature request duration in seconds',
            ['feature_name'],
            registry=self.registry
        )
        
        # Kafka metrics
        self.kafka_messages_produced_total = Counter(
            'kafka_messages_produced_total',
            'Total Kafka messages produced',
            ['topic', 'status'],
            registry=self.registry
        )
        
        self.kafka_messages_consumed_total = Counter(
            'kafka_messages_consumed_total',
            'Total Kafka messages consumed',
            ['topic', 'status'],
            registry=self.registry
        )
        
        self.kafka_consumer_lag_seconds = Gauge(
            'kafka_consumer_lag_seconds',
            'Kafka consumer lag in seconds',
            ['topic', 'partition'],
            registry=self.registry
        )
        
    async def initialize(self) -> None:
        """Initialize the metrics service."""
        try:
            # Start background metrics collection
            asyncio.create_task(self._collect_system_metrics())
            
            self.is_initialized = True
            logger.info("Prometheus Metrics Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics service: {e}")
            raise
    
    async def _collect_system_metrics(self) -> None:
        """Collect system metrics in background."""
        while True:
            try:
                # Update uptime
                uptime = time.time() - self._start_time
                self.service_uptime_seconds.set(uptime)
                
                # Update memory usage (simplified)
                import psutil
                memory_info = psutil.virtual_memory()
                self.service_memory_usage_bytes.set(memory_info.used)
                
                # Update CPU usage (simplified)
                cpu_percent = psutil.cpu_percent(interval=1)
                self.service_cpu_usage_percent.set(cpu_percent)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(30)
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float) -> None:
        """Record HTTP request metrics."""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_db_operation(self, operation: str, collection: str, status: str, duration: float) -> None:
        """Record database operation metrics."""
        self.db_operations_total.labels(
            operation=operation,
            collection=collection,
            status=status
        ).inc()
        
        self.db_operation_duration_seconds.labels(
            operation=operation,
            collection=collection
        ).observe(duration)
    
    def record_transaction(self, status: str, transaction_type: str, channel: str, amount: float, currency: str = "USD") -> None:
        """Record transaction metrics."""
        self.transactions_total.labels(
            status=status,
            type=transaction_type,
            channel=channel
        ).inc()
        
        self.transaction_amount_total.labels(
            currency=currency,
            type=transaction_type
        ).inc(amount)
    
    def record_fraud_detection(self, model_type: str, decision: str, risk_level: str, score: float) -> None:
        """Record fraud detection metrics."""
        self.fraud_detections_total.labels(
            model_type=model_type,
            decision=decision,
            risk_level=risk_level
        ).inc()
        
        self.fraud_score_distribution.labels(
            model_type=model_type
        ).observe(score)
    
    def record_aml_pattern(self, pattern_type: str, risk_level: str, duration: float) -> None:
        """Record AML pattern detection metrics."""
        self.aml_patterns_detected_total.labels(
            pattern_type=pattern_type,
            risk_level=risk_level
        ).inc()
        
        self.aml_analysis_duration_seconds.labels(
            pattern_type=pattern_type
        ).observe(duration)
    
    def record_credit_application(self, decision: str, risk_level: str, score: int, duration: float) -> None:
        """Record credit application metrics."""
        self.credit_applications_total.labels(
            decision=decision,
            risk_level=risk_level
        ).inc()
        
        self.credit_score_distribution.observe(score)
        
        self.credit_processing_duration_seconds.observe(duration)
    
    def record_model_prediction(self, model_type: str, model_version: str, status: str, duration: float) -> None:
        """Record model prediction metrics."""
        self.model_predictions_total.labels(
            model_type=model_type,
            model_version=model_version,
            status=status
        ).inc()
        
        self.model_prediction_duration_seconds.labels(
            model_type=model_type,
            model_version=model_version
        ).observe(duration)
    
    def update_model_performance(self, model_type: str, model_version: str, 
                               accuracy: float, precision: float, recall: float, f1_score: float) -> None:
        """Update model performance metrics."""
        self.model_accuracy.labels(
            model_type=model_type,
            model_version=model_version
        ).set(accuracy)
        
        self.model_precision.labels(
            model_type=model_type,
            model_version=model_version
        ).set(precision)
        
        self.model_recall.labels(
            model_type=model_type,
            model_version=model_version
        ).set(recall)
        
        self.model_f1_score.labels(
            model_type=model_type,
            model_version=model_version
        ).set(f1_score)
    
    def record_model_retraining(self, model_type: str, trigger: str, status: str, duration: float) -> None:
        """Record model retraining metrics."""
        self.model_retraining_total.labels(
            model_type=model_type,
            trigger=trigger,
            status=status
        ).inc()
        
        self.model_retraining_duration_seconds.labels(
            model_type=model_type
        ).observe(duration)
    
    def record_model_deployment(self, model_type: str, strategy: str, status: str) -> None:
        """Record model deployment metrics."""
        self.model_deployments_total.labels(
            model_type=model_type,
            strategy=strategy,
            status=status
        ).inc()
    
    def record_feature_request(self, feature_name: str, status: str, duration: float) -> None:
        """Record feature store request metrics."""
        self.feature_requests_total.labels(
            feature_name=feature_name,
            status=status
        ).inc()
        
        self.feature_request_duration_seconds.labels(
            feature_name=feature_name
        ).observe(duration)
    
    def record_kafka_message(self, topic: str, status: str, is_producer: bool = True) -> None:
        """Record Kafka message metrics."""
        if is_producer:
            self.kafka_messages_produced_total.labels(
                topic=topic,
                status=status
            ).inc()
        else:
            self.kafka_messages_consumed_total.labels(
                topic=topic,
                status=status
            ).inc()
    
    def update_kafka_consumer_lag(self, topic: str, partition: int, lag_seconds: float) -> None:
        """Update Kafka consumer lag metrics."""
        self.kafka_consumer_lag_seconds.labels(
            topic=topic,
            partition=str(partition)
        ).set(lag_seconds)
    
    def add_custom_metric(self, name: str, description: str, metric_type: MetricType, 
                         labels: List[str], value: float) -> None:
        """Add a custom metric."""
        metric = CustomMetric(
            name=name,
            description=description,
            metric_type=metric_type,
            labels=labels,
            value=value,
            timestamp=datetime.utcnow()
        )
        
        self.custom_metrics[name] = metric
        self.metric_history[name].append(metric)
    
    def get_metrics_export(self) -> str:
        """Get Prometheus metrics export."""
        try:
            return generate_latest(self.registry)
        except Exception as e:
            logger.error(f"Error generating metrics export: {e}")
            return ""
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for monitoring dashboard."""
        try:
            summary = {
                "service_uptime_seconds": self.service_uptime_seconds._value._value,
                "service_memory_usage_bytes": self.service_memory_usage_bytes._value._value,
                "service_cpu_usage_percent": self.service_cpu_usage_percent._value._value,
                "custom_metrics_count": len(self.custom_metrics),
                "metric_history_size": sum(len(history) for history in self.metric_history.values()),
                "last_update": datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status based on metrics."""
        try:
            # Check if service is responsive
            uptime = self.service_uptime_seconds._value._value
            memory_usage = self.service_memory_usage_bytes._value._value
            cpu_usage = self.service_cpu_usage_percent._value._value
            
            # Health thresholds
            memory_threshold = 1024 * 1024 * 1024  # 1GB
            cpu_threshold = 80.0  # 80%
            
            health_status = "healthy"
            issues = []
            
            if memory_usage > memory_threshold:
                health_status = "warning"
                issues.append(f"High memory usage: {memory_usage / (1024*1024*1024):.2f}GB")
            
            if cpu_usage > cpu_threshold:
                health_status = "warning"
                issues.append(f"High CPU usage: {cpu_usage:.1f}%")
            
            if uptime < 60:  # Less than 1 minute uptime
                health_status = "unhealthy"
                issues.append("Service recently started")
            
            return {
                "status": health_status,
                "uptime_seconds": uptime,
                "memory_usage_bytes": memory_usage,
                "cpu_usage_percent": cpu_usage,
                "issues": issues,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global metrics service instance
metrics_service = PrometheusMetricsService()

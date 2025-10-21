"""
Distributed tracing service with OpenTelemetry integration.
"""
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json
import uuid
from contextlib import asynccontextmanager

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
# from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.pymongo import PymongoInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class TraceStatus(str, Enum):
    """Trace status."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class TraceContext:
    """Trace context information."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: TraceStatus = TraceStatus.SUCCESS
    tags: Dict[str, Any] = None
    logs: List[Dict[str, Any]] = None


class DistributedTracingService:
    """Distributed tracing service with OpenTelemetry."""
    
    def __init__(self):
        self.is_initialized = False
        self.tracer = None
        self.traces: Dict[str, TraceContext] = {}
        self.span_processors = []
        
        # Tracing configuration
        self.tracing_config = {
            "service_name": settings.service_name,
            "service_version": settings.service_version,
            "jaeger_endpoint": settings.jaeger_endpoint,
            "jaeger_agent_host": settings.jaeger_agent_host,
            "jaeger_agent_port": settings.jaeger_agent_port,
            "sampling_rate": settings.tracing_sampling_rate,
            "max_export_batch_size": 512,
            "export_timeout_millis": 30000,
            "schedule_delay_millis": 5000
        }
        
    async def initialize(self) -> None:
        """Initialize the tracing service."""
        try:
            # Set up OpenTelemetry
            await self._setup_opentelemetry()
            
            # Initialize instrumentations
            await self._setup_instrumentations()
            
            # Start background trace processing
            asyncio.create_task(self._process_traces())
            
            self.is_initialized = True
            logger.info("Distributed Tracing Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracing service: {e}")
            raise
    
    async def _setup_opentelemetry(self) -> None:
        """Set up OpenTelemetry tracing."""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.tracing_config["service_name"],
                "service.version": self.tracing_config["service_version"],
                "deployment.environment": settings.environment
            })
            
            # Set up tracer provider
            trace.set_tracer_provider(TracerProvider(resource=resource))
            
            # Get tracer
            self.tracer = trace.get_tracer(__name__)
            
            # Set up Jaeger exporter
            if self.tracing_config["jaeger_endpoint"]:
                jaeger_exporter = JaegerExporter(
                    agent_host_name=self.tracing_config["jaeger_agent_host"],
                    agent_port=self.tracing_config["jaeger_agent_port"],
                    collector_endpoint=self.tracing_config["jaeger_endpoint"]
                )
                
                # Add span processor
                span_processor = BatchSpanProcessor(jaeger_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)
                self.span_processors.append(span_processor)
            
            logger.info("OpenTelemetry tracing configured")
            
        except Exception as e:
            logger.error(f"Error setting up OpenTelemetry: {e}")
            raise
    
    async def _setup_instrumentations(self) -> None:
        """Set up automatic instrumentations."""
        try:
            # FastAPI instrumentation
            FastAPIInstrumentor.instrument_app = self._instrument_fastapi
            
            # HTTP client instrumentation (commented out due to compatibility issues)
            # HTTPXClientInstrumentor().instrument()
            
            # Database instrumentations
            PymongoInstrumentor().instrument()
            RedisInstrumentor().instrument()
            
            logger.info("OpenTelemetry instrumentations configured")
            
        except Exception as e:
            logger.error(f"Error setting up instrumentations: {e}")
    
    def _instrument_fastapi(self, app):
        """Instrument FastAPI application."""
        try:
            FastAPIInstrumentor.instrument_app(app)
            logger.info("FastAPI application instrumented for tracing")
        except Exception as e:
            logger.error(f"Error instrumenting FastAPI: {e}")
    
    async def _process_traces(self) -> None:
        """Process traces in background."""
        while True:
            try:
                # Process any pending traces
                await self._export_traces()
                await asyncio.sleep(5)  # Process every 5 seconds
            except Exception as e:
                logger.error(f"Error processing traces: {e}")
                await asyncio.sleep(5)
    
    async def _export_traces(self) -> None:
        """Export traces to external systems."""
        try:
            # This would export traces to Jaeger or other backends
            # For now, just log the trace count
            if self.traces:
                logger.debug(f"Processing {len(self.traces)} traces")
                
        except Exception as e:
            logger.error(f"Error exporting traces: {e}")
    
    def start_trace(self, operation_name: str, parent_trace_id: Optional[str] = None, 
                   tags: Optional[Dict[str, Any]] = None) -> TraceContext:
        """Start a new trace."""
        try:
            trace_id = str(uuid.uuid4())
            span_id = str(uuid.uuid4())
            
            trace_context = TraceContext(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_trace_id,
                operation_name=operation_name,
                start_time=datetime.utcnow(),
                tags=tags or {},
                logs=[]
            )
            
            self.traces[trace_id] = trace_context
            
            logger.debug(f"Started trace {trace_id} for operation {operation_name}")
            return trace_context
            
        except Exception as e:
            logger.error(f"Error starting trace: {e}")
            return None
    
    def end_trace(self, trace_id: str, status: TraceStatus = TraceStatus.SUCCESS, 
                  error_message: Optional[str] = None) -> None:
        """End a trace."""
        try:
            if trace_id not in self.traces:
                logger.warning(f"Trace {trace_id} not found")
                return
            
            trace_context = self.traces[trace_id]
            trace_context.end_time = datetime.utcnow()
            trace_context.status = status
            
            if trace_context.start_time:
                duration = (trace_context.end_time - trace_context.start_time).total_seconds() * 1000
                trace_context.duration_ms = duration
            
            if error_message:
                trace_context.logs.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": "error",
                    "message": error_message
                })
            
            logger.debug(f"Ended trace {trace_id} with status {status.value}")
            
        except Exception as e:
            logger.error(f"Error ending trace: {e}")
    
    def add_trace_log(self, trace_id: str, level: str, message: str, 
                     attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add a log entry to a trace."""
        try:
            if trace_id not in self.traces:
                logger.warning(f"Trace {trace_id} not found")
                return
            
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": level,
                "message": message,
                "attributes": attributes or {}
            }
            
            self.traces[trace_id].logs.append(log_entry)
            
        except Exception as e:
            logger.error(f"Error adding trace log: {e}")
    
    def add_trace_tag(self, trace_id: str, key: str, value: Any) -> None:
        """Add a tag to a trace."""
        try:
            if trace_id not in self.traces:
                logger.warning(f"Trace {trace_id} not found")
                return
            
            self.traces[trace_id].tags[key] = value
            
        except Exception as e:
            logger.error(f"Error adding trace tag: {e}")
    
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, parent_trace_id: Optional[str] = None,
                            tags: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations."""
        trace_context = None
        try:
            trace_context = self.start_trace(operation_name, parent_trace_id, tags)
            yield trace_context
        except Exception as e:
            if trace_context:
                self.add_trace_log(trace_context.trace_id, "error", str(e))
                self.end_trace(trace_context.trace_id, TraceStatus.ERROR, str(e))
            raise
        finally:
            if trace_context:
                self.end_trace(trace_context.trace_id)
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get trace information."""
        try:
            if trace_id not in self.traces:
                return None
            
            trace_context = self.traces[trace_id]
            return {
                "trace_id": trace_context.trace_id,
                "span_id": trace_context.span_id,
                "parent_span_id": trace_context.parent_span_id,
                "operation_name": trace_context.operation_name,
                "start_time": trace_context.start_time.isoformat() if trace_context.start_time else None,
                "end_time": trace_context.end_time.isoformat() if trace_context.end_time else None,
                "duration_ms": trace_context.duration_ms,
                "status": trace_context.status.value,
                "tags": trace_context.tags,
                "logs": trace_context.logs
            }
            
        except Exception as e:
            logger.error(f"Error getting trace: {e}")
            return None
    
    def get_traces_summary(self) -> Dict[str, Any]:
        """Get traces summary."""
        try:
            total_traces = len(self.traces)
            active_traces = len([t for t in self.traces.values() if t.end_time is None])
            completed_traces = total_traces - active_traces
            
            status_counts = {}
            for trace in self.traces.values():
                status = trace.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "total_traces": total_traces,
                "active_traces": active_traces,
                "completed_traces": completed_traces,
                "status_distribution": status_counts,
                "last_update": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting traces summary: {e}")
            return {}
    
    def get_recent_traces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent traces."""
        try:
            traces = list(self.traces.values())
            traces.sort(key=lambda t: t.start_time or datetime.min, reverse=True)
            
            return [
                {
                    "trace_id": trace.trace_id,
                    "operation_name": trace.operation_name,
                    "start_time": trace.start_time.isoformat() if trace.start_time else None,
                    "duration_ms": trace.duration_ms,
                    "status": trace.status.value,
                    "tags": trace.tags
                }
                for trace in traces[:limit]
            ]
            
        except Exception as e:
            logger.error(f"Error getting recent traces: {e}")
            return []
    
    def cleanup_old_traces(self, max_age_hours: int = 24) -> None:
        """Clean up old traces."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            to_remove = []
            for trace_id, trace_context in self.traces.items():
                if trace_context.end_time and trace_context.end_time < cutoff_time:
                    to_remove.append(trace_id)
            
            for trace_id in to_remove:
                del self.traces[trace_id]
            
            logger.info(f"Cleaned up {len(to_remove)} old traces")
            
        except Exception as e:
            logger.error(f"Error cleaning up traces: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get tracing service health status."""
        try:
            return {
                "is_initialized": self.is_initialized,
                "tracer_available": self.tracer is not None,
                "span_processors": len(self.span_processors),
                "active_traces": len([t for t in self.traces.values() if t.end_time is None]),
                "total_traces": len(self.traces),
                "last_update": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                "is_initialized": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global tracing service instance
tracing_service = DistributedTracingService()

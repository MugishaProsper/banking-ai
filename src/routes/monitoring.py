"""
Advanced monitoring API routes for metrics, alerting, and tracing.
"""
from fastapi import APIRouter, Request, HTTPException, status, Depends, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from src.services.metrics_service import metrics_service
from src.services.alerting_service import alerting_service, AlertSeverity, AlertStatus
from src.services.tracing_service import tracing_service, TraceStatus
from src.middleware.auth_middleware import check_permission, get_client_id
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/v1/monitoring", tags=["Monitoring"])


class AlertRuleRequest(BaseModel):
    """Request model for creating alert rules."""
    
    name: str = Field(..., description="Alert rule name")
    description: str = Field(..., description="Alert rule description")
    metric_name: str = Field(..., description="Metric name to monitor")
    condition: str = Field(..., description="Condition operator (>, <, >=, <=, ==, !=)")
    threshold: float = Field(..., description="Threshold value")
    severity: str = Field(..., description="Alert severity level")
    duration_seconds: int = Field(..., ge=1, description="Duration in seconds")
    notification_channels: List[str] = Field(..., description="Notification channels")
    
    @validator('condition')
    def validate_condition(cls, v):
        valid_conditions = ['>', '<', '>=', '<=', '==', '!=']
        if v not in valid_conditions:
            raise ValueError(f'Condition must be one of: {valid_conditions}')
        return v
    
    @validator('severity')
    def validate_severity(cls, v):
        valid_severities = [s.value for s in AlertSeverity]
        if v not in valid_severities:
            raise ValueError(f'Severity must be one of: {valid_severities}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "name": "High Error Rate",
                "description": "HTTP error rate is too high",
                "metric_name": "http_requests_total",
                "condition": ">",
                "threshold": 0.05,
                "severity": "high",
                "duration_seconds": 300,
                "notification_channels": ["email", "slack"]
            }
        }


class AlertActionRequest(BaseModel):
    """Request model for alert actions."""
    
    action: str = Field(..., description="Action to perform")
    user: str = Field(..., description="User performing the action")
    
    @validator('action')
    def validate_action(cls, v):
        valid_actions = ['acknowledge', 'resolve']
        if v not in valid_actions:
            raise ValueError(f'Action must be one of: {valid_actions}')
        return v


@router.get("/metrics")
async def get_metrics(req: Request):
    """
    Get Prometheus metrics export.
    
    Returns metrics in Prometheus format for scraping by monitoring systems.
    """
    try:
        # Check permissions
        if not check_permission(req, "monitoring_metrics"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for metrics access"
            )
        
        metrics_export = metrics_service.get_metrics_export()
        
        return Response(
            content=metrics_export,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get metrics"
        )


@router.get("/metrics/summary")
async def get_metrics_summary(req: Request):
    """
    Get metrics summary for monitoring dashboard.
    
    Returns high-level metrics summary for dashboard visualization.
    """
    try:
        # Check permissions
        if not check_permission(req, "monitoring_summary"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for metrics summary access"
            )
        
        summary = metrics_service.get_metrics_summary()
        
        return {
            "status": "success",
            "metrics_summary": summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get metrics summary"
        )


@router.get("/alerts")
async def get_active_alerts(req: Request):
    """
    Get active alerts.
    
    Returns currently active alerts with their details and status.
    """
    try:
        # Check permissions
        if not check_permission(req, "monitoring_alerts"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for alerts access"
            )
        
        alerts = alerting_service.get_active_alerts()
        
        return {
            "status": "success",
            "active_alerts": alerts,
            "count": len(alerts),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get active alerts"
        )


@router.get("/alerts/rules")
async def get_alert_rules(req: Request):
    """
    Get alert rules configuration.
    
    Returns all configured alert rules with their settings.
    """
    try:
        # Check permissions
        if not check_permission(req, "monitoring_rules"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for alert rules access"
            )
        
        rules = alerting_service.get_alert_rules()
        
        return {
            "status": "success",
            "alert_rules": rules,
            "count": len(rules),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting alert rules: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get alert rules"
        )


@router.post("/alerts/rules")
async def create_alert_rule(
    request: AlertRuleRequest,
    req: Request
):
    """
    Create a new alert rule.
    
    Creates a new alert rule with specified conditions and notification channels.
    """
    try:
        # Check permissions
        if not check_permission(req, "monitoring_admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for creating alert rules"
            )
        
        # Create alert rule
        from src.services.alerting_service import AlertRule, NotificationChannel
        
        rule = AlertRule(
            rule_id=f"custom_{int(time.time())}",
            name=request.name,
            description=request.description,
            metric_name=request.metric_name,
            condition=request.condition,
            threshold=request.threshold,
            severity=AlertSeverity(request.severity),
            duration_seconds=request.duration_seconds,
            notification_channels=[NotificationChannel(ch) for ch in request.notification_channels],
            enabled=True,
            created_at=datetime.utcnow()
        )
        
        alerting_service.alert_rules[rule.rule_id] = rule
        
        # Log the rule creation
        client_id = get_client_id(req)
        logger.info(
            "Alert rule created",
            rule_id=rule.rule_id,
            name=rule.name,
            metric_name=rule.metric_name,
            threshold=rule.threshold,
            client_id=client_id
        )
        
        return {
            "status": "success",
            "message": "Alert rule created successfully",
            "rule_id": rule.rule_id,
            "rule": {
                "rule_id": rule.rule_id,
                "name": rule.name,
                "description": rule.description,
                "metric_name": rule.metric_name,
                "condition": rule.condition,
                "threshold": rule.threshold,
                "severity": rule.severity.value,
                "duration_seconds": rule.duration_seconds,
                "notification_channels": [ch.value for ch in rule.notification_channels],
                "enabled": rule.enabled,
                "created_at": rule.created_at.isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating alert rule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create alert rule"
        )


@router.post("/alerts/{alert_id}/action")
async def perform_alert_action(
    alert_id: str,
    request: AlertActionRequest,
    req: Request
):
    """
    Perform an action on an alert.
    
    Allows acknowledging or resolving alerts.
    """
    try:
        # Check permissions
        if not check_permission(req, "monitoring_alerts"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for alert actions"
            )
        
        success = False
        
        if request.action == "acknowledge":
            success = await alerting_service.acknowledge_alert(alert_id, request.user)
        elif request.action == "resolve":
            success = await alerting_service.resolve_alert(alert_id, request.user)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert {alert_id} not found or action failed"
            )
        
        # Log the action
        client_id = get_client_id(req)
        logger.info(
            "Alert action performed",
            alert_id=alert_id,
            action=request.action,
            user=request.user,
            client_id=client_id
        )
        
        return {
            "status": "success",
            "message": f"Alert {alert_id} {request.action}d successfully",
            "alert_id": alert_id,
            "action": request.action,
            "user": request.user,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing alert action: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform alert action"
        )


@router.get("/alerts/statistics")
async def get_alert_statistics(req: Request):
    """
    Get alert statistics.
    
    Returns comprehensive alert statistics and metrics.
    """
    try:
        # Check permissions
        if not check_permission(req, "monitoring_stats"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for alert statistics access"
            )
        
        statistics = alerting_service.get_alert_statistics()
        
        return {
            "status": "success",
            "alert_statistics": statistics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting alert statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get alert statistics"
        )


@router.get("/traces")
async def get_traces(
    limit: int = 100,
    req: Request
):
    """
    Get recent traces.
    
    Returns recent trace information for debugging and analysis.
    """
    try:
        # Check permissions
        if not check_permission(req, "monitoring_traces"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for traces access"
            )
        
        traces = tracing_service.get_recent_traces(limit)
        
        return {
            "status": "success",
            "traces": traces,
            "count": len(traces),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting traces: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get traces"
        )


@router.get("/traces/{trace_id}")
async def get_trace_details(
    trace_id: str,
    req: Request
):
    """
    Get detailed trace information.
    
    Returns detailed information for a specific trace.
    """
    try:
        # Check permissions
        if not check_permission(req, "monitoring_traces"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for trace details access"
            )
        
        trace_info = tracing_service.get_trace(trace_id)
        
        if not trace_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trace {trace_id} not found"
            )
        
        return {
            "status": "success",
            "trace": trace_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trace details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get trace details"
        )


@router.get("/traces/summary")
async def get_traces_summary(req: Request):
    """
    Get traces summary.
    
    Returns summary statistics for traces.
    """
    try:
        # Check permissions
        if not check_permission(req, "monitoring_stats"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for traces summary access"
            )
        
        summary = tracing_service.get_traces_summary()
        
        return {
            "status": "success",
            "traces_summary": summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting traces summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get traces summary"
        )


@router.get("/health")
async def get_monitoring_health(req: Request):
    """
    Get monitoring services health status.
    
    Returns health status of all monitoring services.
    """
    try:
        # Check permissions
        if not check_permission(req, "monitoring_health"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for monitoring health access"
            )
        
        # Get health status from all monitoring services
        metrics_health = metrics_service.get_health_status()
        tracing_health = tracing_service.get_health_status()
        
        # Overall health status
        overall_healthy = (
            metrics_health.get("status") == "healthy" and
            tracing_health.get("is_initialized", False) and
            alerting_service.is_initialized
        )
        
        health_status = {
            "overall_status": "healthy" if overall_healthy else "unhealthy",
            "services": {
                "metrics": {
                    "status": metrics_health.get("status", "unknown"),
                    "uptime_seconds": metrics_health.get("uptime_seconds", 0),
                    "memory_usage_bytes": metrics_health.get("memory_usage_bytes", 0),
                    "cpu_usage_percent": metrics_health.get("cpu_usage_percent", 0)
                },
                "alerting": {
                    "status": "healthy" if alerting_service.is_initialized else "unhealthy",
                    "active_alerts": len(alerting_service.active_alerts),
                    "rules_count": len(alerting_service.alert_rules)
                },
                "tracing": {
                    "status": "healthy" if tracing_health.get("is_initialized", False) else "unhealthy",
                    "active_traces": tracing_health.get("active_traces", 0),
                    "total_traces": tracing_health.get("total_traces", 0)
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return {
            "status": "success",
            "monitoring_health": health_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting monitoring health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get monitoring health"
        )


@router.post("/traces/cleanup")
async def cleanup_traces(
    max_age_hours: int = 24,
    req: Request
):
    """
    Clean up old traces.
    
    Removes traces older than specified age to manage storage.
    """
    try:
        # Check permissions
        if not check_permission(req, "monitoring_admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for trace cleanup"
            )
        
        # Clean up old traces
        tracing_service.cleanup_old_traces(max_age_hours)
        
        # Log the cleanup
        client_id = get_client_id(req)
        logger.info(
            "Trace cleanup performed",
            max_age_hours=max_age_hours,
            client_id=client_id
        )
        
        return {
            "status": "success",
            "message": f"Traces older than {max_age_hours} hours cleaned up",
            "max_age_hours": max_age_hours,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up traces: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clean up traces"
        )

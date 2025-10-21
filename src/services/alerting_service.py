"""
Advanced alerting system with configurable thresholds and notification channels.
"""
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json
import smtplib
import requests
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(str, Enum):
    """Notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"


@dataclass
class AlertRule:
    """Alert rule definition."""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., ">", "<", ">=", "<=", "==", "!="
    threshold: float
    severity: AlertSeverity
    duration_seconds: int  # How long condition must be true
    notification_channels: List[NotificationChannel]
    enabled: bool = True
    created_at: datetime = None
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class Alert:
    """Active alert instance."""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    metric_value: float
    threshold: float
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None


class AlertingService:
    """Advanced alerting and notification service."""
    
    def __init__(self):
        self.is_initialized = False
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Notification configuration
        self.notification_config = {
            NotificationChannel.EMAIL: {
                "smtp_server": settings.email_smtp_server,
                "smtp_port": settings.email_smtp_port,
                "username": settings.email_username,
                "password": settings.email_password,
                "from_email": settings.email_from,
                "to_emails": settings.email_to_emails.split(",") if settings.email_to_emails else []
            },
            NotificationChannel.SLACK: {
                "webhook_url": settings.slack_webhook_url,
                "channel": settings.slack_channel,
                "username": settings.slack_username
            },
            NotificationChannel.WEBHOOK: {
                "url": settings.webhook_url,
                "headers": json.loads(settings.webhook_headers) if settings.webhook_headers else {}
            },
            NotificationChannel.PAGERDUTY: {
                "integration_key": settings.pagerduty_integration_key,
                "api_url": "https://events.pagerduty.com/v2/enqueue"
            }
        }
        
        # Default alert rules
        self._init_default_rules()
        
    def _init_default_rules(self) -> None:
        """Initialize default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                description="HTTP error rate is too high",
                metric_name="http_requests_total",
                condition=">",
                threshold=0.05,  # 5% error rate
                severity=AlertSeverity.HIGH,
                duration_seconds=300,  # 5 minutes
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                created_at=datetime.utcnow()
            ),
            AlertRule(
                rule_id="high_response_time",
                name="High Response Time",
                description="Average response time is too high",
                metric_name="http_request_duration_seconds",
                condition=">",
                threshold=2.0,  # 2 seconds
                severity=AlertSeverity.MEDIUM,
                duration_seconds=600,  # 10 minutes
                notification_channels=[NotificationChannel.EMAIL],
                created_at=datetime.utcnow()
            ),
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                description="Memory usage is too high",
                metric_name="service_memory_usage_bytes",
                condition=">",
                threshold=1024 * 1024 * 1024 * 2,  # 2GB
                severity=AlertSeverity.HIGH,
                duration_seconds=300,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.PAGERDUTY],
                created_at=datetime.utcnow()
            ),
            AlertRule(
                rule_id="high_cpu_usage",
                name="High CPU Usage",
                description="CPU usage is too high",
                metric_name="service_cpu_usage_percent",
                condition=">",
                threshold=80.0,  # 80%
                severity=AlertSeverity.MEDIUM,
                duration_seconds=600,
                notification_channels=[NotificationChannel.EMAIL],
                created_at=datetime.utcnow()
            ),
            AlertRule(
                rule_id="high_fraud_rate",
                name="High Fraud Detection Rate",
                description="Fraud detection rate is unusually high",
                metric_name="fraud_detections_total",
                condition=">",
                threshold=100,  # 100 fraud detections per hour
                severity=AlertSeverity.CRITICAL,
                duration_seconds=300,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.PAGERDUTY],
                created_at=datetime.utcnow()
            ),
            AlertRule(
                rule_id="model_accuracy_drop",
                name="Model Accuracy Drop",
                description="Model accuracy has dropped significantly",
                metric_name="model_accuracy",
                condition="<",
                threshold=0.8,  # 80% accuracy
                severity=AlertSeverity.HIGH,
                duration_seconds=1800,  # 30 minutes
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                created_at=datetime.utcnow()
            ),
            AlertRule(
                rule_id="database_connection_issues",
                name="Database Connection Issues",
                description="Database connection failures",
                metric_name="db_operations_total",
                condition=">",
                threshold=10,  # 10 failed operations per minute
                severity=AlertSeverity.CRITICAL,
                duration_seconds=60,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.PAGERDUTY],
                created_at=datetime.utcnow()
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    async def initialize(self) -> None:
        """Initialize the alerting service."""
        try:
            # Start background alert monitoring
            asyncio.create_task(self._monitor_alerts())
            
            # Start alert cleanup task
            asyncio.create_task(self._cleanup_resolved_alerts())
            
            self.is_initialized = True
            logger.info("Alerting Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize alerting service: {e}")
            raise
    
    async def _monitor_alerts(self) -> None:
        """Monitor metrics and trigger alerts."""
        while True:
            try:
                await self._check_alert_conditions()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _check_alert_conditions(self) -> None:
        """Check all alert rule conditions."""
        try:
            from src.services.metrics_service import metrics_service
            
            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                # Get current metric value
                metric_value = await self._get_metric_value(rule.metric_name)
                if metric_value is None:
                    continue
                
                # Check if condition is met
                condition_met = self._evaluate_condition(metric_value, rule.condition, rule.threshold)
                
                if condition_met:
                    # Check if alert should be triggered
                    if await self._should_trigger_alert(rule, metric_value):
                        await self._trigger_alert(rule, metric_value)
                else:
                    # Reset trigger count if condition is not met
                    rule.trigger_count = 0
                    
        except Exception as e:
            logger.error(f"Error checking alert conditions: {e}")
    
    async def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric."""
        try:
            from src.services.metrics_service import metrics_service
            
            # This would query the actual metrics service
            # For now, simulate based on metric name
            if metric_name == "http_requests_total":
                return 1000.0
            elif metric_name == "http_request_duration_seconds":
                return 1.5
            elif metric_name == "service_memory_usage_bytes":
                return 1024 * 1024 * 1024  # 1GB
            elif metric_name == "service_cpu_usage_percent":
                return 45.0
            elif metric_name == "fraud_detections_total":
                return 50.0
            elif metric_name == "model_accuracy":
                return 0.85
            elif metric_name == "db_operations_total":
                return 5.0
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting metric value for {metric_name}: {e}")
            return None
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        try:
            if condition == ">":
                return value > threshold
            elif condition == "<":
                return value < threshold
            elif condition == ">=":
                return value >= threshold
            elif condition == "<=":
                return value <= threshold
            elif condition == "==":
                return value == threshold
            elif condition == "!=":
                return value != threshold
            else:
                logger.warning(f"Unknown condition operator: {condition}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
    
    async def _should_trigger_alert(self, rule: AlertRule, metric_value: float) -> bool:
        """Check if alert should be triggered based on duration."""
        try:
            rule.trigger_count += 1
            
            # Check if condition has been met for required duration
            required_checks = rule.duration_seconds // 30  # 30-second intervals
            return rule.trigger_count >= required_checks
            
        except Exception as e:
            logger.error(f"Error checking if alert should trigger: {e}")
            return False
    
    async def _trigger_alert(self, rule: AlertRule, metric_value: float) -> None:
        """Trigger an alert."""
        try:
            alert_id = f"{rule.rule_id}_{int(time.time())}"
            
            # Check if similar alert is already active
            if any(alert.rule_id == rule.rule_id and alert.status == AlertStatus.ACTIVE 
                   for alert in self.active_alerts.values()):
                return
            
            # Create alert
            alert = Alert(
                alert_id=alert_id,
                rule_id=rule.rule_id,
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                message=f"{rule.name}: {rule.description} (Value: {metric_value}, Threshold: {rule.threshold})",
                metric_value=metric_value,
                threshold=rule.threshold,
                triggered_at=datetime.utcnow()
            )
            
            # Add to active alerts
            self.active_alerts[alert_id] = alert
            
            # Add to history
            self.alert_history.append(alert)
            
            # Update rule
            rule.last_triggered = datetime.utcnow()
            rule.trigger_count = 0
            
            # Send notifications
            await self._send_notifications(alert, rule)
            
            logger.warning(f"Alert triggered: {alert_id} - {alert.message}")
            
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
    
    async def _send_notifications(self, alert: Alert, rule: AlertRule) -> None:
        """Send notifications for an alert."""
        try:
            for channel in rule.notification_channels:
                try:
                    if channel == NotificationChannel.EMAIL:
                        await self._send_email_notification(alert, rule)
                    elif channel == NotificationChannel.SLACK:
                        await self._send_slack_notification(alert, rule)
                    elif channel == NotificationChannel.WEBHOOK:
                        await self._send_webhook_notification(alert, rule)
                    elif channel == NotificationChannel.PAGERDUTY:
                        await self._send_pagerduty_notification(alert, rule)
                        
                except Exception as e:
                    logger.error(f"Error sending {channel} notification: {e}")
                    
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
    
    async def _send_email_notification(self, alert: Alert, rule: AlertRule) -> None:
        """Send email notification."""
        try:
            config = self.notification_config[NotificationChannel.EMAIL]
            
            if not config["to_emails"]:
                return
            
            msg = MIMEMultipart()
            msg['From'] = config["from_email"]
            msg['To'] = ", ".join(config["to_emails"])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {rule.name}"
            
            body = f"""
            Alert: {rule.name}
            Severity: {alert.severity.value.upper()}
            Status: {alert.status.value}
            Message: {alert.message}
            Metric Value: {alert.metric_value}
            Threshold: {alert.threshold}
            Triggered At: {alert.triggered_at.isoformat()}
            
            Rule: {rule.description}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
            server.starttls()
            server.login(config["username"], config["password"])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    async def _send_slack_notification(self, alert: Alert, rule: AlertRule) -> None:
        """Send Slack notification."""
        try:
            config = self.notification_config[NotificationChannel.SLACK]
            
            if not config["webhook_url"]:
                return
            
            # Determine emoji based on severity
            emoji_map = {
                AlertSeverity.LOW: ":white_circle:",
                AlertSeverity.MEDIUM: ":yellow_circle:",
                AlertSeverity.HIGH: ":orange_circle:",
                AlertSeverity.CRITICAL: ":red_circle:"
            }
            
            emoji = emoji_map.get(alert.severity, ":white_circle:")
            
            payload = {
                "channel": config["channel"],
                "username": config["username"],
                "text": f"{emoji} *{rule.name}*",
                "attachments": [
                    {
                        "color": "danger" if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL] else "warning",
                        "fields": [
                            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                            {"title": "Status", "value": alert.status.value, "short": True},
                            {"title": "Message", "value": alert.message, "short": False},
                            {"title": "Metric Value", "value": str(alert.metric_value), "short": True},
                            {"title": "Threshold", "value": str(alert.threshold), "short": True},
                            {"title": "Triggered At", "value": alert.triggered_at.isoformat(), "short": False}
                        ]
                    }
                ]
            }
            
            response = requests.post(config["webhook_url"], json=payload)
            response.raise_for_status()
            
            logger.info(f"Slack notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    async def _send_webhook_notification(self, alert: Alert, rule: AlertRule) -> None:
        """Send webhook notification."""
        try:
            config = self.notification_config[NotificationChannel.WEBHOOK]
            
            if not config["url"]:
                return
            
            payload = {
                "alert_id": alert.alert_id,
                "rule_id": rule.rule_id,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "message": alert.message,
                "metric_value": alert.metric_value,
                "threshold": alert.threshold,
                "triggered_at": alert.triggered_at.isoformat(),
                "rule_name": rule.name,
                "rule_description": rule.description
            }
            
            response = requests.post(
                config["url"],
                json=payload,
                headers=config["headers"]
            )
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
    
    async def _send_pagerduty_notification(self, alert: Alert, rule: AlertRule) -> None:
        """Send PagerDuty notification."""
        try:
            config = self.notification_config[NotificationChannel.PAGERDUTY]
            
            if not config["integration_key"]:
                return
            
            # Only send for high/critical severity
            if alert.severity not in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                return
            
            payload = {
                "routing_key": config["integration_key"],
                "event_action": "trigger",
                "dedup_key": alert.alert_id,
                "payload": {
                    "summary": f"{rule.name}: {alert.message}",
                    "severity": alert.severity.value,
                    "source": "ai-banking-microservice",
                    "timestamp": alert.triggered_at.isoformat(),
                    "custom_details": {
                        "rule_id": rule.rule_id,
                        "metric_value": alert.metric_value,
                        "threshold": alert.threshold,
                        "description": rule.description
                    }
                }
            }
            
            response = requests.post(config["api_url"], json=payload)
            response.raise_for_status()
            
            logger.info(f"PagerDuty notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending PagerDuty notification: {e}")
    
    async def _cleanup_resolved_alerts(self) -> None:
        """Clean up old resolved alerts."""
        while True:
            try:
                # Remove alerts older than 7 days
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                
                # Clean up active alerts
                to_remove = []
                for alert_id, alert in self.active_alerts.items():
                    if alert.status == AlertStatus.RESOLVED and alert.resolved_at and alert.resolved_at < cutoff_time:
                        to_remove.append(alert_id)
                
                for alert_id in to_remove:
                    del self.active_alerts[alert_id]
                
                # Clean up history (keep last 1000 alerts)
                if len(self.alert_history) > 1000:
                    self.alert_history = self.alert_history[-1000:]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up alerts: {e}")
                await asyncio.sleep(3600)
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an alert."""
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            alert.resolved_by = resolved_by
            
            logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts."""
        return [
            {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "message": alert.message,
                "metric_value": alert.metric_value,
                "threshold": alert.threshold,
                "triggered_at": alert.triggered_at.isoformat(),
                "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                "acknowledged_by": alert.acknowledged_by
            }
            for alert in self.active_alerts.values()
        ]
    
    def get_alert_rules(self) -> List[Dict[str, Any]]:
        """Get alert rules."""
        return [
            {
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
                "created_at": rule.created_at.isoformat(),
                "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None,
                "trigger_count": rule.trigger_count
            }
            for rule in self.alert_rules.values()
        ]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        active_count = len(self.active_alerts)
        total_count = len(self.alert_history)
        
        severity_counts = {}
        for alert in self.active_alerts.values():
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "active_alerts": active_count,
            "total_alerts": total_count,
            "severity_distribution": severity_counts,
            "rules_count": len(self.alert_rules),
            "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
            "last_update": datetime.utcnow().isoformat()
        }


# Global alerting service instance
alerting_service = AlertingService()

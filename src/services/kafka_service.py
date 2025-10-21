"""
Kafka integration for event streaming and real-time updates.
"""
import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class KafkaEventProducer:
    """Kafka producer for publishing events."""
    
    def __init__(self):
        self.producer: Optional[AIOKafkaProducer] = None
        self.is_connected = False
        
    async def initialize(self) -> None:
        """Initialize Kafka producer."""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=settings.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retry_backoff_ms=100,
                request_timeout_ms=30000,
                max_block_ms=10000
            )
            
            await self.producer.start()
            self.is_connected = True
            logger.info("Kafka producer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            self.is_connected = False
            raise
    
    async def publish_fraud_score(self, fraud_data: Dict[str, Any]) -> None:
        """Publish fraud score event."""
        try:
            if not self.is_connected or not self.producer:
                logger.warning("Kafka producer not connected, skipping fraud score publish")
                return
            
            event = {
                "event_type": "fraud_score",
                "transaction_id": fraud_data.get("transaction_id"),
                "probability": fraud_data.get("probability"),
                "decision": fraud_data.get("decision"),
                "model_version": fraud_data.get("model_version"),
                "timestamp": datetime.utcnow().isoformat(),
                "data": fraud_data
            }
            
            await self.producer.send_and_wait(
                "fraud_score_events",
                value=event,
                key=fraud_data.get("transaction_id")
            )
            
            logger.debug(f"Published fraud score event for transaction {fraud_data.get('transaction_id')}")
            
        except Exception as e:
            logger.error(f"Error publishing fraud score event: {e}")
    
    async def publish_fraud_alert(self, alert_data: Dict[str, Any]) -> None:
        """Publish fraud alert event."""
        try:
            if not self.is_connected or not self.producer:
                logger.warning("Kafka producer not connected, skipping fraud alert publish")
                return
            
            event = {
                "event_type": "fraud_alert",
                "transaction_id": alert_data.get("transaction_id"),
                "alert_level": alert_data.get("alert_level", "high"),
                "reason": alert_data.get("reason"),
                "timestamp": datetime.utcnow().isoformat(),
                "data": alert_data
            }
            
            await self.producer.send_and_wait(
                "fraud_alerts",
                value=event,
                key=alert_data.get("transaction_id")
            )
            
            logger.info(f"Published fraud alert for transaction {alert_data.get('transaction_id')}")
            
        except Exception as e:
            logger.error(f"Error publishing fraud alert event: {e}")
    
    async def publish_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """Publish analyst feedback event."""
        try:
            if not self.is_connected or not self.producer:
                logger.warning("Kafka producer not connected, skipping feedback publish")
                return
            
            event = {
                "event_type": "fraud_feedback",
                "transaction_id": feedback_data.get("transaction_id"),
                "final_label": feedback_data.get("final_label"),
                "analyst_id": feedback_data.get("analyst_id"),
                "timestamp": datetime.utcnow().isoformat(),
                "data": feedback_data
            }
            
            await self.producer.send_and_wait(
                "fraud_feedback",
                value=event,
                key=feedback_data.get("transaction_id")
            )
            
            logger.info(f"Published feedback event for transaction {feedback_data.get('transaction_id')}")
            
        except Exception as e:
            logger.error(f"Error publishing feedback event: {e}")
    
    async def publish_model_metrics(self, metrics_data: Dict[str, Any]) -> None:
        """Publish model performance metrics."""
        try:
            if not self.is_connected or not self.producer:
                logger.warning("Kafka producer not connected, skipping metrics publish")
                return
            
            event = {
                "event_type": "model_metrics",
                "model_version": metrics_data.get("model_version"),
                "metrics": metrics_data.get("metrics", {}),
                "timestamp": datetime.utcnow().isoformat(),
                "data": metrics_data
            }
            
            await self.producer.send_and_wait(
                "model_metrics",
                value=event,
                key=metrics_data.get("model_version")
            )
            
            logger.debug(f"Published model metrics for version {metrics_data.get('model_version')}")
            
        except Exception as e:
            logger.error(f"Error publishing model metrics: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown Kafka producer."""
        try:
            if self.producer:
                await self.producer.stop()
                self.is_connected = False
                logger.info("Kafka producer shutdown successfully")
        except Exception as e:
            logger.error(f"Error shutting down Kafka producer: {e}")


class KafkaEventConsumer:
    """Kafka consumer for processing events."""
    
    def __init__(self):
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.is_connected = False
        self.running = False
        
    async def initialize(self) -> None:
        """Initialize Kafka consumer."""
        try:
            self.consumer = AIOKafkaConsumer(
                "txn_stream",
                "fraud_feedback",
                bootstrap_servers=settings.kafka_bootstrap_servers,
                group_id=settings.kafka_group_id,
                auto_offset_reset=settings.kafka_auto_offset_reset,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                enable_auto_commit=True,
                auto_commit_interval_ms=1000
            )
            
            await self.consumer.start()
            self.is_connected = True
            logger.info("Kafka consumer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            self.is_connected = False
            raise
    
    async def start_consuming(self) -> None:
        """Start consuming events."""
        if not self.is_connected or not self.consumer:
            logger.error("Kafka consumer not connected")
            return
        
        self.running = True
        logger.info("Started consuming Kafka events")
        
        try:
            async for msg in self.consumer:
                if not self.running:
                    break
                
                try:
                    await self._process_message(msg)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except Exception as e:
            logger.error(f"Error in Kafka consumer loop: {e}")
        finally:
            logger.info("Kafka consumer stopped")
    
    async def _process_message(self, msg) -> None:
        """Process incoming Kafka message."""
        try:
            topic = msg.topic
            data = msg.value
            
            if topic == "txn_stream":
                await self._process_transaction_event(data)
            elif topic == "fraud_feedback":
                await self._process_feedback_event(data)
            else:
                logger.warning(f"Unknown topic: {topic}")
                
        except Exception as e:
            logger.error(f"Error processing message from topic {msg.topic}: {e}")
    
    async def _process_transaction_event(self, data: Dict[str, Any]) -> None:
        """Process transaction stream event."""
        try:
            transaction_id = data.get("transaction_id")
            logger.debug(f"Processing transaction event: {transaction_id}")
            
            # Here you would trigger fraud scoring or update GNN graph
            # For now, just log the event
            logger.info(f"Received transaction event: {transaction_id}")
            
        except Exception as e:
            logger.error(f"Error processing transaction event: {e}")
    
    async def _process_feedback_event(self, data: Dict[str, Any]) -> None:
        """Process feedback event."""
        try:
            transaction_id = data.get("transaction_id")
            final_label = data.get("final_label")
            analyst_id = data.get("analyst_id")
            
            logger.info(f"Received feedback for transaction {transaction_id}: label={final_label}, analyst={analyst_id}")
            
            # Here you would trigger model retraining or update training data
            
        except Exception as e:
            logger.error(f"Error processing feedback event: {e}")
    
    async def stop_consuming(self) -> None:
        """Stop consuming events."""
        self.running = False
        logger.info("Stopping Kafka consumer")
    
    async def shutdown(self) -> None:
        """Shutdown Kafka consumer."""
        try:
            await self.stop_consuming()
            if self.consumer:
                await self.consumer.stop()
                self.is_connected = False
                logger.info("Kafka consumer shutdown successfully")
        except Exception as e:
            logger.error(f"Error shutting down Kafka consumer: {e}")


# Global Kafka instances
kafka_producer = KafkaEventProducer()
kafka_consumer = KafkaEventConsumer()

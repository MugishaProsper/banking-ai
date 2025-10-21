"""
Test script for fraud detection service.
"""
import asyncio
import json
from datetime import datetime
from src.services.fraud_detection import fraud_service


async def test_fraud_detection():
    """Test the fraud detection service."""
    
    print("Testing Fraud Detection Service...")
    
    # Initialize service
    await fraud_service.initialize()
    
    # Test transaction data
    test_transactions = [
        {
            "transaction_id": "txn_test_001",
            "timestamp": datetime.utcnow().isoformat(),
            "amount": 50.0,
            "currency": "USD",
            "sender_account": "acc_123456",
            "receiver_account": "acc_789012",
            "channel": "mobile_app",
            "transaction_type": "transfer",
            "description": "Normal transaction",
            "features": {
                "device_risk_score": 0.2,
                "location_risk_score": 0.1,
                "merchant_risk_score": 0.3
            }
        },
        {
            "transaction_id": "txn_test_002",
            "timestamp": datetime.utcnow().isoformat(),
            "amount": 5000.0,
            "currency": "USD",
            "sender_account": "acc_123456",
            "receiver_account": "acc_789012",
            "channel": "web",
            "transaction_type": "transfer",
            "description": "High amount transaction",
            "features": {
                "device_risk_score": 0.8,
                "location_risk_score": 0.9,
                "merchant_risk_score": 0.7
            }
        },
        {
            "transaction_id": "txn_test_003",
            "timestamp": datetime.utcnow().isoformat(),
            "amount": 25.0,
            "currency": "USD",
            "sender_account": "acc_123456",
            "receiver_account": "acc_789012",
            "channel": "mobile_app",
            "transaction_type": "payment",
            "description": "Late night transaction",
            "features": {
                "device_risk_score": 0.1,
                "location_risk_score": 0.2,
                "merchant_risk_score": 0.1
            }
        }
    ]
    
    # Test each transaction
    for i, transaction in enumerate(test_transactions, 1):
        print(f"\n--- Test Transaction {i} ---")
        print(f"Amount: ${transaction['amount']}")
        print(f"Channel: {transaction['channel']}")
        print(f"Features: {transaction['features']}")
        
        try:
            result = await fraud_service.score_transaction(transaction)
            
            print(f"Fraud Score: {result['probability']:.3f}")
            print(f"Decision: {result['decision']}")
            print(f"Model Version: {result['model_version']}")
            print(f"Latency: {result['latency_ms']}ms")
            print(f"Top Features:")
            for feature in result['explanation']['top_features'][:3]:
                print(f"  - {feature['feature']}: {feature['impact']:.3f}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    # Test model status
    print(f"\n--- Model Status ---")
    status = await fraud_service.get_model_status()
    print(json.dumps(status, indent=2, default=str))
    
    print("\nFraud Detection Service Test Complete!")


if __name__ == "__main__":
    asyncio.run(test_fraud_detection())

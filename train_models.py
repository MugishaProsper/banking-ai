"""
Simple model training script for fraud detection.
This creates basic models for testing the fraud detection service.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 10000) -> tuple:
    """Generate synthetic fraud detection data."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate features
    data = {
        'amount_log': np.random.normal(2.5, 1.0, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'transaction_frequency_24h': np.random.poisson(2, n_samples),
        'transaction_frequency_7d': np.random.poisson(10, n_samples),
        'avg_transaction_amount_30d': np.random.normal(100, 50, n_samples),
        'device_risk_score': np.random.uniform(0, 1, n_samples),
        'location_risk_score': np.random.uniform(0, 1, n_samples),
        'merchant_risk_score': np.random.uniform(0, 1, n_samples),
        'account_age_days': np.random.uniform(30, 3650, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate fraud labels based on feature combinations
    # Higher amounts, unusual hours, high risk scores = more likely fraud
    fraud_prob = (
        0.1 * (df['amount_log'] > 3.0).astype(int) +
        0.1 * ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)).astype(int) +
        0.2 * (df['device_risk_score'] > 0.7).astype(int) +
        0.2 * (df['location_risk_score'] > 0.7).astype(int) +
        0.2 * (df['merchant_risk_score'] > 0.7).astype(int) +
        0.1 * (df['transaction_frequency_24h'] > 5).astype(int) +
        0.1 * np.random.random(n_samples)
    )
    
    # Convert to binary labels
    fraud_labels = (fraud_prob > 0.5).astype(int)
    
    logger.info(f"Generated {n_samples} samples with {fraud_labels.sum()} fraud cases ({fraud_labels.mean():.2%})")
    
    return df, fraud_labels


def train_tabular_model(X: pd.DataFrame, y: np.ndarray) -> tuple:
    """Train tabular fraud detection model."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"Tabular model AUC: {auc_score:.3f}")
    logger.info("Classification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    
    return model, scaler


def train_anomaly_model(X: pd.DataFrame) -> IsolationForest:
    """Train anomaly detection model."""
    
    # Use only non-fraud cases for training (in practice)
    # For simplicity, we'll use all data but this is not ideal
    model = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_estimators=100
    )
    
    model.fit(X)
    
    logger.info("Anomaly model trained")
    
    return model


def main():
    """Main training function."""
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Generate synthetic data
    logger.info("Generating synthetic fraud detection data...")
    X, y = generate_synthetic_data(10000)
    
    # Train tabular model
    logger.info("Training tabular model...")
    tabular_model, scaler = train_tabular_model(X, y)
    
    # Train anomaly model
    logger.info("Training anomaly model...")
    anomaly_model = train_anomaly_model(X)
    
    # Save models
    logger.info("Saving models...")
    
    joblib.dump(tabular_model, models_dir / "fraud_tabular_model.pkl")
    joblib.dump(scaler, models_dir / "fraud_tabular_scaler.pkl")
    joblib.dump(anomaly_model, models_dir / "fraud_anomaly_model.pkl")
    
    # Save feature names
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, models_dir / "feature_names.pkl")
    
    logger.info("Models saved successfully!")
    logger.info(f"Models saved to: {models_dir.absolute()}")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': tabular_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Feature Importance:")
    logger.info(f"\n{feature_importance}")


if __name__ == "__main__":
    main()

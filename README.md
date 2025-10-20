# Fraud AI Microservice — SRS & System Description

## 1. Purpose

Provide a production-grade AI microservice that ingests transactions (real-time and batch), scores them for fraud risk, explains the scoring, accepts analyst feedback for retraining, and exposes health and operational metrics to the banking backend and compliance systems.

## 2. Scope

* Real-time scoring for individual transactions (sub-100ms goal for tabular path).
* Asynchronous ingestion and periodic graph/GNN scoring updates.
* Explainability outputs (top features + graph evidence).
* Feedback endpoint for analyst dispositions and confirmed fraud labels.
* Model training pipelines: offline retrain, incremental updates, scheduled retrains.
* Monitoring, model versioning, drift detection, audit logs.
* Secure, auditable, compliant with data retention and explainability policies.

## 3. Objectives & Key Requirements

* **Accuracy:** High recall for true fraud cases while maintaining acceptable operational precision (business-configurable).
* **Latency:** Real-time tabular scores: p99 < 150ms end-to-end (network + inference). Graph/GNN may be asynchronous with best-effort latency.
* **Explainability:** Per-inference top-5 feature impacts (SHAP or equivalent) plus graph traces (up to depth 3).
* **Resilience:** Graceful degradation — fallback to simpler models if model service or feature store unavailable.
* **Auditability:** Immutable logs for every inference and retrain (who/when/why).
* **Security & Compliance:** PII handling, encryption at rest/in transit, GDPR right-to-erasure support (with careful audit handling).

---

## 4. Actors

* **Client systems** (API Gateway / Transaction Service)
* **Analyst/Case Manager UI**
* **Model Training Pipeline / Data Engineers**
* **Monitoring & DevOps**
* **Compliance & Audit teams**

---

## 5. High-Level Architecture (textual)

```
                +----------------+
                |  API Gateway   |
                +--------+-------+
                         |
               +---------v---------+
               | Fraud AI Service  | <--- REST/gRPC
               |  - Scorer (Fast)  |
               |  - Explainability |
               |  - Feedback       |
               +----+----+----+----+
                    |    |    |
    Kafka topic <---+    |    +---> Kafka topic (alerts, logs)
               Streams|    |
                    |    |
           +--------v----v--------+
           | Feature Store (Feast) |
           +--------+----+--------+
                    |
           +--------v---------+
           | Model Training   |
           |  - Autoencoder   |
           |  - GNN           |
           |  - Tabular model |
           +------------------+
```

Notes:

* Fast path: feature-store lookup + LightGBM/MLP -> score.
* Network path: asynchronous GNN update enriches entity risk profiles and updates cache.
* All requests and responses are logged to append-only audit storage (S3 or write-once DB).

---

## 6. Functional Requirements

### 6.1 Endpoints (REST + gRPC)

All endpoints require OAuth2 JWT with scopes. Prefer gRPC for low-latency internal calls; provide REST wrapper for compatibility.

1. **POST /v1/score**
   Purpose: Real-time scoring for a transaction.
   Input (JSON):

   ```json
   {
     "_id": mongooseObjectId,
     "timestamp": "2025-10-20T13:45:30Z",
     "amount": 120.50,
     "currency": "USD",
     "sender_account": mongooseObjectId,
     "receiver_account": mongooseObjectId,
     "channel": "mobile_app",
     "features": { "optional_precomputed": {} }
   }
   ```

   Output:

   ```json
   {
     "transaction_id": mongooseObjectId,
     "model_version": "tabular_v2025-10-18-14",
     "probability": 0.87,
     "decision": "HOLD",         // one of: ALLOW, HOLD, BLOCK
     "explanation": {
       "top_features": [
         {"feature":"new_device_flag","impact":0.45},
         {"feature":"amount_log","impact":0.32}
       ],
       "graph_evidence": [
         {"path":["ACC123","DEV789","ACC999"], "suspiciousness": 0.8}
       ]
     },
     "latency_ms": 38
   }
   ```

   Behavior:

   * Score computed using online features + cached last-known graph risk.
   * Decision derived by policy engine (configurable threshold).
   * If GNN info not available, the response includes `graph_evidence: null` and `provisional: true`.

2. **POST /v1/feedback**
   Purpose: Accept human analyst feedback or confirmed outcomes. Used for label correction and model retraining.
   Input:

   ```json
   {
     "_id": mongooseObjectId,
     "final_label": 1,  // 1 fraud, 0 not fraud
     "analyst_id":"ANL42",
     "notes":"confirmed chargeback"
   }
   ```

   Behavior:

   * Persist to feedback store and produce Kafka event `fraud_feedback`.

3. **GET /v1/health**
   Purpose: Service health, model load, feature-store connectivity, last retrain timestamp. Returns 2xx only if core models are loaded.

4. **GET /v1/model/{model_name}/status**
   Info on versions, A/B splits, active model, training metrics.

5. **POST /v1/retrain** (protected, admin-only)
   Trigger scheduled or on-demand retrain; returns job id.

6. **GET /v1/explain/{transaction_id}**
   Return full explanation and graph trace for a previously scored transaction (for audit).

### 6.2 Kafka Topics

* `txn_stream` — raw transaction feed (from backend).
* `fraud_score_events` — scored events + metadata.
* `fraud_alerts` — alerts to case management.
* `fraud_feedback` — analyst labels and dispositions.
* `model_metrics` — periodic model performance reports.

---

## 7. Data Contracts & Schemas

### 7.1 Feature Schema (example)

* `txn.amount_log` float
* `txn.hour_of_day` int
* `cust.last_24h_txn_count` int
* `cust.avg_txn_amount_30d` float
* `device.is_new` bool
* `graph.node_degree_cust` int
* `graph.neighbor_risk_mean` float

Store schema in Git (JSON Schema) and version it. Feature mismatch → fail-fast with schema error.

### 7.2 Audit Log Record

* `event_id`, `timestamp`, `transaction_id`, `input_payload`, `model_version`, `score`, `decision`, `explanation`, `requester_service`, `request_id`

Write to write-once storage (S3 + object versioning) and index minimal finders in elasticsearch for retrieval.

---

## 8. Model Design & Training

### 8.1 Models in the stack

1. **Tabular model (production fast scorer)**

   * LightGBM / CatBoost / small MLP as fallback.
   * Input: online features + cached graph risk features + anomaly score.
   * Output: probability [0,1].
   * Training: supervised weighted objective; use time-based CV and class-weighting. Save calibration.

2. **Behavioral Anomaly Detector**

   * LSTM/Transformer or Dense Autoencoder over recent sequence windows per entity.
   * Output: reconstruction error -> anomaly score. Trained on presumed-normal windows or using label filtering.

3. **Graph Model (GNN)**

   * GraphSAGE / GAT trained to produce node risk embeddings and supervised fraud label prediction.
   * Trained periodically offline on transaction graph snapshots; produce per-entity risk and subgraph suspicious paths.

4. **Ensemble & Policy**

   * Final decision: deterministic function combining tabular_prob, anomaly_score, graph_risk, and business rules. Policy stored in a rules engine (configurable via UI/feature flag).

### 8.2 Training Pipeline

* **Ingestion**: pull labeled transactional history + feature view snapshots + feedback labels.
* **Feature engineering**: batch compute sliding-window aggregates, encode categorical features, compute graph features.
* **Split**: time-based training/validation/test splits to avoid leakage.
* **Model training**: run experiments tracked in MLflow (hyperparams, metrics, artifacts).
* **Evaluation**: AUC-PR, Precision@K, Recall@K, confusion matrix at chosen thresholds, stability checks.
* **Explainability**: compute global SHAP and local distributions; store exemplar cases for audit.
* **Deployment**: package model as Docker image, push to registry, deploy via CI/CD with canary rollout.

### 8.3 Retraining frequency

* **Tabular:** weekly or triggered by drift detection.
* **Autoencoder:** weekly or continuous incremental updates.
* **GNN:** nightly batch recompute or frequency depending on graph size (e.g., nightly/4-hr).
* **Thresholds / policy**: reviewed monthly with compliance.

---

## 9. Explainability & Evidence

* **Tabular:** SHAP-based top-5 feature impacts returned with each score. Keep SHAP calculations performant (TreeExplainer cached; for deep nets use approximations).
* **Graph:** return up to 3-hop suspicious paths, with node types and edge types and aggregated suspiciousness score.
* **Human-friendly summary:** a short plain-English sentence describing the main reason for the score (generated from template + feature impacts).
* **Retention:** keep full explanation records for compliance for configured retention period (e.g., 7 years or per regulation).

---

## 10. Non-Functional Requirements

* **Latency:** p99 inference time < 150ms for tabular path; GNN best-effort <= several seconds (async).
* **Throughput:** scale to 10k scores/sec with autoscaling.
* **Availability:** 99.99% with multi-AZ deployment and readiness probes.
* **Scalability:** horizontal scaling of prediction replicas; feature store caching via Redis.
* **Security:** TLS 1.3, mTLS between services, JWT auth, RBAC.
* **Data protection:** PII tokenized/hashes. Use envelope encryption for stored artifacts.
* **Logging:** structured JSON logs, correlation IDs across request flow.

---

## 11. Operational Requirements

### 11.1 Monitoring & Observability

* Metrics: latency, throughput, p99/p95, inputs null rates, model score distribution, PSI per feature, daily precision/recall, alerts per time window.
* Tracing: distributed tracing (OpenTelemetry) from API Gateway → Fraud AI → backend.
* Dashboards: Grafana dashboards for model health, Prometheus for metrics.
* Alerts: PagerDuty alerts for model degradation, feature-store unavailability, or anomalous score distribution.

### 11.2 CI/CD

* Infrastructure-as-code (Terraform/Helm).
* Build images via pipeline, run unit + integration tests, deploy to canary namespace, run smoke tests with synthetic traffic, promote to prod on success.
* Model artifacts versioned in MLflow and container labels.

### 11.3 Rollout Strategy

* Canary 5% traffic → monitor key metrics for 1–4 hours → ramp to 25% → 100% if OK.
* Automatic rollback if precision@K drops below threshold or latency spikes.

---

## 12. Security & Compliance

* **Authentication:** OAuth2/JWT; internal services use mTLS.
* **Authorization:** RBAC for endpoints. Admin-only for retrain/redeploy.
* **PII handling:** Hash and salt identifiers (device_id, ip) before storing. Keep raw PII in a separate, highly guarded vault only when needed.
* **Retention:** Configurable retention policy; ensure `explain` logs persist for regulatory windows.
* **Audit:** Immutable records for any model changes, retraining runs, or policy changes with operator id and timestamp.
* **Regulatory:** Provide exportable reports for AI decisions (model version, top features, rationale) for auditors.

---

## 13. Testing Plan

### 13.1 Unit Tests

* Feature parsing, schema validation, small-model predictions.

### 13.2 Integration Tests

* End-to-end flow: feature-store read → scoring → Kafka publish → audit log write.

### 13.3 Performance Tests

* Load testing to validate p99 latency at target throughput.

### 13.4 Security Tests

* Pen tests, dependency scanning, container image scanning.

### 13.5 Model Tests

* Data validation (missing values, distribution changes), backtest performance on holdout sets.
* Shadow mode deployment for new models — score in parallel and compare.

---

## 14. Error Handling & Fallbacks

* If feature store is down: use cached features (Redis) and return `provisional: true`. Log incident.
* If the main model is unavailable: use a simpler cached model (last-known stable) or rule-based fallback.
* If inference fails: return 5xx and the backend should treat as `provisional_hold` depending on policy.

---

## 15. API Rate Limits & Quotas

* Default: 200 requests/sec per client service, configurable via gateway.
* Burst handling: token bucket; excess requests are queued or throttled.

---

## 16. Maintenance & Ownership

* **SRE Team:** Responsible for infra, SLAs, and scaling.
* **ML Team:** Model training, feature engineering, drift monitoring.
* **Product/Compliance:** Define thresholds and review policies monthly.
* **On-call Rota:** 24/7 for critical incidents.

---

## 17. Metrics & KPIs (to track)

* Precision@1000 (daily)
* Recall@1000 (daily)
* False positives per 10k transactions
* Time-to-detect (mean)
* p95/p99 latency
* Model drift alerts per month
* Retrain success rate

---

## 18. Backward Compatibility & Versioning

* Model versions immutable. API versions must be maintained with deprecation notices.
* Feature schema changes must support backward compatibility or go through migration path.

---

## 19. Sample Deployment Diagram (ASCII)

```
   Clients (API Gateway)
        |
  +-----v------+
  | Fraud API  | <--- REST / gRPC
  +-----+------+
        |
  +-----v---------+          +--------------------+
  | Scoring Cache |<--redis--| Feature Store (Feast)|
  +-----+---------+          +--------------------+
        |
  +-----v---------+         +----------------+
  | Fast Scorer   |         | GNN Batch Job  |
  | (LightGBM)    |         | (Spark + DGL)  |
  +-----+---------+         +----------------+
        |
  +-----v---------+
  | Explainability|
  | (SHAP server) |
  +-----+---------+
        |
  +-----v---------+
  | Kafka Topics  |
  +-----+---------+
```

---
## 21. Deliverables (what I would give you next if you want it)

* Open-source starter repo skeleton: API server (FastAPI/gRPC), model serving (FastAPI + LightGBM), SHAP explainer, Kafka producers/consumers, feature-store connector sample, Dockerfile, Kubernetes manifests, GitHub Actions pipeline.
* Example synthetic dataset generator for testing.
* Terraform + Helm charts for infra.
* Detailed runbook and SLOs for on-call.
<div align="center">

<img src="https://raw.githubusercontent.com/devopsier/castor/main/docs/assets/castor-logo.svg" alt="Castor Logo" width="120" height="120" />

# Castor

### Predictive Pod & Cluster Steering for Kubernetes

**ML-powered traffic spike forecasting. Proactive autoscaling. Zero surprise failures.**

[![CI](https://github.com/devopsier/castor/actions/workflows/ci.yaml/badge.svg)](https://github.com/devopsier/castor/actions/workflows/ci.yaml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Made by DevOpsier](https://img.shields.io/badge/made%20by-DevOpsier-0070f3?style=flat)](https://github.com/devopsier)

[Overview](#-overview) · [Features](#-key-features) · [Architecture](#-architecture) · [Quickstart](#-quickstart) · [Configuration](#-configuration) · [API Reference](#-api-reference) · [Roadmap](#-roadmap)

</div>

---

> **Made by [DevOpsier](https://github.com/devopsier)** — open-source cloud-native tooling for Kubernetes operators.

## 🧠 Overview

**Castor** is an open-source, cloud-native intelligence layer for Kubernetes autoscaling. Instead of reacting to traffic spikes *after* they occur, Castor **predicts** them **15–30 minutes in advance** using historical Prometheus metrics and an XGBoost time-series forecasting model.

The result: your Horizontal Pod Autoscaler (HPA), [KEDA](https://keda.sh) scaler, or custom Kubernetes operator gets a pre-computed prediction it can act on — scaling pods *before* users notice any degradation.

```
┌─────────────────────┐      pull metrics      ┌──────────────────┐
│   Prometheus / VMs  │ ──────────────────────► │  Castor Ingestor │
└─────────────────────┘                         └────────┬─────────┘
                                                         │ DataFrame
                                                         ▼
                                                ┌──────────────────┐
                                                │ XGBoost Predictor│
                                                │  (lag features,  │
                                                │ asymmetric loss) │
                                                └────────┬─────────┘
                                                         │ PredictionResult
                              ┌──────────────────────────┼───────────────────────────┐
                              │                          │                           │
                              ▼                          ▼                           ▼
                    ┌──────────────────┐      ┌──────────────────┐       ┌─────────────────────┐
                    │  REST API        │      │ Webhook Dispatcher│       │   gRPC (Roadmap)    │
                    │  /v1/predict     │      │  (KEDA, Slack,   │       │   ScaleTrigger      │
                    │  /v1/webhooks    │      │   PagerDuty …)   │       │                     │
                    └──────────────────┘      └──────────────────┘       └─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  KEDA / HPA /    │
                    │  Custom Operator │
                    └──────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Proactive Forecasting** | XGBoost regressor trained on lag + rolling features forecasts load 15–30 min ahead |
| **Asymmetric Loss** | Custom objective penalises under-prediction 2.5× more than over-prediction — bias toward safety |
| **Prometheus Native** | Pulls CPU, Memory, and HTTP RPS via PromQL range queries out of the box |
| **REST API** | FastAPI endpoints for polling predictions (`/v1/predict`) and managing webhooks |
| **Event-Driven Webhooks** | Async HMAC-signed POST delivery with exponential back-off to any HTTP receiver |
| **Synthetic Dev Mode** | Realistic time-series simulation — full pipeline runs with zero external dependencies |
| **uv-Native** | Blazing-fast dependency management and project tooling with [uv](https://docs.astral.sh/uv/) |
| **Type-Safe** | Strict `mypy` typing across the entire codebase; Pydantic v2 for all request/response schemas |
| **Observable** | Structured JSON logging via `structlog`; health probe endpoints for Kubernetes |
| **Container-Ready** | Minimal multi-stage Docker image; runs as non-root user |

---

## 🏗️ Architecture

```
castor/
├── .github/
│   └── workflows/
│       └── ci.yaml              # Lint → Type Check → Test → Docker Build
├── src/castor/
│   ├── main.py                  # FastAPI factory + lifespan + background scheduler
│   ├── data/
│   │   └── ingestor.py          # PrometheusIngestor (httpx + DataFrame output)
│   ├── models/
│   │   ├── train.py             # SpikePredictorXGB + feature engineering + asymmetric loss
│   │   └── predict.py           # SpikePredictor facade + PredictionResult dataclass
│   └── api/
│       ├── routes.py            # /v1/predict, /v1/webhooks REST endpoints
│       └── webhooks.py          # WebhookDispatcher (tenacity retries, HMAC signing)
├── tests/
│   └── test_smoke.py            # Data, model, and API smoke tests
├── config.toml                  # All infrastructure settings (parsed by tomllib)
├── pyproject.toml               # uv project manifest + tooling config
└── Dockerfile                   # Multi-stage build: builder (uv) → runtime (slim)
```

### Data Flow

```
Prometheus ──(PromQL range)──► PrometheusIngestor.fetch_metric()
                                           │
                               pandas.DataFrame[timestamp, value]
                                           │
                               build_features()
                               ├── lag_5m, lag_15m, lag_30m, lag_60m, lag_120m
                               ├── roll_mean_10m, roll_std_30m, roll_max_60m …
                               └── hour_sin, hour_cos, day_of_week
                                           │
                               SpikePredictorXGB.fit() ──► .ubj artifact
                                           │
                               SpikePredictor.predict()
                                           │
                               PredictionResult { predicted_value, confidence, spike_imminent }
                                           │
                          ┌────────────────┴─────────────────┐
                          │                                  │
                     REST response                  WebhookDispatcher.fire_event()
                  (polling by KEDA)                 └──► POST to KEDA / Slack / PagerDuty
```

---

## 🚀 Quickstart

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 1. Clone & Install

```bash
git clone https://github.com/devopsier/castor.git
cd castor

# Install all dependencies into an isolated .venv — takes ~15 seconds
uv sync
```

### 2. Configure

```bash
# Review defaults and update Prometheus URL and webhook targets
vim config.toml
```

Key settings to configure for your environment:

```toml
[prometheus]
url = "http://your-prometheus:9090"   # ← Update this

[[webhooks.targets]]
name = "keda-http-scaler"
url  = "http://keda-service/scale-trigger"  # ← Update this
```

### 3. Run the Development Server

```bash
# Castor will auto-use synthetic data if Prometheus is unreachable
uv run castor
```

The API is now live at **http://localhost:8080**. Visit:
- **http://localhost:8080/docs** — Interactive Swagger UI
- **http://localhost:8080/redoc** — ReDoc API documentation
- **http://localhost:8080/healthz** — Health probe

### 4. Trigger Your First Prediction

```bash
curl -s -X POST http://localhost:8080/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"metric": "http_rps"}' | python -m json.tool
```

Expected response:
```json
{
  "metric_kind": "http_rps",
  "current_value": 347.8,
  "predicted_value": 812.3,
  "horizon_minutes": 30,
  "confidence": 0.8742,
  "spike_imminent": true,
  "threshold_used": 521.7,
  "evaluated_at": "2026-06-16T20:00:00+00:00"
}
```

### 5. Run Tests

```bash
uv run pytest -v
```

### 6. Run with Docker

```bash
docker build -t castor:dev .
docker run -p 8080:8080 \
  -v $(pwd)/config.toml:/app/config.toml:ro \
  castor:dev
```

---

## ⚙️ Configuration

All configuration is managed in `config.toml` and parsed at startup using Python's **native `tomllib`** (no third-party TOML library required).

| Section | Key | Default | Description |
|---|---|---|---|
| `[castor]` | `log_level` | `"INFO"` | Logging verbosity |
| `[server]` | `port` | `8080` | HTTP server port |
| `[prometheus]` | `url` | `http://localhost:9090` | Prometheus base URL |
| `[prometheus]` | `lookback_hours` | `72` | Historical window for training |
| `[model]` | `lag_windows` | `[5,15,30,60,120]` | Lag feature sizes (minutes) |
| `[model]` | `artifact_dir` | `./artifacts/models` | Model persistence directory |
| `[scheduler]` | `forecast_horizon_minutes` | `30` | How far ahead to predict |
| `[thresholds]` | `cpu_spike_ratio` | `0.80` | CPU ratio that triggers a spike alert |
| `[webhooks]` | `max_retries` | `3` | Webhook delivery retry limit |

Sensitive values (webhook secrets) can be set via the **`CASTOR_WEBHOOK_SECRET`** environment variable — they are never required to be stored in the TOML file.

---

## 📡 API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/predict` | Run on-demand inference for a metric |
| `GET` | `/v1/predict/{metric}` | Get latest prediction for `cpu`, `memory`, or `http_rps` |
| `POST` | `/v1/webhooks/register` | Register a new webhook target at runtime |
| `GET` | `/v1/webhooks` | List all registered webhook targets |
| `DELETE` | `/v1/webhooks/{name}` | Remove a webhook target by name |
| `GET` | `/healthz` | Kubernetes liveness probe |
| `GET` | `/readyz` | Kubernetes readiness probe |
| `GET` | `/docs` | Swagger UI |

---

## 🗺️ Roadmap — "Build in Public" (6-Week Plan)

| Week | Theme | Deliverables |
|---|---|---|
| **Week 1** 🏗️ | **Foundation** | Project scaffold (this release), CI pipeline, synthetic data engine, core XGBoost pipeline, README |
| **Week 2** 🔗 | **Live Prometheus Integration** | Real PromQL client, metric normalisation, graceful fallback, connection pooling, integration tests |
| **Week 3** 🤖 | **Model Hardening** | Hyperparameter tuning with Optuna, cross-validation on time-series folds, model versioning, experiment tracking |
| **Week 4** 🌐 | **KEDA & HPA Integration** | KEDA External Scaler gRPC implementation, Helm chart, sample `ScaledObject` manifests, end-to-end demo |
| **Week 5** 📊 | **Observability & Alerting** | Prometheus `/metrics` endpoint (exposing prediction outputs), Grafana dashboard template, PagerDuty webhook adapter |
| **Week 6** 🚀 | **Production Hardening** | Multi-model support (one model per namespace/service), online incremental learning, Kubernetes RBAC manifests, v0.1.0 GitHub Release |

> **Follow along:** Star ⭐ the repo and watch for weekly release tags. Community contributions welcome at any stage — see [CONTRIBUTING.md](./CONTRIBUTING.md).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Install dev dependencies: `uv sync`
4. Make your changes with tests
5. Run the full check suite: `uv run ruff check . && uv run mypy src/ && uv run pytest`
6. Open a Pull Request

---

## 📄 License

Distributed under the **Apache 2.0 License**. See [LICENSE](./LICENSE) for details.

---

<div align="center">

Built with ❤️ by **[DevOpsier](https://github.com/devopsier)**

Powered by [FastAPI](https://fastapi.tiangolo.com) · [XGBoost](https://xgboost.readthedocs.io) · [uv](https://docs.astral.sh/uv/) · [structlog](https://www.structlog.org)

</div>

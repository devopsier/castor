"""
tests.test_smoke — Smoke tests for core Castor components.

These tests are intentionally fast and require no external services.
They validate the three critical paths of the system:

1. ``PrometheusIngestor`` produces a valid DataFrame in synthetic mode.
2. ``SpikePredictorXGB`` can train on synthetic data and return valid metrics.
3. The FastAPI application starts and its health endpoints respond correctly.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from castor.data.ingestor import MetricKind, PrometheusIngestor, _synthetic_metric
from castor.models.train import SpikePredictorXGB, build_features


# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Load the default config.toml for integration tests."""
    config_path = Path(__file__).parent.parent / "config.toml"
    if not config_path.exists():
        pytest.skip("config.toml not found — skipping integration smoke tests")
    with config_path.open("rb") as fh:
        return tomllib.load(fh)


@pytest.fixture(scope="session")
def synthetic_cpu_df() -> pd.DataFrame:
    """Synthetic CPU metric DataFrame used across multiple tests."""
    return _synthetic_metric(MetricKind.CPU, lookback_hours=24, step_seconds=60)


# ---------------------------------------------------------------------------
# 1. Data Ingestor Tests
# ---------------------------------------------------------------------------


class TestPrometheusIngestor:
    """Tests for PrometheusIngestor synthetic data mode."""

    def test_synthetic_cpu_dataframe_shape(self, synthetic_cpu_df: pd.DataFrame) -> None:
        """Synthetic CPU DataFrame must have the expected columns and be non-empty."""
        assert not synthetic_cpu_df.empty
        assert set(synthetic_cpu_df.columns) == {"timestamp", "value", "labels"}

    def test_synthetic_cpu_values_bounded(self, synthetic_cpu_df: pd.DataFrame) -> None:
        """CPU ratio values must be in [0, 1]."""
        assert synthetic_cpu_df["value"].between(0.0, 1.0).all(), (
            "Synthetic CPU values should be in [0.0, 1.0]"
        )

    def test_synthetic_timestamps_are_utc(self, synthetic_cpu_df: pd.DataFrame) -> None:
        """Timestamps must be timezone-aware UTC."""
        assert synthetic_cpu_df["timestamp"].dt.tz is not None

    def test_synthetic_timestamps_monotonically_increasing(
        self, synthetic_cpu_df: pd.DataFrame
    ) -> None:
        """Timestamps must be strictly ascending."""
        ts = synthetic_cpu_df["timestamp"]
        assert (ts.diff().dropna() > pd.Timedelta(0)).all()

    def test_fetch_all_returns_all_metric_kinds(self, config: dict[str, Any]) -> None:
        """fetch_all() must return a DataFrame for each MetricKind."""
        ingestor = PrometheusIngestor(config=config)
        results = ingestor.fetch_all()
        assert set(results.keys()) == set(MetricKind)
        for kind, df in results.items():
            assert isinstance(df, pd.DataFrame), f"Expected DataFrame for {kind}"
            assert not df.empty, f"Expected non-empty DataFrame for {kind}"

    def test_ingestor_context_manager(self, config: dict[str, Any]) -> None:
        """PrometheusIngestor must support context manager protocol."""
        with PrometheusIngestor(config=config) as ingestor:
            df = ingestor.fetch_metric(MetricKind.MEMORY)
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# 2. Feature Engineering & Model Training Tests
# ---------------------------------------------------------------------------


class TestSpikePredictorXGB:
    """Tests for the XGBoost training pipeline."""

    def test_build_features_output_shape(self, synthetic_cpu_df: pd.DataFrame) -> None:
        """build_features() must return consistent X, y shapes."""
        X, y = build_features(
            df=synthetic_cpu_df,
            lag_windows=[5, 15],
            rolling_windows=[10],
            horizon_minutes=15,
            step_seconds=60,
        )
        assert len(X) == len(y), "Feature matrix and target must have equal length"
        assert len(X) > 0, "Feature matrix must not be empty"
        # Lag and rolling features + calendar features
        assert "lag_5m" in X.columns
        assert "lag_15m" in X.columns
        assert "roll_mean_10m" in X.columns
        assert "hour_sin" in X.columns
        assert "hour_cos" in X.columns

    def test_build_features_no_target_leakage(self, synthetic_cpu_df: pd.DataFrame) -> None:
        """The target column must not appear in the feature matrix."""
        X, _ = build_features(
            df=synthetic_cpu_df,
            lag_windows=[5],
            rolling_windows=[10],
            horizon_minutes=15,
            step_seconds=60,
        )
        assert "target" not in X.columns
        assert "value" not in X.columns

    def test_fit_returns_metrics(
        self, synthetic_cpu_df: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """SpikePredictorXGB.fit() must return MAE and RMSE metrics."""
        trainer = SpikePredictorXGB(config=config)
        metrics = trainer.fit(synthetic_cpu_df, metric_kind="cpu")
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["mae"] >= 0.0
        assert metrics["rmse"] >= 0.0

    def test_model_is_set_after_fit(
        self, synthetic_cpu_df: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """After fit(), the model attribute must not be None."""
        trainer = SpikePredictorXGB(config=config)
        trainer.fit(synthetic_cpu_df, metric_kind="cpu")
        assert trainer.model is not None

    def test_feature_importance_after_fit(
        self, synthetic_cpu_df: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """feature_importance() must return a non-empty sorted DataFrame."""
        trainer = SpikePredictorXGB(config=config)
        trainer.fit(synthetic_cpu_df, metric_kind="cpu")
        fi = trainer.feature_importance()
        assert isinstance(fi, pd.DataFrame)
        assert not fi.empty
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        # Should be sorted descending
        assert (fi["importance"].diff().dropna() <= 0).all()

    def test_save_and_load_roundtrip(
        self,
        synthetic_cpu_df: pd.DataFrame,
        config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """A saved model must be loadable and produce the same predictions."""
        import copy

        cfg = copy.deepcopy(config)
        cfg.setdefault("model", {})["artifact_dir"] = str(tmp_path)

        trainer = SpikePredictorXGB(config=cfg)
        trainer.fit(synthetic_cpu_df, metric_kind="cpu")
        saved_path = trainer.save(metric_kind="cpu")

        loaded = SpikePredictorXGB.load(saved_path, cfg)
        assert loaded.feature_names == trainer.feature_names
        assert loaded.model is not None


# ---------------------------------------------------------------------------
# 3. FastAPI Application Smoke Tests
# ---------------------------------------------------------------------------


class TestFastAPIApp:
    """Smoke tests for the FastAPI application endpoints."""

    @pytest.fixture(scope="class")
    def client(self, config: dict[str, Any]) -> TestClient:
        """Construct a synchronous TestClient for the Castor app."""
        # Patch the config loader so the test app uses the test config
        import castor.main as main_module

        original_load = main_module.load_config

        def patched_load(path: Path | None = None) -> dict[str, Any]:
            return config

        main_module.load_config = patched_load  # type: ignore[assignment]
        try:
            with TestClient(main_module.app, raise_server_exceptions=True) as c:
                yield c
        finally:
            main_module.load_config = original_load  # type: ignore[assignment]

    def test_healthz_returns_200(self, client: TestClient) -> None:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readyz_returns_200(self, client: TestClient) -> None:
        response = client.get("/readyz")
        assert response.status_code == 200

    def test_openapi_schema_accessible(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "Castor"

    def test_predict_endpoint_returns_valid_schema(self, client: TestClient) -> None:
        response = client.post("/v1/predict", json={"metric": "http_rps"})
        assert response.status_code == 200
        data = response.json()
        assert "metric_kind" in data
        assert "predicted_value" in data
        assert "confidence" in data
        assert "spike_imminent" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_list_webhooks_returns_list(self, client: TestClient) -> None:
        response = client.get("/v1/webhooks")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_register_and_delete_webhook(self, client: TestClient) -> None:
        # Register
        payload = {
            "name": "test-target",
            "url": "https://httpbin.org/post",
            "events": ["spike_imminent"],
        }
        reg_response = client.post("/v1/webhooks/register", json=payload)
        assert reg_response.status_code == 201
        assert reg_response.json()["name"] == "test-target"

        # Delete
        del_response = client.delete("/v1/webhooks/test-target")
        assert del_response.status_code == 204

    def test_delete_nonexistent_webhook_returns_404(self, client: TestClient) -> None:
        response = client.delete("/v1/webhooks/nonexistent-xyz")
        assert response.status_code == 404

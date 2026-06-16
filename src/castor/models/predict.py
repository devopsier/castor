"""
castor.models.predict — Inference logic for the spike forecaster.

``SpikePredictor`` is a thin facade that:

* Loads trained ``SpikePredictorXGB`` models from the artifact directory.
* Accepts the latest slice of metric data and returns a ``PredictionResult``.
* Computes a **confidence score** derived from the spread of XGBoost tree
  leaf predictions (a proxy for epistemic uncertainty).
* Determines whether the prediction exceeds a configured spike threshold.

Typical usage::

    predictor = SpikePredictor(config=cfg)
    result = predictor.predict(MetricKind.HTTP_RPS, recent_df)
    if result.spike_imminent:
        await webhook_dispatcher.fire("spike_imminent", result)
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from castor.data.ingestor import MetricKind
from castor.models.train import SpikePredictorXGB, build_features

logger: structlog.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result Types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class PredictionResult:
    """Immutable result of a spike prediction inference call.

    Attributes:
        metric_kind: Which metric was evaluated.
        current_value: The most recent observed metric value.
        predicted_value: Forecasted metric value at ``horizon_minutes`` ahead.
        horizon_minutes: Forecast look-ahead window in minutes.
        confidence: Score in [0, 1] — higher means more certain.
        spike_imminent: ``True`` if the prediction exceeds the configured
            spike threshold *and* confidence is above the minimum.
        threshold_used: The absolute threshold value that was compared against.
        evaluated_at: UTC timestamp at the moment of inference.
    """

    metric_kind: MetricKind
    current_value: float
    predicted_value: float
    horizon_minutes: int
    confidence: float
    spike_imminent: bool
    threshold_used: float
    evaluated_at: dt.datetime = dataclasses.field(
        default_factory=lambda: dt.datetime.now(tz=dt.UTC)
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary suitable for JSON encoding."""
        return {
            "metric_kind": self.metric_kind.value,
            "current_value": self.current_value,
            "predicted_value": self.predicted_value,
            "horizon_minutes": self.horizon_minutes,
            "confidence": round(self.confidence, 4),
            "spike_imminent": self.spike_imminent,
            "threshold_used": self.threshold_used,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# SpikePredictor — Inference Facade
# ---------------------------------------------------------------------------


class SpikePredictor:
    """Facade for loading trained models and running spike inference.

    Manages a cache of loaded ``SpikePredictorXGB`` instances, one per
    ``MetricKind``.  Models are loaded lazily on first prediction request.

    Args:
        config: Parsed configuration dictionary (output of ``load_config()``).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._model_cfg: dict[str, Any] = config.get("model", {})
        self._artifact_dir: Path = Path(self._model_cfg.get("artifact_dir", "./artifacts/models"))
        self._threshold_cfg: dict[str, Any] = config.get("thresholds", {})
        self._horizon_minutes: int = int(
            config.get("scheduler", {}).get("forecast_horizon_minutes", 30)
        )
        self._lag_windows: list[int] = list(self._model_cfg.get("lag_windows", [5, 15, 30, 60]))
        self._rolling_windows: list[int] = list(
            self._model_cfg.get("rolling_windows", [10, 30, 60])
        )
        self._min_confidence: float = float(self._threshold_cfg.get("min_confidence", 0.70))

        # Lazy model cache: MetricKind → SpikePredictorXGB
        self._model_cache: dict[MetricKind, SpikePredictorXGB] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, kind: MetricKind, df: pd.DataFrame) -> PredictionResult:
        """Run a spike prediction for the given metric type.

        Loads the corresponding trained model (from disk or cache), builds
        features from the most recent metric slice, and returns a
        ``PredictionResult``.

        Args:
            kind: The metric type to predict (CPU, MEMORY, or HTTP_RPS).
            df: Recent metric DataFrame with columns ``["timestamp", "value"]``.
                Should contain at least ``max(lag_windows)`` minutes of history.

        Returns:
            A ``PredictionResult`` dataclass with all inference outputs.

        Raises:
            FileNotFoundError: If no trained model artifact exists for ``kind``
                and a synthetic model cannot be created.
        """
        trainer = self._load_or_train(kind, df)
        if trainer.model is None:
            # Model training failed; return a safe no-op prediction
            logger.error("model_unavailable", metric=kind.value)
            return self._noop_result(kind, df)

        # Build features for the latest available row
        X, _ = build_features(
            df=df,
            lag_windows=self._lag_windows,
            rolling_windows=self._rolling_windows,
            horizon_minutes=self._horizon_minutes,
            step_seconds=int(self._config.get("prometheus", {}).get("step_seconds", 60)),
        )

        if X.empty:
            logger.warning("insufficient_data_for_prediction", metric=kind.value, rows=len(df))
            return self._noop_result(kind, df)

        # Predict on the last row (most recent timestep)
        X_latest = X.tail(1)
        predicted_value = float(trainer.model.predict(X_latest)[0])
        current_value = float(df["value"].iloc[-1]) if not df.empty else 0.0
        confidence = self._compute_confidence(trainer, X_latest)
        threshold = self._resolve_threshold(kind, current_value)
        spike_imminent = (predicted_value >= threshold) and (confidence >= self._min_confidence)

        result = PredictionResult(
            metric_kind=kind,
            current_value=current_value,
            predicted_value=predicted_value,
            horizon_minutes=self._horizon_minutes,
            confidence=confidence,
            spike_imminent=spike_imminent,
            threshold_used=threshold,
        )
        logger.info("prediction_complete", **result.to_dict())
        return result

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _load_or_train(self, kind: MetricKind, df: pd.DataFrame) -> SpikePredictorXGB:
        """Return a trained model, loading from disk or training on-the-fly."""
        if kind in self._model_cache:
            return self._model_cache[kind]

        artifact_path = self._artifact_dir / f"castor_{kind.value}.ubj"
        if artifact_path.exists():
            trainer = SpikePredictorXGB.load(artifact_path, self._config)
        else:
            logger.warning(
                "no_model_artifact_found",
                metric=kind.value,
                path=str(artifact_path),
                action="training_on_the_fly",
            )
            trainer = SpikePredictorXGB(config=self._config)
            trainer.fit(df, metric_kind=kind.value)
            try:
                trainer.save(metric_kind=kind.value)
            except OSError as exc:
                logger.error("model_save_failed", exc_info=exc)

        self._model_cache[kind] = trainer
        return trainer

    def _compute_confidence(
        self,
        trainer: SpikePredictorXGB,
        X: pd.DataFrame,
    ) -> float:
        """Estimate prediction confidence using XGBoost leaf variance.

        XGBoost supports returning leaf indices (``pred_leaf=True``), which
        encode which terminal node each tree routed the sample to.  The
        variance across tree leaves serves as a proxy for ensemble disagreement
        and therefore model uncertainty.

        A high-variance leaf assignment → low confidence.
        A low-variance leaf assignment → high confidence.

        The raw score is normalised to [0, 1] via a monotone sigmoid squash.

        Args:
            trainer: A trained ``SpikePredictorXGB`` instance.
            X: Feature matrix for a single sample (shape: ``[1, n_features]``).

        Returns:
            Confidence score in ``[0.0, 1.0]``.
        """
        if trainer.model is None:
            return 0.0

        try:
            dmatrix = trainer.model.get_booster().DMatrix(X)  # type: ignore[attr-defined]
            leaves: np.ndarray = trainer.model.get_booster().predict(dmatrix, pred_leaf=True)
            # leaves shape: (1, n_trees) — variance across trees for this sample
            leaf_std: float = float(np.std(leaves[0]))
            # Sigmoid-based normalisation: low std → high confidence
            confidence: float = float(1.0 / (1.0 + leaf_std / 10.0))
        except Exception as exc:
            logger.warning("confidence_estimation_failed", exc_info=exc)
            confidence = 0.5  # Fallback: indeterminate confidence

        return round(min(max(confidence, 0.0), 1.0), 4)

    def _resolve_threshold(self, kind: MetricKind, current_value: float) -> float:
        """Translate config thresholds into an absolute metric threshold value.

        Args:
            kind: Metric type.
            current_value: Latest observed value (used for percentage-based
                thresholds such as ``http_rps_spike_pct``).

        Returns:
            The absolute threshold value to compare ``predicted_value`` against.
        """
        match kind:
            case MetricKind.CPU:
                return float(self._threshold_cfg.get("cpu_spike_ratio", 0.80))
            case MetricKind.HTTP_RPS:
                pct = float(self._threshold_cfg.get("http_rps_spike_pct", 50.0))
                return current_value * (1.0 + pct / 100.0)
            case MetricKind.MEMORY:
                # Memory: trigger if predicted exceeds 90% of available RAM.
                # This is a placeholder; real capacity should come from Kubernetes.
                return float(self._threshold_cfg.get("memory_threshold_bytes", 900 * 1024 * 1024))
            case _:
                return float("inf")

    @staticmethod
    def _noop_result(kind: MetricKind, df: pd.DataFrame) -> PredictionResult:
        """Return a safe, non-alarming result when a real prediction is unavailable."""
        current = float(df["value"].iloc[-1]) if not df.empty else 0.0
        return PredictionResult(
            metric_kind=kind,
            current_value=current,
            predicted_value=current,
            horizon_minutes=0,
            confidence=0.0,
            spike_imminent=False,
            threshold_used=float("inf"),
        )

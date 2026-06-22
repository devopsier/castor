"""
castor.models.train — XGBoost training pipeline with lag feature engineering.

This module contains ``SpikePredictorXGB``, which:

* Consumes a raw metric ``pandas.DataFrame`` (as produced by
  ``PrometheusIngestor``).
* Constructs a rich feature matrix via **lag features** and **rolling-window
  statistics** (mean, std, min, max).
* Trains an XGBoost regressor to predict the metric *N* minutes into the
  future.
* Optionally applies an **asymmetric loss function** (custom XGBoost objective)
  that penalises under-prediction (false negatives) more heavily than
  over-prediction — critical for proactive autoscaling.
* Persists trained models to disk using ``joblib`` for subsequent inference.

Typical usage::

    from castor.data.ingestor import PrometheusIngestor, MetricKind
    from castor.models.train import SpikePredictorXGB

    ingestor = PrometheusIngestor(config=cfg)
    df = ingestor.fetch_metric(MetricKind.HTTP_RPS)

    trainer = SpikePredictorXGB(config=cfg)
    trainer.fit(df)
    trainer.save()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import structlog
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FeatureMatrix = pd.DataFrame
TargetVector = pd.Series


# ---------------------------------------------------------------------------
# Asymmetric Loss (Custom XGBoost Objective)
# ---------------------------------------------------------------------------


def asymmetric_mse_objective(
    y_pred: np.ndarray,
    dtrain: xgb.DMatrix,
    *,
    under_penalty: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Asymmetric squared-error objective for XGBoost.

    Penalises **under-prediction** (residual < 0, i.e. prediction is below the
    true value) by ``under_penalty`` times more than over-prediction.  This is
    desirable for proactive autoscaling: better to scale up unnecessarily than
    to under-provision during a real traffic spike.

    The gradient and Hessian are derived from the following loss::

        L(r) = 0.5 * w(r) * r²
        w(r) = under_penalty  if r < 0  (we under-predicted)
               1.0             otherwise

    where ``r = y_pred - y_true``.

    Args:
        y_pred: XGBoost model predictions (1-D array).
        dtrain: XGBoost ``DMatrix`` containing the true labels.
        under_penalty: Multiplier applied to the gradient/hessian when the
            model under-predicts.  Must be ≥ 1.0.

    Returns:
        A ``(gradient, hessian)`` tuple, each of shape ``(n_samples,)``.
    """
    y_true: np.ndarray = dtrain.get_label()
    residual: np.ndarray = y_pred - y_true

    # Weight vector: larger penalty when we under-predict (residual < 0)
    weights: np.ndarray = np.where(residual < 0, under_penalty, 1.0)

    gradient: np.ndarray = weights * residual
    hessian: np.ndarray = weights * np.ones_like(residual)

    return gradient, hessian


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------


def build_features(
    df: pd.DataFrame,
    lag_windows: list[int],
    rolling_windows: list[int],
    horizon_minutes: int,
    step_seconds: int = 60,
) -> tuple[FeatureMatrix, TargetVector]:
    """Transform a raw metric DataFrame into a supervised learning dataset.

    Feature construction strategy:

    1. **Lag features** — the metric value at ``t - k`` minutes for each ``k``
       in ``lag_windows``.  These capture the immediate history and
       autocorrelation structure.
    2. **Rolling statistics** — mean, standard deviation, min, and max over
       each window in ``rolling_windows``.  These encode trend and volatility.
    3. **Calendar features** — hour of day, day of week, and minute of hour
       extracted from ``timestamp``.  These capture diurnal patterns.
    4. **Target** — the metric value at ``t + horizon_minutes`` (forward-shifted
       by the forecast horizon).

    Rows containing ``NaN`` values (head and tail of each series) are dropped.

    Args:
        df: Raw DataFrame with columns ``["timestamp", "value"]``.
        lag_windows: Lag sizes in minutes, e.g. ``[5, 15, 30, 60]``.
        rolling_windows: Rolling window sizes in minutes, e.g. ``[10, 30, 60]``.
        horizon_minutes: Number of minutes ahead to predict.
        step_seconds: Data resolution in seconds (used to convert minutes to
            integer row offsets).

    Returns:
        A ``(X, y)`` tuple where ``X`` is the feature ``DataFrame`` and ``y``
        is the target ``Series``.
    """
    step_minutes: float = step_seconds / 60.0
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["value"] = df["value"].astype(np.float64)

    # -- Lag features -------------------------------------------------------
    for lag_min in lag_windows:
        lag_periods = max(1, round(lag_min / step_minutes))
        df[f"lag_{lag_min}m"] = df["value"].shift(lag_periods)

    # -- Rolling statistics -------------------------------------------------
    for win_min in rolling_windows:
        win_periods = max(1, round(win_min / step_minutes))
        roll = df["value"].rolling(window=win_periods, min_periods=1)
        df[f"roll_mean_{win_min}m"] = roll.mean()
        df[f"roll_std_{win_min}m"] = roll.std().fillna(0.0)
        df[f"roll_min_{win_min}m"] = roll.min()
        df[f"roll_max_{win_min}m"] = roll.max()

    # -- Calendar features --------------------------------------------------
    df["hour_of_day"] = df["timestamp"].dt.hour.astype(np.float32)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(np.float32)
    df["minute_of_hour"] = df["timestamp"].dt.minute.astype(np.float32)
    # Cyclical encoding of hour (avoids discontinuity at midnight)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)

    # -- Target (forward shift) ---------------------------------------------
    horizon_periods = max(1, round(horizon_minutes / step_minutes))
    df["target"] = df["value"].shift(-horizon_periods)

    # Drop rows with missing values caused by lag/shift
    df.dropna(inplace=True)

    feature_cols: list[str] = [
        c for c in df.columns if c not in ("timestamp", "value", "target", "labels")
    ]
    X: FeatureMatrix = df[feature_cols].reset_index(drop=True)
    y: TargetVector = df["target"].reset_index(drop=True)

    logger.info(
        "features_built",
        n_features=len(feature_cols),
        n_samples=len(X),
        horizon_minutes=horizon_minutes,
    )
    return X, y


# ---------------------------------------------------------------------------
# XGBoost Trainer
# ---------------------------------------------------------------------------


class SpikePredictorXGB:
    """XGBoost-based time-series spike forecaster.

    Wraps an ``xgboost.XGBRegressor`` with Castor-specific feature engineering,
    an asymmetric loss objective, and model persistence helpers.

    Args:
        config: Parsed configuration dictionary (output of ``load_config()``).

    Attributes:
        model: The underlying ``xgboost.XGBRegressor`` instance.  ``None``
            until ``fit()`` is called.
        feature_names: Ordered list of feature column names from training.
            Used to validate inference inputs.

    Example::

        trainer = SpikePredictorXGB(config=cfg)
        trainer.fit(df, metric_kind="http_rps")
        trainer.save()
    """

    MODEL_FILENAME_TEMPLATE = "castor_{metric_kind}.ubj"

    def __init__(self, config: dict[str, Any]) -> None:
        model_cfg: dict[str, Any] = config.get("model", {})
        sched_cfg: dict[str, Any] = config.get("scheduler", {})

        self._artifact_dir: Path = Path(model_cfg.get("artifact_dir", "./artifacts/models"))
        self._lag_windows: list[int] = list(model_cfg.get("lag_windows", [5, 15, 30, 60, 120]))
        self._rolling_windows: list[int] = list(model_cfg.get("rolling_windows", [10, 30, 60]))
        self._validation_split: float = float(model_cfg.get("validation_split", 0.15))
        self._horizon_minutes: int = int(
            sched_cfg.get("forecast_horizon_minutes", 30)
        )
        self._use_asymmetric_loss: bool = True

        xgb_params: dict[str, Any] = dict(model_cfg.get("xgb_params", {}))
        xgb_params.setdefault("n_estimators", 300)
        xgb_params.setdefault("max_depth", 6)
        xgb_params.setdefault("learning_rate", 0.05)
        xgb_params.setdefault("subsample", 0.8)
        xgb_params.setdefault("colsample_bytree", 0.8)
        xgb_params.setdefault("tree_method", "hist")
        xgb_params.setdefault("seed", 42)

        # When using a custom objective the built-in objective string is ignored
        if self._use_asymmetric_loss:
            xgb_params.pop("objective", None)

        self.model: xgb.XGBRegressor | None = None
        self.feature_names: list[str] = []
        self._xgb_params: dict[str, Any] = xgb_params

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        metric_kind: str = "unknown",
        *,
        step_seconds: int = 60,
    ) -> dict[str, float]:
        """Train the XGBoost model on a metric time-series DataFrame.

        Performs the following pipeline steps:
        1. Feature engineering via ``build_features()``.
        2. Chronological train/validation split (no shuffle — preserves
           temporal order).
        3. Model training, optionally with the asymmetric loss objective.
        4. Evaluation on the held-out validation set.

        Args:
            df: Raw metric ``DataFrame`` with at least ``["timestamp", "value"]``
                columns.
            metric_kind: String tag used for artifact naming and logging.
            step_seconds: Resolution of ``df`` in seconds.

        Returns:
            A dictionary with validation metrics: ``{"mae": ..., "rmse": ...}``.
        """
        logger.info("training_started", metric_kind=metric_kind)

        X, y = build_features(
            df=df,
            lag_windows=self._lag_windows,
            rolling_windows=self._rolling_windows,
            horizon_minutes=self._horizon_minutes,
            step_seconds=step_seconds,
        )
        self.feature_names = list(X.columns)

        # Chronological split — do NOT shuffle time-series data
        split_idx = int(len(X) * (1 - self._validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model = xgb.XGBRegressor(**self._xgb_params)

        if self._use_asymmetric_loss:
            # XGBoost custom objective: pass a closure that captures no mutable state
            def _objective(
                y_pred: np.ndarray,
                dtrain: xgb.DMatrix,
            ) -> tuple[np.ndarray, np.ndarray]:
                return asymmetric_mse_objective(y_pred, dtrain, under_penalty=2.5)

            self.model.set_params(objective=_objective)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate on validation split
        y_pred_val: np.ndarray = self.model.predict(X_val)
        mae: float = float(mean_absolute_error(y_val, y_pred_val))
        rmse: float = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))

        metrics = {"mae": mae, "rmse": rmse}
        logger.info("training_complete", metric_kind=metric_kind, **metrics)
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, metric_kind: str = "model") -> Path:
        """Serialise the trained model to a ``.ubj`` (binary JSON) artifact.

        The artifact directory is created if it does not exist.

        Args:
            metric_kind: Tag incorporated into the filename so each metric type
                has its own model file.

        Returns:
            The absolute ``Path`` to the saved model file.

        Raises:
            RuntimeError: If ``fit()`` has not been called yet.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        filename = self.MODEL_FILENAME_TEMPLATE.format(metric_kind=metric_kind)
        path = self._artifact_dir / filename
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)
        logger.info("model_saved", path=str(path))
        return path

    @classmethod
    def load(cls, path: Path, config: dict[str, Any]) -> SpikePredictorXGB:
        """Load a previously saved model from disk.

        Args:
            path: Path to the ``.ubj`` artefact saved by ``save()``.
            config: Active application configuration dictionary.

        Returns:
            A ``SpikePredictorXGB`` instance with ``model`` and
            ``feature_names`` restored.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        artefact: dict[str, Any] = joblib.load(path)
        instance = cls(config=config)
        instance.model = artefact["model"]
        instance.feature_names = artefact["feature_names"]
        logger.info("model_loaded", path=str(path))
        return instance

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        """Return a sorted DataFrame of feature importances.

        Requires the model to have been trained.

        Returns:
            A ``DataFrame`` with columns ``["feature", "importance"]``,
            sorted descending by importance score.

        Raises:
            RuntimeError: If the model is not yet trained.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        scores: np.ndarray = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": scores})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

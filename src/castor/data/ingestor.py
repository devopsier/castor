"""
castor.data.ingestor — Prometheus API client and metric collection.

This module provides ``PrometheusIngestor``, a class responsible for:

* Parsing ``config.toml`` with Python's native ``tomllib``.
* Executing PromQL range-queries against the Prometheus HTTP API using ``httpx``.
* Normalising the raw Prometheus JSON response into a typed ``pandas.DataFrame``.
* Falling back to synthetic data generation when Prometheus is unreachable
  (useful for local development and integration tests).

Typical usage::

    ingestor = PrometheusIngestor(config=load_config())
    df = ingestor.fetch_metric(MetricKind.HTTP_RPS)
"""

from __future__ import annotations

import datetime as dt
import enum
from typing import Any, ClassVar

import httpx
import numpy as np
import pandas as pd
import structlog

logger: structlog.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Public Enumerations
# ---------------------------------------------------------------------------


class MetricKind(enum.StrEnum):
    """Enumeration of the metric types Castor can ingest."""

    CPU = "cpu"
    MEMORY = "memory"
    HTTP_RPS = "http_rps"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_prometheus_range_result(raw: dict[str, Any]) -> pd.DataFrame:
    """Convert a Prometheus ``/query_range`` JSON response into a DataFrame.

    The Prometheus range-query response has the shape::

        {
          "status": "success",
          "data": {
            "resultType": "matrix",
            "result": [
              {
                "metric": {"__name__": "...", "pod": "..."},
                "values": [[<unix_ts>, "<value>"], ...]
              },
              ...
            ]
          }
        }

    Args:
        raw: Parsed JSON dictionary from the Prometheus API response.

    Returns:
        A ``pandas.DataFrame`` with columns ``["timestamp", "value", "labels"]``
        where ``timestamp`` is timezone-aware UTC and ``value`` is ``float64``.

    Raises:
        ValueError: If ``raw["status"]`` is not ``"success"``.
    """
    if raw.get("status") != "success":
        raise ValueError(f"Prometheus query failed: {raw.get('error', 'unknown error')}")

    records: list[dict[str, Any]] = []
    for series in raw.get("data", {}).get("result", []):
        label_str = str(series.get("metric", {}))
        for unix_ts, value_str in series.get("values", []):
            records.append(
                {
                    "timestamp": pd.Timestamp(unix_ts, unit="s", tz="UTC"),
                    "value": float(value_str),
                    "labels": label_str,
                }
            )

    if not records:
        return pd.DataFrame(columns=["timestamp", "value", "labels"])

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _synthetic_metric(
    metric_kind: MetricKind,
    lookback_hours: int,
    step_seconds: int,
) -> pd.DataFrame:
    """Generate realistic synthetic time-series data for offline development.

    Produces a sine-wave baseline with injected Gaussian noise and a simulated
    traffic spike, enabling full pipeline testing without a live Prometheus
    instance.

    Args:
        metric_kind: Which metric to simulate.
        lookback_hours: Duration of the generated time window (hours).
        step_seconds: Interval between synthetic data points (seconds).

    Returns:
        A ``pandas.DataFrame`` with columns ``["timestamp", "value", "labels"]``.
    """
    end = dt.datetime.now(tz=dt.UTC)
    start = end - dt.timedelta(hours=lookback_hours)
    freq = pd.tseries.frequencies.to_offset(f"{step_seconds}s")
    timestamps = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    n = len(timestamps)

    rng = np.random.default_rng(seed=42)
    t = np.linspace(0, 4 * np.pi, n)

    match metric_kind:
        case MetricKind.CPU:
            # CPU ratio: 0.15 - 0.95 with a sine wave and spike
            baseline = 0.35 + 0.20 * np.sin(t) + rng.normal(0, 0.02, n)
            spike_idx = n // 2
            baseline[spike_idx : spike_idx + 30] += 0.45
            values = np.clip(baseline, 0.0, 1.0)
        case MetricKind.MEMORY:
            # Memory (bytes): steady growth with fluctuation ~512 MiB
            baseline = (0.5 + 0.1 * np.sin(t)) * 512 * 1024 * 1024
            values = baseline + rng.normal(0, 10 * 1024 * 1024, n)
        case MetricKind.HTTP_RPS:
            # HTTP req/s: workday pattern, 10-800 rps
            baseline = 200 + 300 * np.sin(t / 2) + rng.normal(0, 15, n)
            spike_idx = int(n * 0.65)
            baseline[spike_idx : spike_idx + 20] += 500
            values = np.clip(baseline, 0.0, None)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": values,
            "labels": f'{{synthetic="true", metric="{metric_kind.value}"}}',
        }
    )
    logger.warning(
        "using_synthetic_data",
        metric=metric_kind.value,
        rows=len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Main Class
# ---------------------------------------------------------------------------


class PrometheusIngestor:
    """HTTP client for the Prometheus query API with typed DataFrame output.

    Reads all connection parameters from the ``[prometheus]`` section of
    ``config.toml``.  Callers should instantiate this once and reuse it;
    internally it holds a persistent ``httpx.Client`` connection pool.

    Args:
        config: Parsed configuration dictionary (output of ``load_config()``).

    Example::

        ingestor = PrometheusIngestor(config=cfg)
        df = ingestor.fetch_metric(MetricKind.CPU)
        print(df.head())
    """

    # PromQL query config keys by MetricKind
    _QUERY_KEYS: ClassVar[dict[MetricKind, str]] = {
        MetricKind.CPU: "query_cpu",
        MetricKind.MEMORY: "query_memory",
        MetricKind.HTTP_RPS: "query_http_rps",
    }

    def __init__(self, config: dict[str, Any]) -> None:
        self._prom_cfg: dict[str, Any] = config.get("prometheus", {})
        self._base_url: str = self._prom_cfg.get("url", "http://localhost:9090")
        self._timeout: float = float(self._prom_cfg.get("timeout_seconds", 30))
        self._lookback_hours: int = int(self._prom_cfg.get("lookback_hours", 72))
        self._step_seconds: int = int(self._prom_cfg.get("step_seconds", 60))
        self._synthetic_fallback: bool = True  # Flip to False in strict prod mode

        self._client: httpx.Client = httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={"User-Agent": "castor/0.1.0"},
        )
        logger.info("prometheus_ingestor_initialized", base_url=self._base_url)

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def fetch_metric(self, kind: MetricKind) -> pd.DataFrame:
        """Fetch a single metric time-series from Prometheus.

        Executes a ``/api/v1/query_range`` HTTP GET request with a PromQL
        expression resolved from ``config.toml``.  If the request fails
        (network error or non-success status), and ``_synthetic_fallback``
        is ``True``, a synthetic dataset is returned so the rest of the
        pipeline can continue operating.

        Args:
            kind: The metric type to retrieve (CPU, MEMORY, or HTTP_RPS).

        Returns:
            A ``pandas.DataFrame`` with columns:
            * ``timestamp`` (``DatetimeTZDtype[ns, UTC]``) — observation time.
            * ``value`` (``float64``) — metric value at that timestamp.
            * ``labels`` (``object``) — serialised Prometheus label set.

        Raises:
            RuntimeError: If the query fails and synthetic fallback is disabled.
        """
        query_key = self._QUERY_KEYS[kind]
        promql = self._prom_cfg.get(query_key, "")
        if not promql:
            logger.warning("missing_promql_query", metric=kind.value, key=query_key)
            return self._fallback(kind)

        end = dt.datetime.now(tz=dt.UTC)
        start = end - dt.timedelta(hours=self._lookback_hours)

        params: dict[str, str] = {
            "query": promql,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "step": str(self._step_seconds),
        }

        try:
            response = self._client.get("/api/v1/query_range", params=params)
            response.raise_for_status()
            raw: dict[str, Any] = response.json()
            df = _parse_prometheus_range_result(raw)
            logger.info("metric_fetched", metric=kind.value, rows=len(df))
            return df

        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.error("prometheus_fetch_error", metric=kind.value, exc_info=exc)
            return self._fallback(kind)

    def fetch_all(self) -> dict[MetricKind, pd.DataFrame]:
        """Fetch all configured metric kinds in a single call.

        Returns:
            A mapping from ``MetricKind`` to its corresponding ``DataFrame``.
        """
        results: dict[MetricKind, pd.DataFrame] = {}
        for kind in MetricKind:
            results[kind] = self.fetch_metric(kind)
        return results

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _fallback(self, kind: MetricKind) -> pd.DataFrame:
        """Return synthetic data or raise, depending on ``_synthetic_fallback``."""
        if self._synthetic_fallback:
            return _synthetic_metric(kind, self._lookback_hours, self._step_seconds)
        raise RuntimeError(
            f"Prometheus query failed for metric '{kind.value}' and synthetic fallback is disabled."
        )

    def close(self) -> None:
        """Release the underlying ``httpx`` connection pool.

        Should be called during application shutdown.
        """
        self._client.close()
        logger.info("prometheus_ingestor_closed")

    def __enter__(self) -> PrometheusIngestor:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

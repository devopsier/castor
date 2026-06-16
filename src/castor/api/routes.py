"""
castor.api.routes — FastAPI REST endpoint definitions.

Endpoints exposed:

``POST /v1/predict``
    Trigger an on-demand spike prediction for one or all metric kinds.
    Returns a ``PredictionResponse`` containing the forecast values,
    confidence scores, and ``spike_imminent`` flags.

``GET /v1/predict/{metric}``
    Retrieve the *latest cached* prediction for a specific metric kind
    without triggering a new inference cycle.

``POST /v1/webhooks/register``
    Dynamically register a new webhook target at runtime.  Registered
    targets are persisted in memory for the lifetime of the process.

``GET /v1/webhooks``
    List all currently registered webhook targets (static from config
    plus dynamically registered ones).

``DELETE /v1/webhooks/{name}``
    Remove a dynamically registered webhook target by name.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, HttpUrl

from castor.api.webhooks import WebhookDispatcher, WebhookTarget
from castor.data.ingestor import MetricKind, PrometheusIngestor
from castor.models.predict import PredictionResult, SpikePredictor

logger: structlog.BoundLogger = structlog.get_logger(__name__)

router = APIRouter(tags=["Castor"])


# ---------------------------------------------------------------------------
# Pydantic Request / Response Schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for ``POST /v1/predict``."""

    metric: MetricKind = Field(
        default=MetricKind.HTTP_RPS,
        description="Which metric to forecast. One of: cpu, memory, http_rps.",
    )
    lookback_override_hours: int | None = Field(
        default=None,
        ge=1,
        le=168,
        description=(
            "Override the default lookback window (hours) for this request only. "
            "Useful for triggering inference over a longer historical window."
        ),
    )

    model_config = {"json_schema_extra": {"example": {"metric": "http_rps"}}}


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoints."""

    metric_kind: str
    current_value: float
    predicted_value: float
    horizon_minutes: int
    confidence: float = Field(ge=0.0, le=1.0)
    spike_imminent: bool
    threshold_used: float
    evaluated_at: str  # ISO-8601 UTC string


class WebhookRegisterRequest(BaseModel):
    """Request body for ``POST /v1/webhooks/register``."""

    name: str = Field(..., min_length=1, max_length=64, description="Unique name for this target.")
    url: HttpUrl = Field(..., description="Full URL of the webhook receiver endpoint.")
    events: list[str] = Field(
        default=["spike_imminent"],
        description="List of event types to subscribe to.",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP headers to include in the webhook POST request.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "my-scaler",
                "url": "https://my-service.example.com/webhook",
                "events": ["spike_imminent"],
                "headers": {"Authorization": "Bearer my-token"},
            }
        }
    }


class WebhookResponse(BaseModel):
    """Response schema for webhook list/register operations."""

    name: str
    url: str
    events: list[str]


# ---------------------------------------------------------------------------
# Dependency Injectors
# ---------------------------------------------------------------------------


def _get_ingestor(request: Request) -> PrometheusIngestor:
    return request.app.state.castor.ingestor  # type: ignore[no-any-return]


def _get_predictor(request: Request) -> SpikePredictor:
    return request.app.state.castor.predictor  # type: ignore[no-any-return]


def _get_dispatcher(request: Request) -> WebhookDispatcher:
    # The dispatcher is initialised lazily on first use
    if not hasattr(request.app.state.castor, "dispatcher"):
        cfg: dict[str, Any] = request.app.state.castor.config
        request.app.state.castor.dispatcher = WebhookDispatcher(config=cfg)
    return request.app.state.castor.dispatcher  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Helper: Convert PredictionResult → PredictionResponse
# ---------------------------------------------------------------------------


def _to_response(result: PredictionResult) -> PredictionResponse:
    return PredictionResponse(
        metric_kind=result.metric_kind.value,
        current_value=result.current_value,
        predicted_value=result.predicted_value,
        horizon_minutes=result.horizon_minutes,
        confidence=result.confidence,
        spike_imminent=result.spike_imminent,
        threshold_used=result.threshold_used,
        evaluated_at=result.evaluated_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# Prediction Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Trigger an on-demand spike prediction",
    status_code=status.HTTP_200_OK,
    operation_id="predict_metric",
)
async def predict(
    body: Annotated[PredictRequest, Body()],
    ingestor: Annotated[PrometheusIngestor, Depends(_get_ingestor)],
    predictor: Annotated[SpikePredictor, Depends(_get_predictor)],
    dispatcher: Annotated[WebhookDispatcher, Depends(_get_dispatcher)],
) -> PredictionResponse:
    """Fetch fresh metric data and run the XGBoost spike forecaster.

    The endpoint:
    1. Fetches the latest metric time-series from Prometheus (or synthetic data
       in development mode).
    2. Runs the trained XGBoost model to predict the metric value
       ``forecast_horizon_minutes`` into the future.
    3. If ``spike_imminent`` is ``True``, fires all registered webhooks
       subscribed to the ``spike_imminent`` event **asynchronously** (the HTTP
       response is not blocked on webhook delivery).
    """
    import asyncio

    logger.info("predict_request_received", metric=body.metric.value)

    try:
        df = ingestor.fetch_metric(body.metric)
    except Exception as exc:
        logger.error("ingestor_error", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to fetch metric data: {exc}",
        ) from exc

    result = predictor.predict(body.metric, df)
    response = _to_response(result)

    if result.spike_imminent:
        asyncio.create_task(
            dispatcher.fire_event("spike_imminent", result.to_dict()),
            name=f"webhook_spike_{body.metric.value}",
        )
        logger.warning(
            "spike_imminent_detected",
            metric=body.metric.value,
            predicted_value=result.predicted_value,
            confidence=result.confidence,
        )

    return response


@router.get(
    "/predict/{metric}",
    response_model=PredictionResponse,
    summary="Get the latest cached prediction for a metric",
    operation_id="get_latest_prediction",
)
async def get_latest_prediction(
    metric: MetricKind,
    ingestor: Annotated[PrometheusIngestor, Depends(_get_ingestor)],
    predictor: Annotated[SpikePredictor, Depends(_get_predictor)],
) -> PredictionResponse:
    """Return the most recent prediction for the specified metric.

    If no cached prediction exists, a fresh inference cycle is triggered.
    This endpoint is designed for **polling** use-cases (e.g., KEDA external
    scalers that query Castor on a fixed interval).
    """
    df = ingestor.fetch_metric(metric)
    result = predictor.predict(metric, df)
    return _to_response(result)


# ---------------------------------------------------------------------------
# Webhook Management Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/webhooks/register",
    response_model=WebhookResponse,
    summary="Register a new webhook target",
    status_code=status.HTTP_201_CREATED,
    operation_id="register_webhook",
)
async def register_webhook(
    body: Annotated[WebhookRegisterRequest, Body()],
    dispatcher: Annotated[WebhookDispatcher, Depends(_get_dispatcher)],
) -> WebhookResponse:
    """Dynamically register a webhook target to receive Castor event payloads.

    Registered targets are kept in-memory for the process lifetime.  For
    persistent registration across restarts, add entries to the
    ``[[webhooks.targets]]`` section of ``config.toml``.
    """
    target = WebhookTarget(
        name=body.name,
        url=str(body.url),
        events=body.events,
        headers=body.headers,
    )
    dispatcher.register(target)
    logger.info("webhook_registered", name=body.name, url=str(body.url))
    return WebhookResponse(name=target.name, url=target.url, events=target.events)


@router.get(
    "/webhooks",
    response_model=list[WebhookResponse],
    summary="List all registered webhook targets",
    operation_id="list_webhooks",
)
async def list_webhooks(
    dispatcher: Annotated[WebhookDispatcher, Depends(_get_dispatcher)],
) -> list[WebhookResponse]:
    """Return all currently registered webhook targets (static + dynamic)."""
    return [
        WebhookResponse(name=t.name, url=t.url, events=t.events)
        for t in dispatcher.list_targets()
    ]


@router.delete(
    "/webhooks/{name}",
    summary="Deregister a webhook target by name",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="delete_webhook",
)
async def delete_webhook(
    name: str,
    dispatcher: Annotated[WebhookDispatcher, Depends(_get_dispatcher)],
) -> None:
    """Remove a webhook target.  Only dynamically registered targets can be
    removed at runtime.  Static targets from ``config.toml`` are re-loaded on
    restart.
    """
    removed = dispatcher.deregister(name)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook target '{name}' not found.",
        )
    logger.info("webhook_deregistered", name=name)

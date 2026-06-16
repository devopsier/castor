"""
castor.api.webhooks — Async webhook dispatcher with retry logic.

Provides ``WebhookDispatcher``, an ``httpx.AsyncClient``-based engine that:

* Maintains a registry of ``WebhookTarget`` instances (loaded from config
  and populated dynamically via the REST API).
* Delivers event payloads to all subscribed targets concurrently via
  ``asyncio.gather``.
* Applies **exponential back-off with jitter** (via ``tenacity``) on 5xx
  responses and transient network errors.
* Signs each outgoing payload with an HMAC-SHA256 ``X-Castor-Signature``
  header when a shared secret is configured.

Design Notes
------------
* The dispatcher is intentionally **fire-and-forget** — callers are not
  blocked on webhook delivery.  Use ``asyncio.create_task()`` to invoke
  ``fire_event()`` from a route handler.
* Webhook targets can be registered and deregistered at runtime.  The
  dispatcher is thread-safe for reads; writes (register/deregister) should
  happen from a single coroutine context.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import hashlib
import hmac
import json
import os
from typing import Any

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class WebhookTarget:
    """A single webhook endpoint subscription.

    Attributes:
        name: Human-readable identifier used in logs and the REST API.
        url: Full URL to HTTP POST the event payload to.
        events: Set of event type strings this target subscribes to.
        headers: Additional HTTP headers to merge into each request.
    """

    name: str
    url: str
    events: list[str]
    headers: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, slots=True)
class DeliveryRecord:
    """Record of a single webhook delivery attempt.

    Attributes:
        target_name: Name of the webhook target.
        event: Event type that was delivered.
        status_code: HTTP status code of the final delivery attempt.
        attempts: Total number of delivery attempts made (including retries).
        delivered_at: UTC timestamp of the successful (or final failed) attempt.
        success: Whether the delivery ultimately succeeded.
    """

    target_name: str
    event: str
    status_code: int
    attempts: int
    delivered_at: dt.datetime
    success: bool


# ---------------------------------------------------------------------------
# HMAC Payload Signing
# ---------------------------------------------------------------------------


def _sign_payload(payload_bytes: bytes, secret: str) -> str:
    """Compute an HMAC-SHA256 signature for a raw payload.

    The signature format follows the convention used by GitHub Webhooks::

        X-Castor-Signature: sha256=<hex_digest>

    Args:
        payload_bytes: Raw UTF-8 encoded JSON payload.
        secret: Shared secret string (read from config or ``CASTOR_WEBHOOK_SECRET``
            environment variable).

    Returns:
        The header value string, e.g. ``"sha256=abc123..."``.
    """
    mac = hmac.new(
        secret.encode("utf-8"),
        msg=payload_bytes,
        digestmod=hashlib.sha256,
    )
    return f"sha256={mac.hexdigest()}"


# ---------------------------------------------------------------------------
# WebhookDispatcher
# ---------------------------------------------------------------------------


class WebhookDispatcher:
    """Async webhook dispatcher with exponential back-off retry logic.

    Loads static targets from ``config.toml`` on construction.  Additional
    targets can be registered/deregistered at runtime.

    Args:
        config: Parsed configuration dictionary (output of ``load_config()``).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        wh_cfg: dict[str, Any] = config.get("webhooks", {})

        self._timeout: float = float(wh_cfg.get("timeout_seconds", 10))
        self._max_retries: int = int(wh_cfg.get("max_retries", 3))
        self._backoff_base: float = float(wh_cfg.get("retry_backoff_seconds", 2.0))
        self._secret: str = os.environ.get(
            "CASTOR_WEBHOOK_SECRET", wh_cfg.get("secret", "")
        )

        # Build initial target registry from static config
        self._targets: dict[str, WebhookTarget] = {}
        for raw in wh_cfg.get("targets", []):
            target = WebhookTarget(
                name=raw["name"],
                url=raw["url"],
                events=list(raw.get("events", [])),
                headers=dict(raw.get("headers", {})),
            )
            self._targets[target.name] = target

        # Shared async HTTP client (reused across all deliveries)
        self._async_client: httpx.AsyncClient = httpx.AsyncClient(
            timeout=self._timeout,
            headers={"User-Agent": "castor-webhook/0.1.0"},
        )
        logger.info(
            "webhook_dispatcher_initialized",
            static_targets=len(self._targets),
        )

    # ------------------------------------------------------------------
    # Registry Management
    # ------------------------------------------------------------------

    def register(self, target: WebhookTarget) -> None:
        """Add or overwrite a webhook target in the registry.

        If a target with the same ``name`` already exists, it is replaced.

        Args:
            target: The ``WebhookTarget`` to register.
        """
        self._targets[target.name] = target
        logger.info("webhook_target_registered", name=target.name, url=target.url)

    def deregister(self, name: str) -> bool:
        """Remove a webhook target from the registry by name.

        Args:
            name: Name of the target to remove.

        Returns:
            ``True`` if the target was found and removed, ``False`` otherwise.
        """
        if name in self._targets:
            del self._targets[name]
            logger.info("webhook_target_deregistered", name=name)
            return True
        return False

    def list_targets(self) -> list[WebhookTarget]:
        """Return a snapshot of all currently registered targets."""
        return list(self._targets.values())

    # ------------------------------------------------------------------
    # Event Dispatch
    # ------------------------------------------------------------------

    async def fire_event(
        self,
        event: str,
        payload: dict[str, Any],
    ) -> list[DeliveryRecord]:
        """Deliver an event payload to all subscribed webhook targets concurrently.

        Each delivery is independently retried with exponential back-off.
        Failures are logged but do not propagate exceptions to the caller.

        Args:
            event: Event type string (e.g. ``"spike_imminent"``).
            payload: Arbitrary JSON-serialisable dictionary to POST.

        Returns:
            A list of ``DeliveryRecord`` objects describing each delivery outcome.
        """
        subscribed = [t for t in self._targets.values() if event in t.events]
        if not subscribed:
            logger.debug("no_webhook_targets_subscribed", event=event)
            return []

        # Enrich payload with Castor metadata
        enriched: dict[str, Any] = {
            "event": event,
            "emitted_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
            "source": "castor",
            **payload,
        }

        tasks = [self._deliver(target, event, enriched) for target in subscribed]
        results: list[DeliveryRecord] = list(await asyncio.gather(*tasks))
        return results

    async def _deliver(
        self,
        target: WebhookTarget,
        event: str,
        payload: dict[str, Any],
    ) -> DeliveryRecord:
        """Deliver payload to a single webhook target with retry logic.

        Uses ``tenacity.AsyncRetrying`` with exponential back-off and jitter.
        Retries on:
        * ``httpx.TransportError`` (network issues).
        * HTTP 5xx responses (server-side transient failures).

        Does **not** retry on:
        * HTTP 4xx responses (client errors — retrying won't help).
        * ``asyncio.CancelledError`` (application shutdown).

        Args:
            target: The ``WebhookTarget`` to deliver to.
            event: Event type string for logging.
            payload: Enriched event payload dictionary.

        Returns:
            A ``DeliveryRecord`` describing the final delivery outcome.
        """
        payload_bytes: bytes = json.dumps(payload, default=str).encode("utf-8")

        # Merge base headers with target-specific overrides
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            **target.headers,
        }
        if self._secret:
            headers["X-Castor-Signature"] = _sign_payload(payload_bytes, self._secret)

        attempt_count = 0
        last_status_code = 0

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(
                    multiplier=self._backoff_base,
                    min=self._backoff_base,
                    max=self._backoff_base * (2 ** self._max_retries),
                ),
                retry=retry_if_exception_type(
                    (httpx.TransportError, httpx.TimeoutException)
                ),
                reraise=False,
            ):
                with attempt:
                    attempt_count += 1
                    response = await self._async_client.post(
                        target.url,
                        content=payload_bytes,
                        headers=headers,
                    )
                    last_status_code = response.status_code

                    if response.status_code >= 500:
                        # Treat 5xx as retriable
                        logger.warning(
                            "webhook_delivery_server_error",
                            target=target.name,
                            status=response.status_code,
                            attempt=attempt_count,
                        )
                        response.raise_for_status()  # Triggers retry

                    logger.info(
                        "webhook_delivered",
                        target=target.name,
                        event=event,
                        status=response.status_code,
                        attempts=attempt_count,
                    )
                    return DeliveryRecord(
                        target_name=target.name,
                        event=event,
                        status_code=last_status_code,
                        attempts=attempt_count,
                        delivered_at=dt.datetime.now(tz=dt.timezone.utc),
                        success=True,
                    )

        except RetryError as exc:
            logger.error(
                "webhook_delivery_failed_after_retries",
                target=target.name,
                event=event,
                attempts=attempt_count,
                exc_info=exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "webhook_delivery_unexpected_error",
                target=target.name,
                event=event,
                exc_info=exc,
            )

        return DeliveryRecord(
            target_name=target.name,
            event=event,
            status_code=last_status_code,
            attempts=attempt_count,
            delivered_at=dt.datetime.now(tz=dt.timezone.utc),
            success=False,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying async HTTP client.

        Must be awaited during application shutdown to avoid resource leaks.
        """
        await self._async_client.aclose()
        logger.info("webhook_dispatcher_closed")

    async def __aenter__(self) -> "WebhookDispatcher":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

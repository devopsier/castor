"""
castor.main — Application entry point.

Bootstraps the FastAPI application, configures structured logging, registers
all API routers, and starts the Uvicorn ASGI server. A background scheduler
drives periodic metric ingestion and model re-training cycles.
"""

from __future__ import annotations

import asyncio
import os
import tomllib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from castor import __version__
from castor.api.routes import router as api_router
from castor.data.ingestor import PrometheusIngestor
from castor.models.predict import SpikePredictor

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------------------------

_CONFIG_PATH_ENV = "CASTOR_CONFIG_PATH"
_DEFAULT_CONFIG = Path(__file__).parent.parent.parent.parent / "config.toml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load and parse ``config.toml`` using Python's native ``tomllib``.

    Resolution order:
    1. ``path`` argument (if supplied).
    2. ``CASTOR_CONFIG_PATH`` environment variable.
    3. ``config.toml`` in the project root (development default).

    Args:
        path: Optional explicit path to the TOML configuration file.

    Returns:
        A nested dictionary representing the parsed configuration.

    Raises:
        FileNotFoundError: If the resolved config file does not exist.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
    """
    resolved = path or Path(os.environ.get(_CONFIG_PATH_ENV, str(_DEFAULT_CONFIG)))
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with resolved.open("rb") as fh:
        cfg: dict[str, Any] = tomllib.load(fh)
    logger.info("configuration_loaded", path=str(resolved))
    return cfg


# ---------------------------------------------------------------------------
# Application State — shared across request handlers
# ---------------------------------------------------------------------------

class AppState:
    """Mutable application-wide state injected into the FastAPI ``app.state``."""

    config: dict[str, Any]
    ingestor: PrometheusIngestor
    predictor: SpikePredictor


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and graceful shutdown of background services."""
    cfg = load_config()
    state: AppState = AppState()
    state.config = cfg
    state.ingestor = PrometheusIngestor(config=cfg)
    state.predictor = SpikePredictor(config=cfg)

    app.state.castor = state

    logger.info(
        "castor_starting",
        version=__version__,
        environment=cfg.get("castor", {}).get("environment", "unknown"),
    )

    # Start background periodic tasks
    ingest_task = asyncio.create_task(_periodic_ingest(state), name="periodic_ingest")

    try:
        yield  # Application is live
    finally:
        ingest_task.cancel()
        try:
            await ingest_task
        except asyncio.CancelledError:
            pass
        logger.info("castor_shutdown_complete")


async def _periodic_ingest(state: AppState) -> None:
    """Background coroutine: pull fresh metrics on a fixed interval."""
    interval: int = state.config.get("scheduler", {}).get("ingest_interval_seconds", 60)
    while True:
        try:
            await asyncio.sleep(interval)
            logger.debug("periodic_ingest_tick")
            # Ingestion is I/O-bound; run in the default thread executor
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, state.ingestor.fetch_all)
        except asyncio.CancelledError:
            break
        except Exception as exc:  # noqa: BLE001
            logger.error("periodic_ingest_error", exc_info=exc)


# ---------------------------------------------------------------------------
# FastAPI Application Factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Construct and configure the FastAPI application instance."""
    app = FastAPI(
        title="Castor",
        description=(
            "Predictive Pod/Cluster Steering — ML-powered traffic spike "
            "forecasting for Kubernetes autoscalers."
        ),
        version=__version__,
        license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS — tighten in production via config
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Routes
    app.include_router(api_router, prefix="/v1")

    # Health probe (no auth, used by Kubernetes liveness / readiness probes)
    @app.get("/healthz", include_in_schema=False, tags=["Health"])
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok", "version": __version__})

    @app.get("/readyz", include_in_schema=False, tags=["Health"])
    async def readyz() -> JSONResponse:
        return JSONResponse({"status": "ready"})

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

app: FastAPI = create_app()


def run() -> None:
    """CLI entrypoint — called by ``castor`` script defined in pyproject.toml."""
    cfg = load_config()
    server_cfg = cfg.get("server", {})
    uvicorn.run(
        "castor.main:app",
        host=server_cfg.get("host", "0.0.0.0"),
        port=server_cfg.get("port", 8080),
        workers=server_cfg.get("workers", 1),
        reload=server_cfg.get("reload", False),
        log_level=cfg.get("castor", {}).get("log_level", "info").lower(),
    )


if __name__ == "__main__":
    run()

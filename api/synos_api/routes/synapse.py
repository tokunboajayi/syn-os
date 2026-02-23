"""
Synapse API Routes — System Intelligence Endpoints.

Exposes the HardwareFingerprinter and ExperienceReplay buffer to the
rest of the Syn OS API so the frontend can display health scores and
trigger manual training.
"""

import logging
import time
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/synapse", tags=["synapse"])


# ── Pydantic Schemas ─────────────────────────────────────────────────────────

class MetricsPayload(BaseModel):
    cpu_percent: float = Field(0.0, ge=0, le=100)
    memory_percent: float = Field(0.0, ge=0, le=100)
    disk_io_read_mb: float = Field(0.0, ge=0)
    disk_io_write_mb: float = Field(0.0, ge=0)
    net_bytes_in_mb: float = Field(0.0, ge=0)
    net_bytes_out_mb: float = Field(0.0, ge=0)


class HealthResponse(BaseModel):
    health_score: float
    reconstruction_error: float
    is_anomalous: bool
    latent_vector: list
    buffer_stats: dict


class TrainRequest(BaseModel):
    epochs: int = Field(5, ge=1, le=50)
    batch_size: int = Field(64, ge=8, le=512)


class TrainResponse(BaseModel):
    status: str
    detail: dict


# ── Singletons (lazy init) ──────────────────────────────────────────────────

_fingerprinter = None
_replay = None


def _get_fingerprinter():
    global _fingerprinter
    if _fingerprinter is None:
        try:
            from synos_ml.models.fingerprinter import HardwareFingerprinter
            _fingerprinter = HardwareFingerprinter(model_dir="models")
        except ImportError:
            # Fallback: use the API-local stub
            from api.synos_api.core.mock_kernel import MockFingerprinter
            _fingerprinter = MockFingerprinter()
    return _fingerprinter


def _get_replay():
    global _replay
    if _replay is None:
        try:
            from synos_ml.core.replay_buffer import ExperienceReplay
            _replay = ExperienceReplay(persist_dir="data/replay")
        except ImportError:
            _replay = None
    return _replay


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=HealthResponse)
async def ingest_metrics(payload: MetricsPayload):
    """
    Ingest current system metrics.
    Returns: real-time health assessment from the fingerprinter.
    """
    fp = _get_fingerprinter()
    metrics = payload.model_dump()

    # 1. Fingerprint
    result = fp.fingerprint(metrics)

    # 2. Store in replay buffer
    replay = _get_replay()
    if replay:
        replay.push_metrics(
            state=metrics,
            action="observe",
            reward=result.health_score / 100.0,
        )

    return HealthResponse(
        health_score=result.health_score,
        reconstruction_error=result.reconstruction_error,
        is_anomalous=result.is_anomalous,
        latent_vector=result.latent_vector,
        buffer_stats=replay.stats() if replay else {},
    )


@router.get("/health-score")
async def get_health_score():
    """Quick health check — fingerprint the current system state."""
    try:
        import psutil
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io_read_mb": 0.0,
            "disk_io_write_mb": 0.0,
            "net_bytes_in_mb": 0.0,
            "net_bytes_out_mb": 0.0,
        }
    except ImportError:
        metrics = {
            "cpu_percent": 25.0,
            "memory_percent": 40.0,
            "disk_io_read_mb": 5.0,
            "disk_io_write_mb": 2.0,
            "net_bytes_in_mb": 1.0,
            "net_bytes_out_mb": 0.5,
        }

    fp = _get_fingerprinter()
    result = fp.fingerprint(metrics)
    return {
        "health_score": result.health_score,
        "is_anomalous": result.is_anomalous,
        "timestamp": time.time(),
    }


@router.get("/replay/stats")
async def replay_stats():
    """Return current replay buffer statistics."""
    replay = _get_replay()
    if not replay:
        raise HTTPException(status_code=503, detail="Replay buffer not available")
    return replay.stats()


@router.post("/train", response_model=TrainResponse)
async def trigger_training(req: TrainRequest, bg: BackgroundTasks):
    """Manually trigger a training run in the background."""
    try:
        from synos_ml.train_online import run_training
    except ImportError:
        raise HTTPException(status_code=503, detail="ML training module not available")

    def _do_train():
        result = run_training(epochs=req.epochs, batch_size=req.batch_size)
        logger.info(f"Background training result: {result}")

    bg.add_task(_do_train)
    return TrainResponse(
        status="started",
        detail={"epochs": req.epochs, "batch_size": req.batch_size},
    )

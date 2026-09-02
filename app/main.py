"""
FastAPI application — RM Remark Quality Inference Service.

Production-ready API for scoring Relationship Manager visit remarks on a
0–100 quality scale and mapping them to Trash / Average / Good labels.

Architecture:
  • Artifacts load once at startup via ArtifactManager singleton
  • All inference delegates to RemarkQualityEngine
  • No business logic in routes
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.middleware.security import (
    RateLimitMiddleware,
    sanitise_remark,
    verify_api_key,
)
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    StoryElements,
    VersionResponse,
)
from app.services.artifact_loader import get_artifact_manager
from app.services.inference_pipeline import score_batch, score_remark

# ═══════════════════════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rm_quality_api")


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics (in-memory, lightweight)
# ═══════════════════════════════════════════════════════════════════════════════

class _Metrics:
    def __init__(self) -> None:
        self.start_time = time.time()
        self.total_requests = 0
        self.total_predictions = 0
        self.total_errors = 0
        self._latency_sum = 0.0
        self._latency_count = 0

    def record_request(self, num_predictions: int, latency_s: float) -> None:
        self.total_requests += 1
        self.total_predictions += num_predictions
        self._latency_sum += latency_s
        self._latency_count += 1

    def record_error(self) -> None:
        self.total_requests += 1
        self.total_errors += 1

    @property
    def avg_latency_ms(self) -> float:
        if self._latency_count == 0:
            return 0.0
        return (self._latency_sum / self._latency_count) * 1000

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time


_metrics = _Metrics()


# ═══════════════════════════════════════════════════════════════════════════════
#  Application lifespan (startup / shutdown)
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all artifacts at startup; clean up on shutdown."""
    settings = get_settings()
    logger.info("Starting RM Remark Quality API …")
    logger.info("Artifact directory: %s", settings.artifact_dir)

    try:
        manager = get_artifact_manager()
        manager.load(settings.artifact_dir)
        logger.info("🚀 Service ready.")
    except Exception:
        logger.exception("❌ Failed to load artifacts — service cannot start.")
        raise

    yield  # Application runs here

    logger.info("Shutting down RM Remark Quality API.")


# ═══════════════════════════════════════════════════════════════════════════════
#  FastAPI app
# ═══════════════════════════════════════════════════════════════════════════════

settings = get_settings()

app = FastAPI(
    title="RM Remark Quality API",
    description=(
        "Production inference service for scoring Relationship Manager "
        "visit remarks on a 0–100 quality scale. Powered by a Stacked "
        "Ensemble model (LightGBM + XGBoost + CatBoost + RandomForest "
        "with Ridge meta-learner) over BAAI/bge-large-en-v1.5 sentence "
        "embeddings and 37 hand-crafted semantic features."
    ),
    version=settings.model_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)


# ═══════════════════════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════════════════════

# ── Health ────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Operations"],
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """Returns the health status of the service and whether artifacts are loaded."""
    manager = get_artifact_manager()
    return HealthResponse(
        status="healthy" if manager.is_loaded else "degraded",
        model_loaded=manager.is_loaded,
        artifact_count=manager.artifact_count,
    )


# ── Version ───────────────────────────────────────────────────────────────

@app.get(
    "/version",
    response_model=VersionResponse,
    tags=["Operations"],
    summary="Service version info",
)
async def version_info() -> VersionResponse:
    """Returns model version, architecture, and artifact metadata."""
    manager = get_artifact_manager()
    return VersionResponse(
        model_version=settings.model_version,
        model_architecture=manager.model_architecture or "unknown",
        artifact_version=settings.model_version,
        build_timestamp=settings.build_timestamp,
        feature_count=len(manager.feature_columns),
        pca_components=manager.pca.n_components_ if manager.pca else 0,
    )


# ── Metrics ───────────────────────────────────────────────────────────────

@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Operations"],
    summary="Request metrics",
)
async def get_metrics() -> MetricsResponse:
    """Returns cumulative request count, latency, and error stats."""
    return MetricsResponse(
        total_requests=_metrics.total_requests,
        total_predictions=_metrics.total_predictions,
        total_errors=_metrics.total_errors,
        avg_latency_ms=round(_metrics.avg_latency_ms, 2),
        uptime_seconds=round(_metrics.uptime_seconds, 1),
    )


# ── Single Prediction ────────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Inference"],
    summary="Score a single RM remark",
    dependencies=[Depends(verify_api_key)],
)
async def predict_single(request: PredictRequest) -> PredictResponse:
    """
    Score a single Relationship Manager visit remark.

    Returns a quality score (0-100), label (Trash/Average/Good),
    detected story elements, and human-readable explanation.
    """
    t0 = time.time()
    try:
        remark = sanitise_remark(request.remark)
        result = score_remark(remark)
        latency = time.time() - t0
        _metrics.record_request(num_predictions=1, latency_s=latency)

        logger.info(
            "Predicted score=%.1f label=%s latency=%.0fms",
            result["quality_score"],
            result["quality_label"],
            latency * 1000,
        )

        return PredictResponse(
            quality_score=result["quality_score"],
            quality_label=result["quality_label"],
            story_elements=StoryElements(**result["story_elements"]),
            strengths=result["strengths"],
            missing_elements=result["missing_elements"],
        )
    except Exception as e:
        _metrics.record_error()
        logger.exception("Prediction failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


# ── Batch Prediction ─────────────────────────────────────────────────────

@app.post(
    "/predict-batch",
    response_model=BatchPredictResponse,
    tags=["Inference"],
    summary="Score a batch of RM remarks",
    dependencies=[Depends(verify_api_key)],
)
async def predict_batch_endpoint(
    request: BatchPredictRequest,
) -> BatchPredictResponse:
    """
    Score multiple remarks in a single request.  Embeddings are generated
    in one batched call for efficiency.
    """
    if len(request.records) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(request.records)} exceeds maximum "
                   f"of {settings.max_batch_size}.",
        )

    t0 = time.time()
    try:
        remarks = [sanitise_remark(r.remark) for r in request.records]
        results = score_batch(remarks)
        latency = time.time() - t0
        _metrics.record_request(
            num_predictions=len(remarks), latency_s=latency,
        )

        logger.info(
            "Batch scored %d remarks in %.0fms",
            len(remarks), latency * 1000,
        )

        return BatchPredictResponse(
            total_records=len(results),
            results=[
                PredictResponse(
                    quality_score=r["quality_score"],
                    quality_label=r["quality_label"],
                    story_elements=StoryElements(**r["story_elements"]),
                    strengths=r["strengths"],
                    missing_elements=r["missing_elements"],
                )
                for r in results
            ],
        )
    except Exception as e:
        _metrics.record_error()
        logger.exception("Batch prediction failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


# ── Artifact Reload (hot-swap) ───────────────────────────────────────────

@app.post(
    "/admin/reload-artifacts",
    tags=["Admin"],
    summary="Hot-reload model artifacts",
    dependencies=[Depends(verify_api_key)],
)
async def reload_artifacts() -> Dict[str, Any]:
    """
    Force-reload all model artifacts from disk.  Use after deploying new
    artifact files without restarting the service.
    """
    try:
        manager = get_artifact_manager()
        manager.reload(settings.artifact_dir)
        return {
            "status": "reloaded",
            "artifact_count": manager.artifact_count,
        }
    except Exception as e:
        logger.exception("Artifact reload failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Artifact reload failed: {str(e)}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Entrypoint (for direct ``python -m app.main`` or ``python app/main.py``)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        workers=settings.workers,
    )

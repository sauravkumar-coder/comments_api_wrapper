"""
Pydantic v2 request / response schemas for the Remark Quality API.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
#  Request schemas
# ═══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    """Single-remark prediction request."""

    remark: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Raw RM visit remark text to score.",
        json_schema_extra={"examples": [
            "Visited Croma store and met SEC regarding Samsung display. "
            "Discussed sell-out performance and trained staff on new Galaxy lineup. "
            "SEC committed to improving attachment rate. Will follow up next week."
        ]},
    )


class BatchRecord(BaseModel):
    """A single record inside a batch request."""

    remark: str = Field(..., min_length=1, max_length=10_000)


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""

    records: List[BatchRecord] = Field(
        ...,
        min_length=1,
        description="List of remark records to score.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Response schemas
# ═══════════════════════════════════════════════════════════════════════════════

class StoryElements(BaseModel):
    """Boolean flags indicating which of the 7 story categories were detected."""

    visit_purpose: bool = False
    stakeholder: bool = False
    discussion: bool = False
    response: bool = False
    action: bool = False
    outcome: bool = False
    followup: bool = False


class PredictResponse(BaseModel):
    """Response for a single remark prediction."""

    quality_score: float = Field(
        ..., ge=0, le=100,
        description="Predicted quality score in [0, 100].",
    )
    quality_label: str = Field(
        ...,
        description="Mapped label: Trash (0-40), Average (41-70), or Good (71-100).",
    )
    story_elements: StoryElements = Field(
        ...,
        description="Which of the 7 business-story categories were detected.",
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Human-readable list of detected story strengths.",
    )
    missing_elements: List[str] = Field(
        default_factory=list,
        description="Human-readable list of missing story elements.",
    )


class BatchPredictResponse(BaseModel):
    """Response for a batch prediction request."""

    total_records: int
    results: List[PredictResponse]


# ═══════════════════════════════════════════════════════════════════════════════
#  Operational schemas
# ═══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = False
    artifact_count: int = 0


class VersionResponse(BaseModel):
    model_version: str
    model_architecture: str
    artifact_version: str
    build_timestamp: str
    feature_count: int
    pca_components: int


class MetricsResponse(BaseModel):
    total_requests: int = 0
    total_predictions: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    uptime_seconds: float = 0.0

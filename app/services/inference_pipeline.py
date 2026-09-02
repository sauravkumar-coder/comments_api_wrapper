"""
Inference pipeline orchestration — thin convenience layer over the
RemarkQualityEngine.

Provides module-level ``score_remark()`` and ``score_batch()`` functions
that the FastAPI routes call.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.services.predictor import RemarkQualityEngine

logger = logging.getLogger("rm_quality_api.inference_pipeline")

# Module-level engine instance (initialised once, reused across requests)
_engine: RemarkQualityEngine | None = None


def get_engine() -> RemarkQualityEngine:
    """Return (and lazily create) the module-level engine instance."""
    global _engine
    if _engine is None:
        _engine = RemarkQualityEngine()
    return _engine


def score_remark(remark: str) -> Dict[str, Any]:
    """
    Score a single remark through the full pipeline.

    Parameters
    ----------
    remark : str
        Raw RM visit remark text.

    Returns
    -------
    dict
        Prediction result with score, label, story elements, and explanation.
    """
    engine = get_engine()
    return engine.predict(remark)


def score_batch(remarks: List[str]) -> List[Dict[str, Any]]:
    """
    Score a batch of remarks efficiently.

    Parameters
    ----------
    remarks : list[str]
        Raw remark texts.

    Returns
    -------
    list[dict]
        One prediction result per remark.
    """
    engine = get_engine()
    return engine.predict_batch(remarks)

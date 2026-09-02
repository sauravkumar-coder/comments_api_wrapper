"""
RemarkQualityEngine — the centralised inference engine.

All API endpoints delegate to this engine.  No business logic exists in
FastAPI routes.

Pipeline (Section 5.1 of the technical report):
  Remark → clean → 37 semantic features → embedding (1024-d)
  → PCA (247-d) → concat (284 features) → reindex → predict → clip → label → explain
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from app.services.artifact_loader import ArtifactManager, get_artifact_manager
from app.utils.feature_engineering import (
    extract_all_semantic_features,
    generate_explanation,
    get_story_elements_flags,
)
from app.utils.preprocessing import clean_remark_text

logger = logging.getLogger("rm_quality_api.predictor")


class RemarkQualityEngine:
    """
    Stateless inference engine that uses the singleton ArtifactManager for
    all model artefacts.  Thread-safe for concurrent requests.
    """

    def __init__(self, artifact_manager: ArtifactManager | None = None) -> None:
        self._am = artifact_manager or get_artifact_manager()

    # ── Text preprocessing ────────────────────────────────────────────────

    def preprocess(self, remark: str) -> str:
        """Clean raw remark text."""
        return clean_remark_text(remark)

    # ── Feature generation ────────────────────────────────────────────────

    def generate_features(self, cleaned_text: str) -> Dict[str, Any]:
        """Extract all 37 semantic features from cleaned text."""
        return extract_all_semantic_features(cleaned_text)

    # ── Embedding + PCA ───────────────────────────────────────────────────

    def _embed_single(self, cleaned_text: str) -> np.ndarray:
        """
        Generate a 1024-dim sentence embedding and PCA-reduce to 247 dims.

        CRITICAL: ``normalize_embeddings=True`` must match training.
        """
        # 1024-dim embedding
        embedding = self._am.embedding_model.encode(
            [cleaned_text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]  # shape: (1024,)

        # PCA → 247-dim
        embedding_pca = self._am.pca.transform(
            embedding.reshape(1, -1)
        )[0]  # shape: (247,)

        return embedding_pca

    def _embed_batch(self, cleaned_texts: List[str]) -> np.ndarray:
        """
        Batch-embed and PCA-transform multiple texts at once.

        Returns shape (n, 247).
        """
        embeddings = self._am.embedding_model.encode(
            cleaned_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )  # shape: (n, 1024)

        embeddings_pca = self._am.pca.transform(embeddings)  # shape: (n, 247)
        return embeddings_pca

    # ── Assemble feature row ──────────────────────────────────────────────

    def _build_feature_row(
        self,
        embedding_pca: np.ndarray,
        semantic_feats: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Concatenate PCA components + semantic features and reindex to the
        exact column order from feature_columns.pkl.
        """
        row: Dict[str, Any] = {
            f"pc_{i}": v for i, v in enumerate(embedding_pca)
        }
        row.update(semantic_feats)

        df = pd.DataFrame([row]).reindex(
            columns=self._am.feature_columns, fill_value=0
        )
        return df

    def _build_feature_matrix(
        self,
        embeddings_pca: np.ndarray,
        semantic_feats_list: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Build feature matrix for a batch of remarks."""
        rows = []
        for i, semantic_feats in enumerate(semantic_feats_list):
            row: Dict[str, Any] = {
                f"pc_{j}": v for j, v in enumerate(embeddings_pca[i])
            }
            row.update(semantic_feats)
            rows.append(row)

        df = pd.DataFrame(rows).reindex(
            columns=self._am.feature_columns, fill_value=0
        )
        return df

    # ── Score mapping ─────────────────────────────────────────────────────

    def _clip_score(self, raw_score: float) -> float:
        """Clip prediction to [0, 100]."""
        return max(0.0, min(100.0, raw_score))

    def _map_label(self, score: float) -> str:
        """Map a clipped score to Trash / Average / Good via label_mapping.json."""
        bins = self._am.label_mapping["bins"]     # [0, 40, 70, 100]
        labels = self._am.label_mapping["labels"]  # ["Trash", "Average", "Good"]

        # bins define edges: 0-40 → Trash, 41-70 → Average, 71-100 → Good
        for label, lo, hi in zip(labels, bins[:-1], bins[1:]):
            # First bin: 0 ≤ score ≤ 40 → Trash
            # Second:   41 ≤ score ≤ 70 → Average  (lo=40, so score > 40)
            # Third:    71 ≤ score ≤ 100 → Good     (lo=70, so score > 70)
            if lo == 0:
                if score <= hi:
                    return label
            else:
                if lo < score <= hi:
                    return label

        # Fallback (should not happen with clipped scores)
        return labels[-1]

    # ── Explainability ────────────────────────────────────────────────────

    def explain(
        self, semantic_feats: Dict[str, Any]
    ) -> tuple[List[str], List[str], Dict[str, bool]]:
        """
        Generate human-readable explanation from semantic features.

        Returns
        -------
        (strengths, missing_elements, story_element_flags)
        """
        strengths, missing = generate_explanation(semantic_feats)
        flags = get_story_elements_flags(semantic_feats)
        return strengths, missing, flags

    # ═══════════════════════════════════════════════════════════════════════
    #  Public predict methods
    # ═══════════════════════════════════════════════════════════════════════

    def predict(self, remark: str) -> Dict[str, Any]:
        """
        Score a single remark end-to-end.

        Parameters
        ----------
        remark : str
            Raw RM visit remark.

        Returns
        -------
        dict with keys: quality_score, quality_label, story_elements,
                        strengths, missing_elements
        """
        # 1. Text cleaning
        cleaned = self.preprocess(remark)

        # 2. Semantic feature extraction (37 features)
        semantic_feats = self.generate_features(cleaned)

        # 3. Embedding → PCA
        embedding_pca = self._embed_single(cleaned)

        # 4. Assemble feature row (284 columns, exact order)
        X = self._build_feature_row(embedding_pca, semantic_feats)

        # 5. Model prediction
        raw_score = float(self._am.model.predict(X)[0])

        # 6. Clip and map
        score = self._clip_score(raw_score)
        label = self._map_label(score)

        # 7. Explainability
        strengths, missing, flags = self.explain(semantic_feats)

        return {
            "quality_score": round(score, 1),
            "quality_label": label,
            "story_elements": flags,
            "strengths": strengths,
            "missing_elements": missing,
        }

    def predict_batch(self, remarks: List[str]) -> List[Dict[str, Any]]:
        """
        Score a batch of remarks efficiently.

        Embeddings are generated in a single batched call (not looped
        remark-by-remark), matching the notebook's ``score_dataframe()``
        approach from Section 13.
        """
        if not remarks:
            return []

        # 1. Clean all texts
        cleaned_texts = [self.preprocess(r) for r in remarks]

        # 2. Extract semantic features for all
        semantic_feats_list = [
            self.generate_features(ct) for ct in cleaned_texts
        ]

        # 3. Batch embed → PCA
        embeddings_pca = self._embed_batch(cleaned_texts)

        # 4. Build full feature matrix
        X = self._build_feature_matrix(embeddings_pca, semantic_feats_list)

        # 5. Batch predict
        raw_scores = self._am.model.predict(X)

        # 6. Assemble results
        results = []
        for i, raw_score in enumerate(raw_scores):
            score = self._clip_score(float(raw_score))
            label = self._map_label(score)
            strengths, missing, flags = self.explain(semantic_feats_list[i])
            results.append({
                "quality_score": round(score, 1),
                "quality_label": label,
                "story_elements": flags,
                "strengths": strengths,
                "missing_elements": missing,
            })

        return results

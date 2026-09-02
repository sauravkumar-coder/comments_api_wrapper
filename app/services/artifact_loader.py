"""
Singleton artifact manager — loads all model artifacts once at startup and
holds them in memory for concurrent request serving.

Artifacts loaded:
  1. best_model.pkl      → StackedEnsembleModel
  2. pca_model.pkl        → sklearn PCA (1024 → 247)
  3. feature_columns.pkl  → list[str] (284 column names)
  4. embedding_model_reference.txt → str (BAAI/bge-large-en-v1.5)
  5. label_mapping.json   → dict (bins, labels, rules)
  6. model_architecture.txt → str ("Stacked Ensemble")

All artifacts load via this manager — nothing is loaded per request.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

logger = logging.getLogger("rm_quality_api.artifact_loader")


class ArtifactManager:
    """
    Thread-safe singleton that loads and caches all inference artifacts.

    Usage::

        manager = ArtifactManager()
        manager.load("/path/to/artifacts")

        model = manager.model
        pca = manager.pca
    """

    _instance: Optional["ArtifactManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ArtifactManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._loaded = False
            return cls._instance

    def __init__(self) -> None:
        # Only set attributes on first init
        if not hasattr(self, "_initialised"):
            self._initialised = True
            self._loaded = False
            self.model: Any = None
            self.pca: Any = None
            self.feature_columns: List[str] = []
            self.embedding_model: Any = None
            self.embedding_model_name: str = ""
            self.label_mapping: Dict[str, Any] = {}
            self.model_architecture: str = ""

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def artifact_count(self) -> int:
        """Number of successfully loaded artifacts."""
        count = 0
        if self.model is not None:
            count += 1
        if self.pca is not None:
            count += 1
        if self.feature_columns:
            count += 1
        if self.embedding_model is not None:
            count += 1
        if self.label_mapping:
            count += 1
        if self.model_architecture:
            count += 1
        return count

    def load(self, artifact_dir: str) -> None:
        """
        Load all artifacts from disk.  Idempotent — subsequent calls are no-ops
        unless ``reload()`` is used.

        Parameters
        ----------
        artifact_dir : str
            Absolute path to the artifacts directory.

        Raises
        ------
        FileNotFoundError
            If a required artifact file is missing.
        RuntimeError
            If model loading fails (e.g. missing StackedEnsembleModel class).
        """
        if self._loaded:
            logger.info("Artifacts already loaded — skipping.")
            return

        self._do_load(artifact_dir)

    def reload(self, artifact_dir: str) -> None:
        """
        Force-reload all artifacts (hot-swap support).
        """
        logger.warning("Hot-reloading artifacts from %s", artifact_dir)
        self._loaded = False
        self._do_load(artifact_dir)

    def _do_load(self, artifact_dir: str) -> None:
        art = Path(artifact_dir)
        if not art.is_dir():
            raise FileNotFoundError(f"Artifact directory not found: {art}")

        # ── 1. StackedEnsembleModel class must be importable ─────────────
        # Import the module that patches __main__ — must happen before joblib
        import app.stacked_ensemble  # noqa: F401

        # ── 2. Load best_model.pkl ───────────────────────────────────────
        model_path = art / "best_model.pkl"
        logger.info("Loading model from %s …", model_path)
        self.model = joblib.load(model_path)
        logger.info("Model loaded: %s", type(self.model).__name__)

        # ── 3. Load PCA ──────────────────────────────────────────────────
        pca_path = art / "pca_model.pkl"
        logger.info("Loading PCA from %s …", pca_path)
        self.pca = joblib.load(pca_path)
        logger.info(
            "PCA loaded: %d components, input dim %d",
            self.pca.n_components_, self.pca.n_features_in_,
        )

        # ── 4. Load feature columns ─────────────────────────────────────
        cols_path = art / "feature_columns.pkl"
        self.feature_columns = joblib.load(cols_path)
        logger.info("Feature columns loaded: %d columns", len(self.feature_columns))

        # ── 5. Load embedding model reference ────────────────────────────
        ref_path = art / "embedding_model_reference.txt"
        self.embedding_model_name = ref_path.read_text(encoding="utf-8").strip()
        logger.info("Embedding model name: %s", self.embedding_model_name)
        
        # We no longer load the model locally via sentence_transformers.
        # It will be called remotely via Hugging Face Inference API.
        self.embedding_model = None

        # ── 6. Load label mapping ────────────────────────────────────────
        mapping_path = art / "label_mapping.json"
        with open(mapping_path, encoding="utf-8") as f:
            self.label_mapping = json.load(f)
        logger.info("Label mapping loaded: %s", self.label_mapping.get("labels"))

        # ── 7. Load model architecture ───────────────────────────────────
        arch_path = art / "model_architecture.txt"
        self.model_architecture = arch_path.read_text(encoding="utf-8").strip()
        logger.info("Architecture: %s", self.model_architecture)

        self._loaded = True
        logger.info(
            "✅ All %d artifacts loaded successfully.", self.artifact_count,
        )


def get_artifact_manager() -> ArtifactManager:
    """Return the global singleton artifact manager."""
    return ArtifactManager()

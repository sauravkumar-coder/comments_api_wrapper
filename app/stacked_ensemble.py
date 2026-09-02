"""
StackedEnsembleModel — the custom model class required to deserialise best_model.pkl.

This class MUST be importable before ``joblib.load("best_model.pkl")`` is called,
otherwise Python will raise ``AttributeError: Can't get attribute 'StackedEnsembleModel'``.

The model was serialised from ``__main__`` in the training notebook, so we
monkey-patch this class into ``__main__`` at import time (see bottom of file).
"""

from __future__ import annotations

import sys
from typing import Any, Dict

import numpy as np
import pandas as pd


class StackedEnsembleModel:
    """
    A stacking ensemble that blends four base regressors through a Ridge
    meta-learner.

    Attributes
    ----------
    base_models : dict[str, Any]
        Mapping of name → fitted base regressor.
        Expected keys: ``LightGBM``, ``XGBoost``, ``CatBoost``, ``RandomForest``.
    meta_model : sklearn.linear_model.Ridge
        Fitted Ridge regressor trained on out-of-fold base-model predictions.
    """

    def __init__(
        self,
        base_models: Dict[str, Any] | None = None,
        meta_model: Any | None = None,
    ) -> None:
        self.base_models = base_models or {}
        self.meta_model = meta_model

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Generate predictions by stacking base model outputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        # Collect predictions from each base model → (n_samples, n_base_models)
        base_preds = np.column_stack(
            [model.predict(X) for model in self.base_models.values()]
        )
        # Meta-learner blends them into the final prediction
        return self.meta_model.predict(base_preds)


# ── Monkey-patch into __main__ so joblib.load can resolve the class ──────────
# The pickle was written from a Jupyter notebook where the class lived in
# ``__main__``.  When we load it in a different process the unpickler looks
# for ``__main__.StackedEnsembleModel``.  Injecting it here keeps the rest
# of the codebase clean.
sys.modules["__main__"].StackedEnsembleModel = StackedEnsembleModel  # type: ignore[attr-defined]

"""LightGBM gradient-boosted-trees classifier (the default signal model)."""
from __future__ import annotations

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .base import BaseModel


class LightGBMModel(BaseModel):
    def __init__(self, params: dict | None = None, threshold: float = 0.40):
        self.params = dict(params or {})
        # Let the sklearn wrapper infer objective + class count from y. Explicit
        # objective='multiclass'/num_class breaks folds that have only 2 classes.
        self.params.pop("num_class", None)
        self.params.pop("objective", None)
        self.threshold = threshold
        self.model: lgb.LGBMClassifier | None = None
        self.classes_: list[int] | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> "LightGBMModel":
        self.feature_names_ = list(X.columns)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, np.asarray(y).astype(int), sample_weight=sample_weight)
        self.classes_ = [int(c) for c in self.model.classes_]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        if self.feature_names_:
            # reindex to training columns; missing features -> NaN (LightGBM-safe)
            X = X.reindex(columns=self.feature_names_)
        return self.model.predict_proba(X)

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            return pd.Series(dtype=float)
        return pd.Series(
            self.model.feature_importances_, index=self.feature_names_
        ).sort_values(ascending=False)

    def save(self, path) -> None:
        joblib.dump(
            {
                "params": self.params,
                "threshold": self.threshold,
                "model": self.model,
                "classes_": self.classes_,
                "feature_names_": self.feature_names_,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "LightGBMModel":
        d = joblib.load(path)
        obj = cls(params=d["params"], threshold=d["threshold"])
        obj.model = d["model"]
        obj.classes_ = d["classes_"]
        obj.feature_names_ = d["feature_names_"]
        return obj

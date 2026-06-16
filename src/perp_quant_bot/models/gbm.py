"""LightGBM signal model: seed-ensemble + leak-safe probability calibration.

* Seed-ensemble: train ``n_seeds`` models with different seeds and average their
  probabilities -> lower variance, less overfit to one random path.
* Calibration: optionally fit an isotonic calibrator on a CHRONOLOGICAL holdout
  (last slice of the training fold), so ``prob_threshold`` means what it says.
  The calibration data is strictly before the test set, so there is no leakage.
"""
from __future__ import annotations

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from .base import BaseModel

try:  # sklearn >= 1.6 prefers FrozenEstimator over the deprecated cv="prefit"
    from sklearn.frozen import FrozenEstimator

    _HAS_FROZEN = True
except Exception:  # noqa: BLE001
    _HAS_FROZEN = False


class LightGBMModel(BaseModel):
    def __init__(
        self,
        params: dict | None = None,
        threshold: float = 0.40,
        n_seeds: int = 1,
        calibrate: bool = False,
        base_seed: int = 42,
    ):
        self.params = dict(params or {})
        self.params.pop("num_class", None)
        self.params.pop("objective", None)
        self.threshold = threshold
        self.n_seeds = max(1, int(n_seeds))
        self.calibrate = bool(calibrate)
        self.base_seed = base_seed
        self.estimators_: list = []
        self.classes_: list[int] | None = None
        self.feature_names_: list[str] | None = None

    def _fit_one(self, X: pd.DataFrame, y: pd.Series, sw, seed: int):
        params = dict(self.params)
        params["random_state"] = seed
        clf = lgb.LGBMClassifier(**params)

        all_classes = set(np.unique(y))
        if self.calibrate and len(X) >= 500:
            cut = int(len(X) * 0.8)
            Xb, yb = X.iloc[:cut], y.iloc[:cut]
            Xc, yc = X.iloc[cut:], y.iloc[cut:]
            swb = sw[:cut] if sw is not None else None
            # only calibrate if both slices cover all classes (else isotonic breaks)
            if set(np.unique(yb)) == all_classes and set(np.unique(yc)) == all_classes:
                clf.fit(Xb, yb, sample_weight=swb)
                if _HAS_FROZEN:
                    calibrated = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
                else:
                    calibrated = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
                calibrated.fit(Xc, yc)
                return calibrated
        clf.fit(X, y, sample_weight=sw)
        return clf

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> "LightGBMModel":
        self.feature_names_ = list(X.columns)
        y_arr = np.asarray(y).astype(int)
        y_ser = pd.Series(y_arr, index=X.index)
        sw = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None
        self.classes_ = sorted(int(c) for c in np.unique(y_arr))
        self.estimators_ = [
            self._fit_one(X, y_ser, sw, self.base_seed + k * 101) for k in range(self.n_seeds)
        ]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Model is not fitted")
        if self.feature_names_:
            X = X.reindex(columns=self.feature_names_)
        cls_index = {c: i for i, c in enumerate(self.classes_)}
        agg = np.zeros((len(X), len(self.classes_)), dtype=float)
        for est in self.estimators_:
            proba = est.predict_proba(X)
            for j, c in enumerate(int(c) for c in est.classes_):
                if c in cls_index:
                    agg[:, cls_index[c]] += proba[:, j]
        agg /= len(self.estimators_)
        return agg

    @staticmethod
    def _extract_importance(est):
        if hasattr(est, "feature_importances_"):
            return est.feature_importances_
        try:
            return est.calibrated_classifiers_[0].estimator.feature_importances_
        except Exception:  # noqa: BLE001
            return None

    def feature_importance(self) -> pd.Series:
        imps = [fi for est in self.estimators_ if (fi := self._extract_importance(est)) is not None]
        if not imps:
            return pd.Series(dtype=float)
        return pd.Series(np.mean(imps, axis=0), index=self.feature_names_).sort_values(ascending=False)

    def save(self, path) -> None:
        joblib.dump(
            {
                "params": self.params,
                "threshold": self.threshold,
                "n_seeds": self.n_seeds,
                "calibrate": self.calibrate,
                "base_seed": self.base_seed,
                "estimators_": self.estimators_,
                "classes_": self.classes_,
                "feature_names_": self.feature_names_,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "LightGBMModel":
        d = joblib.load(path)
        obj = cls(
            params=d["params"],
            threshold=d["threshold"],
            n_seeds=d.get("n_seeds", 1),
            calibrate=d.get("calibrate", False),
            base_seed=d.get("base_seed", 42),
        )
        obj.estimators_ = d["estimators_"]
        obj.classes_ = d["classes_"]
        obj.feature_names_ = d["feature_names_"]
        return obj

"""Model interface. Implementations predict 3-class probabilities for {-1, 0, +1}."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    classes_: list[int] | None = None
    threshold: float = 0.40

    @abstractmethod
    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> "BaseModel": ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return array (n_samples, n_classes) ordered by ``self.classes_``."""

    @abstractmethod
    def save(self, path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path) -> "BaseModel": ...

    def predict_signal(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Map probabilities to a trade signal in {-1, 0, +1}.

        A non-neutral class is only emitted when its probability is the argmax
        AND exceeds *threshold*; otherwise the signal is 0 (stay flat).
        """
        thr = self.threshold if threshold is None else threshold
        proba = self.predict_proba(X)
        classes = np.asarray(self.classes_)
        idx = proba.argmax(axis=1)
        chosen = classes[idx]
        maxp = proba.max(axis=1)
        return np.where((chosen != 0) & (maxp >= thr), chosen, 0).astype(int)

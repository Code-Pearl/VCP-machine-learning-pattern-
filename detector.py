"""
Ties features.py + som.py together into a usable VCP detector.

Pipeline:
  1. Slide a window across every series in the training universe, extract
     shape-feature vectors (features.py).
  2. Standardize those vectors (z-score) and train a SOM on them
     UNSUPERVISED - the map organizes itself purely from vector similarity.
  3. Label each neuron using the KNOWN synthetic ground truth: a neuron's
     "vcp_affinity" is the fraction of its assigned training windows that
     came from a true VCP series' pivot region. This is the only place
     labels are used - training itself never sees them.
  4. To score a new window: extract its feature vector, standardize with
     the SAME scaler fit in step 2, find its BMU, and blend:
         match_score = 0.5 * rule_score + 0.5 * neuron_vcp_affinity
     rule_score keeps the result interpretable and grounded in named VCP
     criteria; neuron_vcp_affinity is what the self-organizing map
     contributes - shape similarity to known examples that the rule score's
     fixed weights might not fully capture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from features import extract_window_features, FEATURE_NAMES
from som import SimpleSOM


@dataclass
class TrainingSample:
    symbol: str
    end_idx: int
    vector: np.ndarray
    rule_score: float
    is_near_true_vcp_pivot: bool  # ground truth, used only for neuron labeling


@dataclass
class VCPDetector:
    grid_x: int = 8
    grid_y: int = 8
    lookback: int = 160
    scan_step: int = 5
    feature_mean_: np.ndarray = field(default=None)
    feature_std_: np.ndarray = field(default=None)
    som: SimpleSOM = field(default=None)
    neuron_affinity_: np.ndarray = field(default=None)   # (gx, gy) -> P(near true VCP pivot)
    neuron_counts_: np.ndarray = field(default=None)

    def _standardize(self, vec: np.ndarray) -> np.ndarray:
        return (vec - self.feature_mean_) / np.maximum(self.feature_std_, 1e-9)

    def collect_training_samples(self, data: dict[str, pd.DataFrame], meta: dict,
                                  pivot_tolerance_days: int = 15) -> list[TrainingSample]:
        samples = []
        for sym, df in data.items():
            m = meta[sym]
            n = len(df)
            true_pivot_idx = m.base_end_idx if getattr(m, "is_vcp", False) else None
            for end_idx in range(self.lookback, n, self.scan_step):
                feats = extract_window_features(df, end_idx, lookback=self.lookback)
                if feats is None:
                    continue
                near_true = (
                    true_pivot_idx is not None
                    and abs(end_idx - true_pivot_idx) <= pivot_tolerance_days
                )
                samples.append(TrainingSample(
                    symbol=sym, end_idx=end_idx, vector=feats.vector,
                    rule_score=feats.rule_score, is_near_true_vcp_pivot=near_true,
                ))
        return samples

    def fit(self, data: dict[str, pd.DataFrame], meta: dict,
            n_epochs: int = 120, seed: int = 0, pivot_tolerance_days: int = 15) -> "VCPDetector":
        samples = self.collect_training_samples(data, meta, pivot_tolerance_days)
        if not samples:
            raise ValueError("No training samples collected - check lookback vs. series length.")

        X = np.stack([s.vector for s in samples])
        self.feature_mean_ = X.mean(axis=0)
        self.feature_std_ = X.std(axis=0)
        Xs = (X - self.feature_mean_) / np.maximum(self.feature_std_, 1e-9)

        self.som = SimpleSOM(self.grid_x, self.grid_y, input_dim=X.shape[1], seed=seed)
        self.som.train(Xs, n_epochs=n_epochs, seed=seed)

        affinity_sum = np.zeros((self.grid_x, self.grid_y))
        counts = np.zeros((self.grid_x, self.grid_y))
        for s, xs in zip(samples, Xs):
            idx = self.som.bmu_index(xs)
            counts[idx] += 1
            if s.is_near_true_vcp_pivot:
                affinity_sum[idx] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            self.neuron_affinity_ = np.where(counts > 0, affinity_sum / np.maximum(counts, 1), 0.0)
        self.neuron_counts_ = counts
        self._fit_samples = samples
        self._fit_Xs = Xs
        return self

    def score_window(self, df: pd.DataFrame, end_idx: int) -> dict | None:
        feats = extract_window_features(df, end_idx, lookback=self.lookback)
        if feats is None:
            return None
        xs = self._standardize(feats.vector)
        bmu = self.som.bmu_index(xs)
        neuron_affinity = float(self.neuron_affinity_[bmu])
        match_score = 0.5 * feats.rule_score + 0.5 * neuron_affinity
        return {
            "end_idx": end_idx,
            "rule_score": feats.rule_score,
            "neuron_affinity": neuron_affinity,
            "match_score": match_score,
            "bmu": bmu,
            "sub_scores": feats.sub_scores,
            "pivot_price": feats.pivot_price,
            "n_legs_found": feats.n_legs_found,
        }

    def scan_series(self, df: pd.DataFrame, step: int | None = None) -> list[dict]:
        step = step or self.scan_step
        out = []
        for end_idx in range(self.lookback, len(df), step):
            r = self.score_window(df, end_idx)
            if r is not None:
                out.append(r)
        return out

    def top_matches(self, df: pd.DataFrame, top_k: int = 3, step: int | None = None) -> list[dict]:
        results = self.scan_series(df, step=step)
        results.sort(key=lambda r: r["match_score"], reverse=True)
        # de-duplicate overlapping windows (within lookback/4 days of a stronger hit)
        picked, out = [], []
        for r in results:
            if all(abs(r["end_idx"] - p) > self.lookback // 4 for p in picked):
                picked.append(r["end_idx"])
                out.append(r)
            if len(out) >= top_k:
                break
        return out

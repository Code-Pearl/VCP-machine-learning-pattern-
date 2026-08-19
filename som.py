"""
A small, dependency-free 2D Kohonen Self-Organizing Map (SOM).

This is the "self-organizing" part of the pattern recognizer: it is trained
UNSUPERVISED on shape-feature vectors (see features.py) and learns a 2D
topology over that feature space purely from the structure of the data - it
never sees VCP/non-VCP labels during training.

Labels only enter afterward, to answer "which neurons tend to light up for
known VCP examples" - that's a read-out step on top of an otherwise
unsupervised map, not part of the SOM's training objective.
"""
from __future__ import annotations

import numpy as np


class SimpleSOM:
    def __init__(self, grid_x: int, grid_y: int, input_dim: int,
                 sigma0: float | None = None, lr0: float = 0.5, seed: int = 0):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.input_dim = input_dim
        rng = np.random.default_rng(seed)
        # small random init; weights live in the same (normalized) feature space
        self.weights = rng.normal(0, 0.15, size=(grid_x, grid_y, input_dim))
        xs, ys = np.meshgrid(np.arange(grid_x), np.arange(grid_y), indexing="ij")
        self._coords = np.stack([xs, ys], axis=-1).astype(float)  # (gx, gy, 2)
        self.sigma0 = sigma0 if sigma0 is not None else max(grid_x, grid_y) / 2.0
        self.lr0 = lr0
        self._trained_n_samples = 0

    def _bmu_grid(self, x: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
        """Best matching unit for a single sample x. Returns (idx, sq_dist_grid)."""
        d2 = np.sum((self.weights - x) ** 2, axis=-1)
        idx = np.unravel_index(np.argmin(d2), d2.shape)
        return idx, d2

    def bmu_index(self, x: np.ndarray) -> tuple[int, int]:
        idx, _ = self._bmu_grid(np.asarray(x, dtype=float))
        return idx

    def train(self, X: np.ndarray, n_epochs: int = 150, seed: int = 0) -> "SimpleSOM":
        X = np.asarray(X, dtype=float)
        n = len(X)
        rng = np.random.default_rng(seed)
        for epoch in range(n_epochs):
            t = epoch / max(1, n_epochs)
            sigma = max(0.35, self.sigma0 * np.exp(-t * 3.0))
            lr = max(0.01, self.lr0 * np.exp(-t * 3.0))
            order = rng.permutation(n)
            for i in order:
                x = X[i]
                bmu_idx, _ = self._bmu_grid(x)
                bmu_coord = np.array(bmu_idx, dtype=float)
                dist2_to_bmu = np.sum((self._coords - bmu_coord) ** 2, axis=-1)
                neigh = np.exp(-dist2_to_bmu / (2.0 * sigma ** 2))
                self.weights += lr * neigh[..., None] * (x - self.weights)
        self._trained_n_samples = n
        return self

    def quantization_error(self, X: np.ndarray) -> float:
        """Average distance from each sample to its BMU weight vector -
        a standard SOM fit diagnostic (lower = tighter fit)."""
        X = np.asarray(X, dtype=float)
        errs = []
        for x in X:
            idx, d2 = self._bmu_grid(x)
            errs.append(float(np.sqrt(d2[idx])))
        return float(np.mean(errs)) if errs else float("nan")

    def distance_map(self) -> np.ndarray:
        """U-matrix: for each neuron, mean weight-distance to its grid neighbors.
        Useful for visualizing cluster boundaries on the map."""
        U = np.zeros((self.grid_x, self.grid_y))
        for i in range(self.grid_x):
            for j in range(self.grid_y):
                dists = []
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.grid_x and 0 <= nj < self.grid_y:
                        dists.append(float(np.linalg.norm(self.weights[i, j] - self.weights[ni, nj])))
                U[i, j] = float(np.mean(dists)) if dists else 0.0
        return U

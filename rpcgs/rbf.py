from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


class RadialBasisFunctionNetwork:
    def __init__(self, dim: int, obj: int, kernel_size: int, optimizer: str = "sgd", epoch: int = 5, lr: float = 0.06, alpha: float = 1.0):
        self._dim = int(dim)
        self._obj = int(obj)
        self._kernel_size = int(kernel_size)
        self._optimizer = optimizer
        self._epoch = int(epoch)
        self._lr = float(lr)
        self._alpha = float(alpha)
        self._weight = None
        self._bias = None
        self._centers = None
        self._std = None

    def sync_auto(self, params: dict):
        self.__dict__.update(params)

    def sync_manuel(self, weight=None, bias=None, centers=None, std=None):
        if weight is not None:
            self._weight = np.asarray(weight, dtype=float)
        if bias is not None:
            self._bias = np.asarray(bias, dtype=float)
        if centers is not None:
            self._centers = np.asarray(centers, dtype=float)
        if std is not None:
            self._std = np.asarray(std, dtype=float)

    def train(self, x: np.ndarray, y: np.ndarray):
        if not self.trainable:
            raise RuntimeError("RBF model is missing trainable parameters.")
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(x.shape[0], self._obj)
        self._centers, self._std = self._find_center(x)
        distances = self._dist(self._centers, x.T)
        spreads = np.tile(self._std.reshape(-1, 1), (1, x.shape[0]))
        hidden = np.exp(-((distances / spreads) ** 2)).T
        if self._optimizer == "sgd":
            self._sgd(y, hidden)
        elif self._optimizer == "max-gd":
            self._max_gd(y, hidden)
        else:
            raise ValueError("Unsupported RBF optimizer.")

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.predictable:
            raise RuntimeError("RBF model is not predictable yet.")
        x = np.asarray(x, dtype=float).reshape(-1, self._dim)
        distances = self._dist(self._centers, x.T)
        spreads = np.tile(self._std.reshape(-1, 1), (1, x.shape[0]))
        hidden = np.exp(-((distances / spreads) ** 2)).T
        return hidden @ self._weight + self._bias

    def _find_center(self, train_x: np.ndarray):
        kernel_size = min(self._kernel_size, train_x.shape[0])
        model = KMeans(n_clusters=kernel_size, n_init=10, random_state=0)
        model.fit(train_x)
        centers = model.cluster_centers_
        order = np.argsort(np.sum(centers ** 2, axis=1))
        centers = centers[order]
        all_distances = self._dist(centers, centers.T)
        d_max = all_distances.max(initial=1.0)
        for idx in range(kernel_size):
            all_distances[idx, idx] = d_max + 1.0
        all_distances = np.where(all_distances != 0.0, all_distances, 1e-6)
        std = self._alpha * np.min(all_distances, axis=0)
        return centers, std

    @staticmethod
    def _dist(mat1: np.ndarray, mat2: np.ndarray):
        return cdist(mat1, mat2.T)

    def _sgd(self, y: np.ndarray, hidden: np.ndarray):
        for _ in range(self._epoch):
            for idx in range(hidden.shape[0]):
                pred = hidden[idx].T @ self._weight + self._bias
                err = (pred - y[idx]).reshape(-1)
                dw = hidden[idx].reshape(-1, 1) * err.reshape(1, self._obj)
                self._weight = self._weight - self._lr * dw
                self._bias = self._bias - self._lr * err.reshape(1, self._obj)

    def _max_gd(self, y: np.ndarray, hidden: np.ndarray):
        for _ in range(self._epoch):
            pred = hidden @ self._weight + self._bias
            err = pred - y
            idx = int(np.argmax(np.sum(err ** 2, axis=1)))
            dw = hidden[idx].reshape(-1, 1) * err[idx].reshape(1, self._obj)
            self._weight = self._weight - self._lr * dw
            self._bias = self._bias - self._lr * err[idx].reshape(1, self._obj)

    @property
    def params(self):
        keys = ("_weight", "_bias", "_centers", "_std")
        return {key: getattr(self, key) for key in keys if getattr(self, key) is not None}

    @property
    def trainable(self):
        return all(value is not None for value in (self._epoch, self._lr, self._alpha, self._weight, self._bias, self._optimizer))

    @property
    def predictable(self):
        return all(value is not None for value in (self._weight, self._bias, self._centers, self._std))

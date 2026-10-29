"""Building blocks of RPCGS: least-squares RBF surrogate, conditional generator and small helpers."""
from __future__ import annotations

import os
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pyDOE3 import lhs
from .framework import ClientPackage as Pkg
from scipy.spatial.distance import cdist

try:
    import torch
    from torch import nn, optim
except Exception:  # pragma: no cover
    torch = None
    nn = None
    optim = None

_TORCH_CPU_THREADS_SET = False


class Actions(Enum):
    PUSH_INIT = auto()
    PULL_INIT = auto()
    PUSH_UPDATE = auto()
    PULL_UPDATE = auto()


class ClientPackage(Pkg):
    def __init__(self, cid: int, action: Enum, version: int | None = None, data: Any = None):
        super().__init__(cid, action)
        self.version = version
        self.data = data


class RunFileLogger:
    LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

    def __init__(self, file_path: str | Path, level: str = "DEBUG"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.level = str(level).upper()

    def _write(self, level: str, msg: str):
        if self.LEVELS.get(level, 10) < self.LEVELS.get(self.level, 10):
            return
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}\n")

    def debug(self, msg: str):
        self._write("DEBUG", msg)

    def info(self, msg: str):
        self._write("INFO", msg)

    def warning(self, msg: str):
        self._write("WARNING", msg)


def make_run_tag() -> str:
    return f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{os.getpid()}"


def init_samples(lb, ub, dim: int, size: int) -> np.ndarray:
    lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
    return lhs(dim, samples=size) * (ub - lb) + lb


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    v_min, v_max = float(np.min(values)), float(np.max(values))
    if abs(v_max - v_min) < 1e-12:
        return np.zeros_like(values)
    return (values - v_min) / (v_max - v_min)


def empirical_quantiles(values: np.ndarray) -> np.ndarray:
    """Rank of each value scaled to [0, 1] (0 = smallest)."""
    values = np.asarray(values, dtype=float).reshape(-1)
    q = np.zeros(values.size)
    if values.size > 1:
        q[np.argsort(values)] = np.linspace(0.0, 1.0, values.size)
    return q


def reduce_signature(signature: np.ndarray, out_dim: int) -> np.ndarray:
    signature = np.asarray(signature, dtype=float).reshape(-1)
    if signature.size == 0:
        return np.zeros(out_dim)
    if signature.size == out_dim:
        return signature
    if signature.size < out_dim:
        return np.pad(signature, (0, out_dim - signature.size), mode="edge")
    return signature[np.linspace(0, signature.size - 1, out_dim).astype(int)]


def to_unit_cube(X: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    lb, ub = np.asarray(lb, dtype=float).reshape(1, -1), np.asarray(ub, dtype=float).reshape(1, -1)
    den = ub - lb
    den[den == 0.0] = 1.0
    return np.clip((np.asarray(X, dtype=float) - lb) / den, 0.0, 1.0)


def from_unit_cube(X: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    lb, ub = np.asarray(lb, dtype=float).reshape(1, -1), np.asarray(ub, dtype=float).reshape(1, -1)
    return np.clip(lb + np.asarray(X, dtype=float) * (ub - lb), lb, ub)


def average_generator_states(states: list[Optional[dict]], weights: np.ndarray) -> Optional[dict]:
    """Weighted average of generator states; None unless every client has a trained generator."""
    if not states or any(not s for s in states):
        return None
    w = np.asarray(weights, dtype=float)
    avg = lambda arrays: np.sum(np.stack(arrays) * w.reshape(-1, *([1] * np.asarray(arrays[0]).ndim)), axis=0)
    if all("weights" in s and "biases" in s for s in states):
        return {"weights": [avg([np.asarray(s["weights"][i]) for s in states]) for i in range(len(states[0]["weights"]))],
                "biases": [avg([np.asarray(s["biases"][i]) for s in states]) for i in range(len(states[0]["biases"]))]}
    if all("elite_mean" in s and "elite_std" in s for s in states):
        return {"elite_mean": avg([np.asarray(s["elite_mean"]) for s in states]),
                "elite_std": avg([np.asarray(s["elite_std"]) for s in states])}
    return None


class LeastSquaresRBF:
    """Gaussian RBF network: k-means centers, spreads alpha * distance to the nearest other center, and an
    output layer solved by ridge-regularized least squares (bias not penalized)."""

    def __init__(self, dim: int, kernel_size: int, ridge: float = 1e-6, alpha: float = 1.0):
        self._dim, self._kernel_size, self._ridge, self._alpha = int(dim), int(kernel_size), float(ridge), float(alpha)
        self._weight = self._bias = self._centers = self._std = None
        self._rng = np.random.default_rng(np.random.randint(0, 2 ** 31 - 1))

    @property
    def predictable(self) -> bool:
        return self._weight is not None

    def copy_from(self, other: "LeastSquaresRBF"):
        self._weight, self._bias, self._centers, self._std = other._weight, other._bias, other._centers, other._std

    def _find_center(self, x: np.ndarray):
        """k-means++ followed by Lloyd iterations in plain NumPy (sklearn's KMeans calls threadpoolctl, which
        deadlocks against library loading in other client threads of the same process)."""
        k = self._kernel_size
        centers = [x[self._rng.integers(len(x))]]
        d2 = ((x - centers[0]) ** 2).sum(axis=1)
        for _ in range(1, k):
            total = d2.sum()
            idx = self._rng.choice(len(x), p=d2 / total) if total > 0 else self._rng.integers(len(x))
            centers.append(x[idx])
            d2 = np.minimum(d2, ((x - x[idx]) ** 2).sum(axis=1))
        centers = np.array(centers)
        for _ in range(100):
            label = ((x[:, None, :] - centers[None]) ** 2).sum(axis=2).argmin(axis=1)
            new = np.array([x[label == j].mean(axis=0) if np.any(label == j) else centers[j] for j in range(k)])
            if np.allclose(new, centers):
                break
            centers = new
        centers = centers[np.argsort(np.sum(centers ** 2, axis=1))]
        dist = cdist(centers, centers)
        np.fill_diagonal(dist, dist.max(initial=-10) + 1)
        dist = np.where(dist != 0, dist, 1e-6)
        return centers, self._alpha * dist.min(axis=0)

    def _hidden(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-(cdist(self._centers, x) / self._std.reshape(-1, 1)) ** 2).T

    def train(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(x.shape[0], 1)
        # k-means cannot place more centers than there are distinct points (bootstrap samples repeat rows)
        self._kernel_size = max(1, min(self._kernel_size, np.unique(x, axis=0).shape[0]))
        self._centers, self._std = self._find_center(x)
        design = np.hstack([self._hidden(x), np.ones((x.shape[0], 1))])
        gram = design.T @ design
        penalty = self._ridge * max(np.trace(gram) / design.shape[1], 1e-12) * np.eye(design.shape[1])
        penalty[-1, -1] = 0.0
        coef = np.linalg.solve(gram + penalty, design.T @ y)
        self._weight, self._bias = coef[:-1], coef[-1:].reshape(1, 1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._hidden(np.asarray(x, dtype=float)) @ self._weight + self._bias


def _torch_device(backend: str):
    global _TORCH_CPU_THREADS_SET
    if torch is None or str(backend).lower() == "none":
        return False, None
    if torch.cuda.is_available() and str(backend).lower() in {"auto", "torch", "cuda"}:
        return True, torch.device("cuda")
    if not _TORCH_CPU_THREADS_SET:
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        _TORCH_CPU_THREADS_SET = True
    return True, torch.device("cpu")


if nn is not None:
    class _GeneratorNet(nn.Module):
        """MLP; with gauss=True it outputs a mean and a standard deviation per coordinate instead of a point."""

        def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, gauss: bool = False):
            super().__init__()
            h2 = max(16, hidden_dim // 2)
            self.out_dim, self.gauss = out_dim, gauss
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, h2), nn.ReLU(),
                      nn.Linear(h2, 2 * out_dim if gauss else out_dim)]
            self.net = nn.Sequential(*(layers if gauss else layers + [nn.Sigmoid()]))

        def forward(self, x):
            out = self.net(x)
            if not self.gauss:
                return out
            return torch.sigmoid(out[:, :self.out_dim]), 0.01 + 0.5 * torch.sigmoid(out[:, self.out_dim:])


class ConditionalGenerator:
    """Maps (latent noise, target quantile q, task embedding) to a point in the unit cube.

    loss "mse": weighted regression on elite points (q = their quantile among the client's data); the output
    depends little on the noise, so samples concentrate near a conditional mean. loss "gauss": the network
    outputs a per-coordinate mean and standard deviation, trained by the weighted Gaussian negative
    log-likelihood, and samples are drawn from that Gaussian. Without PyTorch it falls back to a Gaussian
    around the weighted elite mean."""

    def __init__(self, dim: int, embed_dim: int, *, latent_dim: int = 8, hidden_dim: int = 64, lr: float = 1e-3,
                 epochs: int = 50, batch_size: int = 32, backend: str = "auto", seed: Optional[int] = None,
                 loss: str = "mse"):
        self.dim, self.embed_dim, self.latent_dim = int(dim), int(embed_dim), int(latent_dim)
        self.lr, self.epochs, self.batch_size = float(lr), int(epochs), int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.trained = False
        self.elite_mean = np.zeros(self.dim)
        self.elite_std = np.ones(self.dim) * 0.1
        self.gauss = str(loss).lower() == "gauss"
        self.use_torch, self.device = _torch_device(backend)
        self.model = _GeneratorNet(self.latent_dim + 1 + self.embed_dim, int(hidden_dim), self.dim, self.gauss).to(self.device) \
            if self.use_torch else None

    def fit(self, X_unit: np.ndarray, q_vals: np.ndarray, task_embed: np.ndarray, sample_weight: np.ndarray):
        X_unit = np.asarray(X_unit, dtype=float)
        if X_unit.size == 0:
            return
        q_vals = np.asarray(q_vals, dtype=float).reshape(-1, 1)
        sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1, 1)
        sw = sample_weight / max(1e-12, float(np.sum(sample_weight)))
        self.elite_mean = np.sum(X_unit * sw, axis=0)
        self.elite_std = np.sqrt(np.maximum(np.sum(((X_unit - self.elite_mean) ** 2) * sw, axis=0), 1e-6))
        if X_unit.shape[0] < 2 or not self.use_torch:
            self.trained = True
            return
        emb = np.repeat(np.asarray(task_embed, dtype=float).reshape(1, -1), X_unit.shape[0], axis=0)
        z = self.rng.normal(size=(X_unit.shape[0], self.latent_dim))
        X_t = torch.from_numpy(np.hstack([z, q_vals, emb]).astype(np.float32)).to(self.device)
        y_t = torch.from_numpy(X_unit.astype(np.float32)).to(self.device)
        w_t = torch.from_numpy(np.maximum(sample_weight, 1e-4).astype(np.float32)).to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        n = X_unit.shape[0]
        for _ in range(self.epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, self.batch_size):
                bid = idx[start:start + self.batch_size]
                if self.gauss:
                    mean, std = self.model(X_t[bid])
                    nll = torch.log(std) + 0.5 * ((y_t[bid] - mean) / std) ** 2
                    loss = (nll.mean(dim=1, keepdim=True) * w_t[bid]).mean()
                else:
                    loss = (((self.model(X_t[bid]) - y_t[bid]) ** 2).mean(dim=1, keepdim=True) * w_t[bid]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.trained = True

    def sample(self, num: int, q_target: float, task_embed: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        num = int(max(0, num))
        if num == 0:
            return np.empty((0, self.dim))
        if self.use_torch and self.trained:
            z = self.rng.normal(size=(num, self.latent_dim))
            emb = np.repeat(np.asarray(task_embed, dtype=float).reshape(1, -1), num, axis=0)
            inp = np.hstack([z, np.full((num, 1), float(q_target)), emb]).astype(np.float32)
            with torch.no_grad():
                out = self.model(torch.from_numpy(inp).to(self.device))
                if self.gauss:
                    mean, std = (t.cpu().numpy() for t in out)
                    out = np.clip(mean + std * self.rng.normal(size=mean.shape), 0.0, 1.0)
                else:
                    out = out.cpu().numpy()
            return from_unit_cube(out, lb, ub)
        scale = max(0.05, float(q_target) + 0.05)
        samples = self.rng.normal(loc=self.elite_mean, scale=self.elite_std * scale, size=(num, self.dim))
        return from_unit_cube(np.clip(samples, 0.0, 1.0), lb, ub)

    def get_state(self) -> Optional[dict]:
        if not self.trained:
            return None
        if self.use_torch:
            layers = [m for m in self.model.net if isinstance(m, nn.Linear)]
            with torch.no_grad():
                return {"weights": [m.weight.detach().cpu().numpy().copy() for m in layers],
                        "biases": [m.bias.detach().cpu().numpy().reshape(-1, 1).copy() for m in layers]}
        return {"elite_mean": self.elite_mean.copy(), "elite_std": self.elite_std.copy()}

    def set_state(self, state: Optional[dict]):
        if not state:
            return
        if self.use_torch and "weights" in state:
            layers = [m for m in self.model.net if isinstance(m, nn.Linear)]
            with torch.no_grad():
                for m, w, b in zip(layers, state["weights"], state["biases"]):
                    m.weight[:] = torch.from_numpy(np.asarray(w)).to(m.weight.device)
                    m.bias[:] = torch.from_numpy(np.asarray(b).reshape(-1)).to(m.bias.device)
            self.trained = True
        elif "elite_mean" in state:
            self.elite_mean = np.asarray(state["elite_mean"], dtype=float).reshape(-1)
            self.elite_std = np.asarray(state["elite_std"], dtype=float).reshape(-1)
            self.trained = True

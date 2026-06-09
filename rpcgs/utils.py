from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from pyDOE3 import lhs

from .framework import Actions, ClientPackage
from .rbf import RadialBasisFunctionNetwork as RBFNetwork

try:
    import torch
    from torch import nn, optim
except Exception:  # pragma: no cover
    torch = None
    nn = None
    optim = None


_TORCH_CPU_THREADS_SET = False


class RunFileLogger:
    LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

    def __init__(self, file_path: str | Path, level: str = "DEBUG"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.level = str(level).upper()

    def _enabled(self, level: str) -> bool:
        return self.LEVELS.get(level, 10) >= self.LEVELS.get(self.level, 10)

    def _write(self, level: str, msg: str):
        if not self._enabled(level):
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.file_path, "a", encoding="utf-8") as file:
            file.write(f"{timestamp} [{level}] {msg}\n")

    def debug(self, msg: str):
        self._write("DEBUG", msg)

    def info(self, msg: str):
        self._write("INFO", msg)

    def warning(self, msg: str):
        self._write("WARNING", msg)

    def error(self, msg: str):
        self._write("ERROR", msg)


def make_run_tag() -> str:
    return f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{os.getpid()}"


def init_samples(lb, ub, dim: int, size: int) -> np.ndarray:
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    return lhs(dim, samples=size) * (ub - lb) + lb


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if abs(v_max - v_min) < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - v_min) / (v_max - v_min)


def progress_to_signature(history: list[float], window: int) -> np.ndarray:
    history = np.asarray(history[-window:], dtype=float).reshape(-1)
    if history.size < window:
        history = np.pad(history, (window - history.size, 0), mode="constant")
    return minmax_normalize(history)


def ema_blend(previous: np.ndarray, current: np.ndarray, decay: float) -> np.ndarray:
    return decay * np.asarray(previous, dtype=float) + (1.0 - decay) * np.asarray(current, dtype=float)


def _resolve_torch_device(backend: str):
    global _TORCH_CPU_THREADS_SET
    if torch is None:
        return False, None
    mode = str(backend).lower()
    if mode in {"auto", "torch"}:
        if torch.cuda.is_available():
            return True, torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True, torch.device("mps")
        if not _TORCH_CPU_THREADS_SET:
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            _TORCH_CPU_THREADS_SET = True
        return True, torch.device("cpu")
    if mode == "cpu":
        if not _TORCH_CPU_THREADS_SET:
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            _TORCH_CPU_THREADS_SET = True
        return True, torch.device("cpu")
    if mode == "cuda":
        if torch.cuda.is_available():
            return True, torch.device("cuda")
        return True, torch.device("cpu")
    if mode == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True, torch.device("mps")
        return True, torch.device("cpu")
    if mode == "none":
        return False, None
    return False, None


def pairwise_mean_abs_distance(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    diffs = np.abs(matrix[:, None, :] - matrix[None, :, :])
    return np.mean(diffs, axis=2)


def build_total_distance_matrix(
    signatures: dict[str, np.ndarray],
    lambda_rank: float,
    lambda_unc: float,
    lambda_prog: float,
) -> np.ndarray:
    rank_dist = pairwise_mean_abs_distance(signatures["rank"])
    unc_dist = pairwise_mean_abs_distance(signatures["unc"])
    prog_dist = pairwise_mean_abs_distance(signatures["prog"])
    total = lambda_rank * rank_dist + lambda_unc * unc_dist + lambda_prog * prog_dist
    total = 0.5 * (total + total.T)
    np.fill_diagonal(total, 0.0)
    return total


def stable_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    denom = np.sum(exp_values)
    if denom <= 0:
        return np.ones_like(values) / max(1, values.size)
    return exp_values / denom


def reduce_signature(signature: np.ndarray, out_dim: int) -> np.ndarray:
    signature = np.asarray(signature, dtype=float).reshape(-1)
    if signature.size == 0:
        return np.zeros(out_dim, dtype=float)
    if signature.size == out_dim:
        return signature
    if signature.size < out_dim:
        return np.pad(signature, (0, out_dim - signature.size), mode="edge")
    idx = np.linspace(0, signature.size - 1, out_dim).astype(int)
    return signature[idx]


def build_candidate_meta_features(
    mu: np.ndarray,
    unc: np.ndarray,
    disagree: np.ndarray,
    trust: np.ndarray,
    novelty: np.ndarray,
    q_hat: np.ndarray,
) -> np.ndarray:
    mu = np.asarray(mu, dtype=float).reshape(-1)
    unc = np.asarray(unc, dtype=float).reshape(-1)
    disagree = np.asarray(disagree, dtype=float).reshape(-1)
    trust = np.asarray(trust, dtype=float).reshape(-1)
    novelty = np.asarray(novelty, dtype=float).reshape(-1)
    q_hat = np.asarray(q_hat, dtype=float).reshape(-1)

    # high is good for all channels
    mu_good = 1.0 - empirical_quantiles(mu)
    unc_good = 1.0 - empirical_quantiles(unc)
    dis_good = 1.0 - empirical_quantiles(disagree)
    nov_good = empirical_quantiles(novelty)
    trust_good = minmax_normalize(trust)
    agree_good = 1.0 - minmax_normalize(disagree)
    pseudo_good = 1.0 - q_hat
    feats = np.stack(
        [mu_good, unc_good, dis_good, trust_good, nov_good, agree_good, pseudo_good],
        axis=1,
    )
    return feats.astype(np.float32)


def empirical_quantiles(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    n = values.size
    if n == 0:
        return values
    order = np.argsort(values)
    q = np.zeros(n, dtype=float)
    if n == 1:
        q[0] = 0.0
    else:
        q[order] = np.linspace(0.0, 1.0, n)
    return q


def to_unit_cube(X: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    lb = np.asarray(lb, dtype=float).reshape(1, -1)
    ub = np.asarray(ub, dtype=float).reshape(1, -1)
    den = ub - lb
    den[den == 0.0] = 1.0
    return np.clip((X - lb) / den, 0.0, 1.0)


def from_unit_cube(X: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    lb = np.asarray(lb, dtype=float).reshape(1, -1)
    ub = np.asarray(ub, dtype=float).reshape(1, -1)
    return np.clip(lb + X * (ub - lb), lb, ub)


def avg_weighted_arrays(arrays: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    stack = np.stack(arrays, axis=0)
    w = np.asarray(weights, dtype=float).reshape(-1, *([1] * (stack.ndim - 1)))
    return np.sum(stack * w, axis=0)


def build_pairwise_selector_data(
    features: np.ndarray,
    y_true: np.ndarray,
    max_pairs: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    n = features.shape[0]
    if n < 2:
        return np.empty((0, features.shape[1] * 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
    rng = rng or np.random.default_rng(123)
    all_pairs = []
    labels = []
    total = n * (n - 1) // 2
    if max_pairs <= 0 or max_pairs >= total:
        idx_iter = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
    else:
        seen = set()
        idx_iter = []
        while len(idx_iter) < max_pairs:
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            idx_iter.append((a, b))
    for i, j in idx_iter:
        if abs(y_true[i] - y_true[j]) < 1e-12:
            continue
        pair = np.concatenate([features[i], features[j]], axis=0)
        # label 1: second better; 0: first better
        label = 1.0 if y_true[j] < y_true[i] else 0.0
        all_pairs.append(pair)
        labels.append(label)
    if not all_pairs:
        return np.empty((0, features.shape[1] * 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.asarray(all_pairs, dtype=np.float32), np.asarray(labels, dtype=np.float32)


class _SelectorNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        h2 = max(16, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.net(x)


class _TorchGeneratorNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        h2 = max(16, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class TrustConditionedGenerator:
    def __init__(
        self,
        dim: int,
        embed_dim: int,
        *,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        backend: str = "auto",
        seed: Optional[int] = None,
    ):
        self.dim = int(dim)
        self.embed_dim = int(embed_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.backend = backend
        self.rng = np.random.default_rng(seed)
        self.trained = False
        self.elite_mean = np.zeros(self.dim, dtype=float)
        self.elite_std = np.ones(self.dim, dtype=float) * 0.1

        self.use_torch, self.device = _resolve_torch_device(backend)
        self.model = None
        if self.use_torch:
            in_dim = self.latent_dim + 1 + self.embed_dim
            self.model = _TorchGeneratorNet(in_dim, self.hidden_dim, self.dim).to(self.device)

    def fit(self, X_unit: np.ndarray, q_vals: np.ndarray, task_embed: np.ndarray, sample_weight: np.ndarray):
        X_unit = np.asarray(X_unit, dtype=float)
        q_vals = np.asarray(q_vals, dtype=float).reshape(-1, 1)
        sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1, 1)
        if X_unit.size == 0:
            return
        norm_weight = sample_weight / max(1e-12, float(np.sum(sample_weight)))
        self.elite_mean = np.sum(X_unit * norm_weight, axis=0)
        centered = X_unit - self.elite_mean.reshape(1, -1)
        self.elite_std = np.sqrt(np.maximum(np.sum((centered ** 2) * norm_weight, axis=0), 1e-6))
        if X_unit.shape[0] < 2 or not self.use_torch:
            self.trained = True
            return
        task_embed = np.asarray(task_embed, dtype=float).reshape(1, -1)
        task_embed = np.repeat(task_embed, X_unit.shape[0], axis=0)
        z = self.rng.normal(size=(X_unit.shape[0], self.latent_dim)).astype(np.float32)
        inp = np.hstack([z, q_vals.astype(np.float32), task_embed.astype(np.float32)]).astype(np.float32)
        target = X_unit.astype(np.float32)
        weight = np.maximum(sample_weight.astype(np.float32), 1e-4)
        X_t = torch.from_numpy(inp).to(self.device)
        y_t = torch.from_numpy(target).to(self.device)
        w_t = torch.from_numpy(weight).to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        n = X_unit.shape[0]
        for _ in range(self.epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, self.batch_size):
                bid = idx[start:start + self.batch_size]
                pred = self.model(X_t[bid])
                loss = (((pred - y_t[bid]) ** 2).mean(dim=1, keepdim=True) * w_t[bid]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.trained = True

    def sample(self, num: int, q_target: float, task_embed: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        num = int(max(0, num))
        if num == 0:
            return np.empty((0, self.dim), dtype=float)
        task_embed = np.asarray(task_embed, dtype=float).reshape(1, -1)
        if self.use_torch and self.trained:
            z = self.rng.normal(size=(num, self.latent_dim)).astype(np.float32)
            q = np.ones((num, 1), dtype=np.float32) * float(q_target)
            emb = np.repeat(task_embed.astype(np.float32), num, axis=0)
            inp = np.hstack([z, q, emb]).astype(np.float32)
            with torch.no_grad():
                out = self.model(torch.from_numpy(inp).to(self.device)).cpu().numpy()
            return from_unit_cube(out, lb, ub)
        scale = max(0.05, float(q_target) + 0.05)
        samples = self.rng.normal(
            loc=self.elite_mean.reshape(1, -1),
            scale=(self.elite_std * scale).reshape(1, -1),
            size=(num, self.dim),
        )
        return from_unit_cube(np.clip(samples, 0.0, 1.0), lb, ub)

    def get_transfer_state(self) -> Optional[dict]:
        if self.use_torch and self.trained:
            weights = []
            biases = []
            with torch.no_grad():
                for layer in self.model.net:
                    if isinstance(layer, nn.Linear):
                        weights.append(layer.weight.detach().cpu().numpy().copy())
                        biases.append(layer.bias.detach().cpu().numpy().reshape(-1, 1).copy())
            return {"weights": weights, "biases": biases}
        if self.trained:
            return {"elite_mean": self.elite_mean.copy(), "elite_std": self.elite_std.copy()}
        return None

    def set_transfer_state(self, state: Optional[dict]):
        if not state:
            return
        if self.use_torch and "weights" in state and "biases" in state:
            with torch.no_grad():
                wi = 0
                for layer in self.model.net:
                    if isinstance(layer, nn.Linear):
                        layer.weight[:] = torch.from_numpy(np.asarray(state["weights"][wi])).to(layer.weight.device)
                        layer.bias[:] = torch.from_numpy(np.asarray(state["biases"][wi]).reshape(-1)).to(layer.bias.device)
                        wi += 1
            self.trained = True
            return
        if "elite_mean" in state and "elite_std" in state:
            self.elite_mean = np.asarray(state["elite_mean"], dtype=float).reshape(-1)
            self.elite_std = np.asarray(state["elite_std"], dtype=float).reshape(-1)
            self.trained = True


class MetaFeatureSelector:
    def __init__(
        self,
        feat_dim: int,
        *,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        backend: str = "auto",
        seed: Optional[int] = None,
    ):
        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.backend = backend
        self.rng = np.random.default_rng(seed)
        self.trained = False

        self.use_torch, self.device = _resolve_torch_device(backend)
        self.model = None
        if self.use_torch:
            self.model = _SelectorNet(self.feat_dim * 2, self.hidden_dim).to(self.device)

    def fit(self, pair_x: np.ndarray, pair_y: np.ndarray):
        pair_x = np.asarray(pair_x, dtype=np.float32)
        pair_y = np.asarray(pair_y, dtype=np.float32).reshape(-1, 1)
        if pair_x.shape[0] < 4:
            return
        if not self.use_torch:
            self.trained = True
            return
        X_t = torch.from_numpy(pair_x).to(self.device)
        y_t = torch.from_numpy(pair_y).to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        self.model.train()
        n = pair_x.shape[0]
        for _ in range(self.epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, self.batch_size):
                bid = idx[start:start + self.batch_size]
                xb = X_t[bid]
                yb = y_t[bid]
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.trained = True

    def compare_score(self, features: np.ndarray) -> np.ndarray:
        # fallback score if selector not trained
        features = np.asarray(features, dtype=np.float32)
        base = (
            0.35 * features[:, 0] +
            0.20 * features[:, 3] +
            0.15 * features[:, 1] +
            0.10 * features[:, 2] +
            0.10 * features[:, 4] +
            0.10 * features[:, 5]
        )
        return base.astype(np.float32)

    def tournament_select(
        self,
        features: np.ndarray,
        candidate_idx: np.ndarray,
        duels: int,
        rng: np.random.Generator,
    ) -> int:
        candidate_idx = np.asarray(candidate_idx, dtype=int)
        if candidate_idx.size == 1:
            return int(candidate_idx[0])
        if not self.trained or not self.use_torch:
            scores = self.compare_score(features[candidate_idx])
            return int(candidate_idx[int(np.argmax(scores))])

        wins = np.zeros(candidate_idx.shape[0], dtype=float)
        self.model.eval()
        with torch.no_grad():
            for pos, idx in enumerate(candidate_idx):
                if candidate_idx.size == 1:
                    break
                opponents = rng.choice(candidate_idx[candidate_idx != idx], size=min(duels, candidate_idx.size - 1), replace=False)
                fi = np.repeat(features[idx].reshape(1, -1), opponents.shape[0], axis=0)
                fj = features[opponents]
                pair = np.hstack([fi, fj]).astype(np.float32)
                pair_t = torch.from_numpy(pair).to(self.device)
                prob_second = torch.sigmoid(self.model(pair_t)).cpu().numpy().reshape(-1)
                # If second is better, idx loses; win score uses 1 - p(second better)
                wins[pos] = float(np.mean(1.0 - prob_second))
        best_pos = int(np.argmax(wins))
        return int(candidate_idx[best_pos])

    def get_state(self) -> Optional[dict]:
        if self.use_torch and self.trained:
            weights = []
            biases = []
            with torch.no_grad():
                for layer in self.model.net:
                    if isinstance(layer, nn.Linear):
                        weights.append(layer.weight.detach().cpu().numpy().copy())
                        biases.append(layer.bias.detach().cpu().numpy().reshape(-1, 1).copy())
            return {"weights": weights, "biases": biases}
        return None

    def set_state(self, state: Optional[dict]):
        if not state or not self.use_torch:
            return
        if "weights" not in state or "biases" not in state:
            return
        wi = 0
        with torch.no_grad():
            for layer in self.model.net:
                if isinstance(layer, nn.Linear):
                    layer.weight[:] = torch.from_numpy(np.asarray(state["weights"][wi])).to(layer.weight.device)
                    layer.bias[:] = torch.from_numpy(np.asarray(state["biases"][wi]).reshape(-1)).to(layer.bias.device)
                    wi += 1
        self.trained = True


def average_rbf_networks(networks: list[dict], weights: np.ndarray) -> dict:
    if not networks:
        return {}
    keys = ["_weight", "_bias", "_centers", "_std"]
    aggregated = {}
    for key in keys:
        vals = [np.asarray(net[key]) for net in networks if key in net and net[key] is not None]
        if len(vals) != len(networks):
            return {}
        aggregated[key] = avg_weighted_arrays(vals, weights)
    return aggregated


def average_generator_states(states: list[Optional[dict]], weights: np.ndarray) -> Optional[dict]:
    valid = [state for state in states if state]
    if len(valid) != len(states) or not valid:
        return None
    if all(("weights" in state and "biases" in state) for state in valid):
        avg_weights = []
        avg_biases = []
        for idx in range(len(valid[0]["weights"])):
            avg_weights.append(avg_weighted_arrays([np.asarray(state["weights"][idx]) for state in valid], weights))
        for idx in range(len(valid[0]["biases"])):
            avg_biases.append(avg_weighted_arrays([np.asarray(state["biases"][idx]) for state in valid], weights))
        return {"weights": avg_weights, "biases": avg_biases}
    if all(("elite_mean" in state and "elite_std" in state) for state in valid):
        means = [np.asarray(state["elite_mean"]) for state in valid]
        stds = [np.asarray(state["elite_std"]) for state in valid]
        return {"elite_mean": avg_weighted_arrays(means, weights), "elite_std": avg_weighted_arrays(stds, weights)}
    return None


def create_trainable_rbf(dim: int, obj: int, *, epoch: int, optimizer: str, lr: float, alpha: float, rng: np.random.Generator) -> RBFNetwork:
    model = RBFNetwork(dim=dim, obj=obj, kernel_size=2 * dim + 1, epoch=epoch, optimizer=optimizer, lr=lr, alpha=alpha)
    model.sync_manuel(weight=rng.standard_normal((2 * dim + 1, obj)), bias=rng.standard_normal((1, obj)))
    return model


def aggregate_personalized_payloads(
    distances: np.ndarray,
    networks: list[dict],
    generators: list[Optional[dict]],
    top_k: int,
    tau: float,
) -> tuple[list[dict], list[Optional[dict]], dict[int, list[int]], dict[int, list[float]]]:
    net_payloads, gen_payloads, _, neighbor_map, weight_map = aggregate_personalized_payloads_v2(
        distances=distances,
        networks=networks,
        generators=generators,
        selectors=[None for _ in generators],
        top_k=top_k,
        tau=tau,
    )
    return net_payloads, gen_payloads, neighbor_map, weight_map


def aggregate_personalized_payloads_v2(
    distances: np.ndarray,
    networks: list[dict],
    generators: list[Optional[dict]],
    selectors: list[Optional[dict]],
    top_k: int,
    tau: float,
) -> tuple[list[dict], list[Optional[dict]], list[Optional[dict]], dict[int, list[int]], dict[int, list[float]]]:
    num_clients = distances.shape[0]
    top_k = max(1, min(int(top_k), num_clients))
    tau = max(float(tau), 1e-12)
    net_payloads: list[dict] = []
    gen_payloads: list[Optional[dict]] = []
    sel_payloads: list[Optional[dict]] = []
    neighbor_map: dict[int, list[int]] = {}
    weight_map: dict[int, list[float]] = {}
    for idx in range(num_clients):
        neighbor_ids = np.argsort(distances[idx])[:top_k]
        logits = -distances[idx, neighbor_ids] / tau
        weights = np.exp(logits - np.max(logits))
        weights = weights / np.sum(weights)
        selected_networks = [networks[j] for j in neighbor_ids]
        selected_generators = [generators[j] for j in neighbor_ids]
        selected_selectors = [selectors[j] for j in neighbor_ids]
        net_payloads.append(average_rbf_networks(selected_networks, weights))
        gen_payloads.append(average_generator_states(selected_generators, weights))
        sel_payloads.append(average_generator_states(selected_selectors, weights))
        neighbor_map[idx] = neighbor_ids.tolist()
        weight_map[idx] = weights.tolist()
    return net_payloads, gen_payloads, sel_payloads, neighbor_map, weight_map

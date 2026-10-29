from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from .framework import Client, record_runtime
from scipy.stats import norm

from .utils import (
    Actions,
    ClientPackage,
    ConditionalGenerator,
    LeastSquaresRBF,
    RunFileLogger,
    empirical_quantiles,
    minmax_normalize,
    reduce_signature,
    to_unit_cube,
)


class RpcgsClient(Client):
    """
    rbf_ridge: 1.0e-6
    rbf_alpha: 1.0
    bootstrap_num: 5
    y_norm: zscore
    candidate_random: 96
    candidate_perturb: 48
    candidate_generate: 64
    candidate_boundary: 32
    local_region: true
    tr_init: 0.2
    tr_min: 0.0125
    tr_max: 0.5
    tr_succ_tol: 3
    tr_fail_tol: 0
    pseudo_top_quantile: 0.15
    acq_lambda: 2.0
    generator_update_interval: 5
    generator_max_true: 16
    generator_max_pseudo: 24
    q_targets: [0.2, 0.1, 0.05]
    embed_dim: 8
    generator_latent: 8
    generator_hidden: 64
    generator_epochs: 50
    generator_lr: 0.001
    generator_batch_size: 32
    generator_backend: auto
    generator_loss: mse
    task_embed: rank
    allocation: fixed
    proposal: hybrid
    center_alloc: uniform
    center_scale: 0.5
    hybrid_gen_share: 0.75
    alloc_prior: [1.0, 9.0]
    alloc_decay: 0.95
    alloc_min_share: 0.05
    diag_log: true
    seed: 123
    debug_level: DEBUG
    """

    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        self.problem.auto_update_solutions = True
        g = kwargs.get
        self.rbf_ridge, self.rbf_alpha = float(g("rbf_ridge", 1e-6)), float(g("rbf_alpha", 1.0))
        self.bootstrap_num = int(g("bootstrap_num", 5))
        # y_norm: objective values are z-scored on the client before fitting ("none" keeps raw values)
        self.y_norm = str(g("y_norm", "zscore")).lower()
        self.candidate_random = int(g("candidate_random", 96))
        self.candidate_perturb = int(g("candidate_perturb", 48))
        self.candidate_generate = int(g("candidate_generate", 64))
        # boundary candidates: the incumbent with one coordinate moved to its lower or upper bound (2 * dim points),
        # then uniform samples with one random coordinate on a random bound
        self.candidate_boundary = int(g("candidate_boundary", 32))
        # adaptive local region around the incumbent (TuRBO-style success/failure rule); false = fixed sigma 0.1
        self.local_region = bool(g("local_region", True))
        self.tr_init, self.tr_min, self.tr_max = float(g("tr_init", 0.2)), float(g("tr_min", 0.0125)), float(g("tr_max", 0.5))
        self.tr_succ_tol = int(g("tr_succ_tol", 3))
        self.tr_fail_tol = int(g("tr_fail_tol", 0)) or max(4, self.dim)
        self.tr_radius, self._tr_succ, self._tr_fail = self.tr_init, 0, 0
        # generator training data: true elites (best quantile) and the pool candidates whose surrogate quality
        # mu - acq_lambda * normalized spread is in the best pseudo_top_quantile, all with weight 1
        self.pseudo_top_quantile = float(g("pseudo_top_quantile", 0.15))
        self.acq_lambda = float(g("acq_lambda", 2.0))
        self.generator_update_interval = int(g("generator_update_interval", 5))
        self.generator_max_true = int(g("generator_max_true", 16))
        self.generator_max_pseudo = int(g("generator_max_pseudo", 24))
        self.q_targets = list(g("q_targets", [0.2, 0.1, 0.05]))
        self.embed_dim = int(g("embed_dim", 8))
        # task_embed: "rank" conditions the shared generator on the ranks of the local predictions on the shared
        # reference points; "const" gives every client the same condition (ablation)
        self.task_embed = str(g("task_embed", "rank")).lower()
        # allocation: "fixed" keeps the pool sizes above; "bandit" reallocates the same total among the four
        # candidate sources (uniform, local region, generator, boundary) in proportion to default size times
        # the source's estimated success rate. A round counts as a success for the source of the chosen point
        # if it improves the incumbent; the rate is the mean of a Beta(alloc_prior) posterior on exponentially
        # discounted counts (alloc_decay per round); every source keeps at least alloc_min_share of the pool.
        self.allocation = str(g("allocation", "fixed")).lower()
        self.alloc_prior = [float(v) for v in g("alloc_prior", [1.0, 9.0])]
        self.alloc_decay = float(g("alloc_decay", 0.95))
        self.alloc_min_share = float(g("alloc_min_share", 0.05))
        self._alloc_succ, self._alloc_tries = np.zeros(4), np.zeros(4)
        # proposal: "generator" samples the shared conditional generator; "centers" proposes points around the
        # elite centers of all clients (mean and spread of each client's best points in its unit cube, forwarded
        # unchanged by the server) and around their average, the consensus center. Each proposal group is an
        # arm; its first point is the center itself, the others are Gaussian with center_scale * spread.
        # center_alloc: "uniform" splits the generator slots equally among arms; "bandit" splits them in
        # proportion to each arm's estimated success rate (same Beta prior and decay as allocation), where a
        # round is a success for the arm of the chosen point if it improves the incumbent.
        self.proposal = str(g("proposal", "hybrid")).lower()
        self.center_alloc = str(g("center_alloc", "uniform")).lower()
        self.center_scale = float(g("center_scale", 0.5))
        # proposal "hybrid": the shared generator is the consensus arm (key 0) next to the client center arms;
        # before any feedback the generator arm gets hybrid_gen_share of the slots and the center arms share the rest
        self.hybrid_gen_share = float(g("hybrid_gen_share", 0.75))
        self._centers: list[dict] = []
        self._arm_succ: dict = {}
        self._arm_tries: dict = {}
        self.diag_log = bool(g("diag_log", True))
        self.seed = g("seed", None)
        self.debug_level = str(g("debug_level", "DEBUG")).upper()
        self.pull_repeat, self.pull_interval = int(g("pull_repeat", 300)), float(g("pull_interval", 1.0))

        self.rng = np.random.default_rng(self.seed)
        self.lb_vec = self.lb if isinstance(self.lb, np.ndarray) else np.ones(self.dim) * self.lb
        self.ub_vec = self.ub if isinstance(self.ub, np.ndarray) else np.ones(self.dim) * self.ub
        new_rbf = lambda: LeastSquaresRBF(self.dim, 2 * self.dim + 1, ridge=self.rbf_ridge, alpha=self.rbf_alpha)
        self.local_model = new_rbf()
        self.bootstrap_models = [new_rbf() for _ in range(max(1, self.bootstrap_num))]
        self.generator = ConditionalGenerator(
            self.dim, self.embed_dim, latent_dim=int(g("generator_latent", 8)), hidden_dim=int(g("generator_hidden", 64)),
            lr=float(g("generator_lr", 1e-3)), epochs=int(g("generator_epochs", 50)),
            batch_size=int(g("generator_batch_size", 32)), backend=g("generator_backend", "auto"), seed=self.seed,
            loss=g("generator_loss", "mse"))
        self._yn = (0.0, 1.0)  # mean, std
        self.version = 1
        self.d_share = None
        self.file_logger = None
        self._diag_path = None
        self._last_generator_update_ver = 0
        self._last_choice = None

    # ---------------------------------------------------------------- rounds
    def optimize(self):
        if self.d_share is None:
            self._initializing()
        else:
            self._optimizing()

    def _initializing(self):
        self.request_server(ClientPackage(cid=self.id, action=Actions.PUSH_INIT,
                                          data={"dim": self.dim, "bound": (self.lb_vec, self.ub_vec)}))
        resp = self.request_server(ClientPackage(cid=self.id, action=Actions.PULL_INIT), repeat=100, interval=0.1)
        self.d_share = np.asarray(resp["d_share"])
        debug_dir = Path(resp["debug_dir"])
        self.file_logger = RunFileLogger(debug_dir / f"client_{self.id:02d}_{self.problem.name}.log", level=self.debug_level)
        if self.diag_log:
            self._diag_path = debug_dir / f"diag_client_{self.id:02d}.jsonl"

    @record_runtime("Total")
    def _optimizing(self):
        t0 = time.perf_counter()
        self._fit_surrogates()
        self._exchange_generator()
        t1 = time.perf_counter()
        pool = self._build_pool()
        if self.proposal != "centers" and (
                self.version == 1 or self.version - self._last_generator_update_ver >= max(1, self.generator_update_interval)):
            self._train_generator(pool)
            self._last_generator_update_ver = self.version
        t2 = time.perf_counter()
        prev_best, radius = float(self.y_min), float(self.tr_radius)
        next_x = self._select_by_ei(pool)
        t3 = time.perf_counter()
        next_y = float(np.asarray(self.problem.evaluate(next_x)).reshape(-1)[0])
        self._update_local_region(prev_best - next_y, prev_best)
        self._update_allocation(pool, next_y < prev_best)
        self._update_arms(pool, next_y < prev_best)
        if self._diag_path is not None:
            self._write_diagnostics(pool, next_y, prev_best, radius, {"fit_comm": t1 - t0, "pool_gen": t2 - t1, "select": t3 - t2})
        self.record_round_info("FE", self.solutions.size)
        self.record_round_info("Best", f"{self.solutions.y_min:.4f}")
        self.version += 1

    def _exchange_generator(self):
        """Upload the generator state and replace it by the server's average of all clients' generators."""
        data = {}
        if self.proposal in ("centers", "hybrid"):
            data["center"] = self._elite_center()
        if self.proposal in ("generator", "hybrid"):
            data["generator"] = self.generator.get_state()
        self.request_server(ClientPackage(cid=self.id, action=Actions.PUSH_UPDATE, version=self.version, data=data))
        payload = self.request_server(ClientPackage(cid=self.id, action=Actions.PULL_UPDATE, version=self.version),
                                      repeat=self.pull_repeat, interval=self.pull_interval) or {}
        self.generator.set_state(payload.get("generator"))
        self._centers = list(payload.get("centers") or [])

    def _elite_center(self) -> dict:
        """Mean and spread of the client's elite points (best quantile, at most generator_max_true) in its unit cube."""
        y = self.y.reshape(-1)
        k = int(min(self.generator_max_true, max(2, np.ceil(self.pseudo_top_quantile * y.size))))
        elite = to_unit_cube(self.x[np.argsort(y)[:k]], self.lb_vec, self.ub_vec)
        return {"mean": elite.mean(axis=0), "spread": np.maximum(elite.std(axis=0), 0.01)}

    # ---------------------------------------------------------------- surrogate
    def _ty(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return y if self.y_norm == "none" else (y - self._yn[0]) / self._yn[1]

    def _fit_surrogates(self):
        y = np.asarray(self.y, dtype=float).reshape(-1)
        self._yn = (float(np.mean(y)), float(np.std(y)) or 1.0)
        ty = self._ty(self.y)
        self.local_model.train(self.x, ty)
        xu, idx = np.unique(self.x, axis=0, return_index=True)
        for model in self.bootstrap_models:
            if xu.shape[0] <= 3:
                model.copy_from(self.local_model)
                continue
            b = self.rng.integers(0, xu.shape[0], size=xu.shape[0])
            model._kernel_size = 2 * self.dim + 1
            model.train(xu[b], ty[idx][b])

    def _predict(self, X: np.ndarray):
        """Local mean and bootstrap spread on the normalized scale."""
        mu = self.local_model.predict(X).reshape(-1)
        boot = np.stack([m.predict(X).reshape(-1) for m in self.bootstrap_models])
        return mu, np.std(boot, axis=0)

    def _task_embedding(self) -> np.ndarray:
        """Ranks of the local predictions on the shared reference points, the generator's task condition."""
        if self.task_embed == "const":
            return np.full(self.embed_dim, 0.5)
        rank = self.local_model.predict(self.d_share).reshape(-1).argsort().argsort().astype(float)
        return reduce_signature(minmax_normalize(rank), self.embed_dim)

    # ---------------------------------------------------------------- candidates
    def _pool_sizes(self) -> np.ndarray:
        base = np.array([self.candidate_random, self.candidate_perturb, self.candidate_generate, self.candidate_boundary], float)
        if self.allocation != "bandit":
            return base.astype(int)
        a, b = self.alloc_prior
        rate = (self._alloc_succ + a) / (self._alloc_tries + a + b)
        share = base * rate / (base * rate).sum()
        share = np.maximum(share, self.alloc_min_share * (base > 0))
        share /= share.sum()
        sizes = np.floor(share * base.sum()).astype(int)
        sizes[np.argmax(share)] += int(base.sum()) - sizes.sum()
        return sizes

    def _update_allocation(self, pool: dict, success: bool):
        self._alloc_succ *= self.alloc_decay
        self._alloc_tries *= self.alloc_decay
        if self._last_choice is not None and pool["source"][self._last_choice] < 4:
            s = int(pool["source"][self._last_choice])
            self._alloc_tries[s] += 1.0
            self._alloc_succ[s] += float(success)

    @record_runtime("Pool")
    def _build_pool(self) -> dict:
        n_uni, n_loc, n_gen, n_bnd = self._pool_sizes()
        blocks = [self.problem.random_uniform_x(n_uni).reshape(-1, self.dim),
                  self._local_candidates(n_loc),
                  self._generated_candidates(n_gen),
                  self._boundary_candidates(n_bnd)]
        arms = np.concatenate([np.full(len(blocks[0]) + len(blocks[1]), -1), self._gen_arms,
                               np.full(len(blocks[3]), -1)])
        X = np.vstack([b for b in blocks if b.size > 0])
        source = np.concatenate([np.full(len(b), i) for i, b in enumerate(blocks)])  # 0 uniform 1 local 2 gen 3 bound
        mu, unc = self._predict(X)
        return {"X": X, "source": source, "arm": arms, "mu": mu, "unc": unc}

    def _local_candidates(self, num: int) -> np.ndarray:
        if num <= 0:
            return np.empty((0, self.dim))
        span = self.ub_vec - self.lb_vec
        if self.local_region:  # uniform in a box of half-width tr_radius around the incumbent
            center = self.x[int(np.argmin(self.y.reshape(-1)))]
            return np.clip(center + self.rng.uniform(-1.0, 1.0, size=(num, self.dim)) * span * self.tr_radius,
                           self.lb_vec, self.ub_vec)
        elites = self.x[np.argsort(self.y.reshape(-1))[:5]]
        base = elites[self.rng.integers(0, len(elites), size=num)]
        return np.clip(base + self.rng.normal(scale=0.1 * span, size=(num, self.dim)), self.lb_vec, self.ub_vec)

    def _generated_candidates(self, num: int) -> np.ndarray:
        self._gen_arms = np.full(max(num, 0), -1)
        if num <= 0:
            return np.empty((0, self.dim))
        if self.proposal in ("centers", "hybrid"):
            return self._center_candidates(num)
        return self._generator_samples(num)

    def _generator_samples(self, num: int) -> np.ndarray:
        if num <= 0:
            return np.empty((0, self.dim))
        emb = self._task_embedding()
        counts = [num // len(self.q_targets) + (i < num % len(self.q_targets)) for i in range(len(self.q_targets))]
        return np.vstack([self.generator.sample(c, float(q), emb, self.lb_vec, self.ub_vec)
                          for q, c in zip(self.q_targets, counts)])

    def _center_candidates(self, num: int) -> np.ndarray:
        """Proposals around every client's elite center and around the consensus center (arm key -1 is reserved
        for non-center candidates; the consensus arm has key 0, client arms their client id)."""
        if not self._centers:
            self._gen_arms = np.full(0, -1)
            return np.empty((0, self.dim))
        arms = {int(c["cid"]): (np.asarray(c["mean"]), np.asarray(c["spread"])) for c in self._centers}
        hybrid = self.proposal == "hybrid"
        if not hybrid:
            arms[0] = (np.mean([m for m, _ in arms.values()], axis=0), np.mean([s for _, s in arms.values()], axis=0))
        keys = sorted(arms) if not hybrid else [0] + sorted(arms)
        n_cen = len(keys) - 1 if hybrid else len(keys)
        base = np.array([self.hybrid_gen_share if (hybrid and k == 0) else
                         (1.0 - self.hybrid_gen_share if hybrid else 1.0) / n_cen for k in keys])
        if self.center_alloc == "bandit":
            a, b = self.alloc_prior
            w = base * np.array([(self._arm_succ.get(k, 0.0) + a) / (self._arm_tries.get(k, 0.0) + a + b) for k in keys])
        else:
            w = base
        share = w / w.sum() * num
        counts = np.floor(share).astype(int)
        for i in np.argsort(-(share - counts))[:num - counts.sum()]:
            counts[i] += 1
        blocks, labels = [], []
        for k, n in zip(keys, counts):
            if n <= 0:
                continue
            if hybrid and k == 0:  # consensus arm: samples of the shared generator, in the unit cube
                blocks.append(to_unit_cube(self._generator_samples(n), self.lb_vec, self.ub_vec))
                labels.append(np.full(n, 0))
                continue
            mean, spread = arms[k]
            pts = np.vstack([mean, mean + self.center_scale * spread * self.rng.normal(size=(n - 1, self.dim))])
            blocks.append(np.clip(pts, 0.0, 1.0))
            labels.append(np.full(n, k))
        self._gen_arms = np.concatenate(labels)
        return self.lb_vec + np.vstack(blocks) * (self.ub_vec - self.lb_vec)

    def _update_arms(self, pool: dict, success: bool):
        for k in self._arm_tries:
            self._arm_tries[k] *= self.alloc_decay
            self._arm_succ[k] *= self.alloc_decay
        if self._last_choice is not None and pool["arm"][self._last_choice] >= 0:
            k = int(pool["arm"][self._last_choice])
            self._arm_tries[k] = self._arm_tries.get(k, 0.0) + 1.0
            self._arm_succ[k] = self._arm_succ.get(k, 0.0) + float(success)

    def _boundary_candidates(self, num: int) -> np.ndarray:
        if num <= 0:
            return np.empty((0, self.dim))
        inc = self.x[int(np.argmin(self.y.reshape(-1)))]
        proj = np.repeat(inc[None, :], 2 * self.dim, axis=0)
        for j in range(self.dim):
            proj[2 * j, j], proj[2 * j + 1, j] = self.lb_vec[j], self.ub_vec[j]
        extra = max(0, num - len(proj))
        rand = self.problem.random_uniform_x(extra).reshape(-1, self.dim) if extra else np.empty((0, self.dim))
        if extra:
            cols, side = self.rng.integers(0, self.dim, size=extra), self.rng.integers(0, 2, size=extra)
            rand[np.arange(extra), cols] = np.where(side == 0, self.lb_vec[cols], self.ub_vec[cols])
        return np.vstack([proj, rand])[:num]

    # ---------------------------------------------------------------- generator and selection
    @record_runtime("Gen")
    def _train_generator(self, pool: dict):
        y = self.y.reshape(-1)
        q_true = empirical_quantiles(y)
        true_idx = np.where(q_true <= max(self.pseudo_top_quantile, min(self.q_targets)))[0]
        true_idx = true_idx[np.argsort(y[true_idx])[:self.generator_max_true]]
        q_pool = empirical_quantiles(pool["mu"] - self.acq_lambda * minmax_normalize(pool["unc"]))
        pseudo_idx = np.where(q_pool <= self.pseudo_top_quantile)[0]
        pseudo_idx = pseudo_idx[np.argsort(q_pool[pseudo_idx])[:self.generator_max_pseudo]]
        X = np.vstack([self.x[true_idx], pool["X"][pseudo_idx]])
        if X.size == 0:
            return
        q = np.concatenate([q_true[true_idx], q_pool[pseudo_idx]])
        self.generator.fit(to_unit_cube(X, self.lb_vec, self.ub_vec), q, self._task_embedding(), np.ones(len(X)))

    @record_runtime("Select")
    def _select_by_ei(self, pool: dict) -> np.ndarray:
        mu, sigma = pool["mu"], np.maximum(pool["unc"], 1e-12)
        best = float(self._ty(np.array([self.y_min])).reshape(-1)[0])
        z = (best - mu) / sigma
        pool["ei"] = (best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        for idx in np.argsort(-pool["ei"]):
            if not np.any(np.all(np.isclose(self.x, pool["X"][idx]), axis=1)):
                self._last_choice = int(idx)
                return pool["X"][idx].copy()
        self._last_choice = None
        return self.problem.random_uniform_x(1).reshape(-1)

    def _update_local_region(self, improvement: float, prev_best: float):
        if not self.local_region:
            return
        if improvement > 1e-3 * max(abs(prev_best), 1e-12):
            self._tr_succ, self._tr_fail = self._tr_succ + 1, 0
        else:
            self._tr_succ, self._tr_fail = 0, self._tr_fail + 1
        if self._tr_succ >= self.tr_succ_tol:
            self.tr_radius, self._tr_succ = min(2.0 * self.tr_radius, self.tr_max), 0
        elif self._tr_fail >= self.tr_fail_tol:
            self.tr_radius, self._tr_fail = 0.5 * self.tr_radius, 0
            if self.tr_radius < self.tr_min:  # region collapsed: restart from the initial radius
                self.tr_radius = self.tr_init

    def _write_diagnostics(self, pool: dict, y: float, prev_best: float, radius: float, timing: dict):
        rec = {"ver": int(self.version), "fe": int(self.solutions.size), "y": y, "best_before": prev_best,
               "improved": bool(y < prev_best), "tr_radius": radius, "pool_size": int(len(pool["X"])),
               "pool_sources": np.bincount(pool["source"], minlength=4).tolist(),
               "timing": {k: round(v, 4) for k, v in timing.items()}}
        i = self._last_choice
        if i is not None:
            rec["chosen"] = {"source": int(pool["source"][i]), "arm": int(pool["arm"][i]), "mu": float(pool["mu"][i]),
                             "unc": float(pool["unc"][i]), "ei": float(pool["ei"][i])}
        with self._diag_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    @property
    def x(self):
        return self.solutions.x

    @property
    def y(self):
        return self.solutions.y

    @property
    def y_min(self):
        return self.solutions.y_min

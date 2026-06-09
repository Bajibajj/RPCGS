from __future__ import annotations

from pathlib import Path

import numpy as np
from .framework import Client, record_runtime
from .rbf import RadialBasisFunctionNetwork as RBFNetwork
from .utils import (
    Actions,
    ClientPackage,
    MetaFeatureSelector,
    TrustConditionedGenerator,
    build_candidate_meta_features,
    build_pairwise_selector_data,
    create_trainable_rbf,
    empirical_quantiles,
    minmax_normalize,
    progress_to_signature,
    reduce_signature,
    to_unit_cube,
    RunFileLogger,
)


class RpcgsClient(Client):
    """
    rbf_epoch: 5
    rbf_optimizer: sgd
    rbf_lr: 0.06
    rbf_alpha: 1.0
    candidate_random: 64
    candidate_perturb: 32
    candidate_generate: 64
    pseudo_top_quantile: 0.2
    trust_alpha_u: 2.0
    trust_alpha_d: 2.0
    trust_min: 0.05
    use_bootstrap: true
    bootstrap_num: 3
    perturb_sigma: 0.1
    embed_dim: 8
    generator_latent: 8
    generator_hidden: 64
    generator_epochs: 50
    generator_lr: 0.001
    generator_batch_size: 32
    generator_backend: auto
    selector_hidden: 64
    selector_epochs: 50
    selector_lr: 0.001
    selector_batch_size: 32
    selector_backend: auto
    selector_pool: 64
    selector_duels: 12
    selector_pairs: 2048
    history_window: 32
    generator_update_interval: 5
    selector_update_interval: 3
    generator_max_true: 16
    generator_max_pseudo: 24
    q_targets: [0.2, 0.1, 0.05]
    acq_lambda: 2.0
    acq_beta: 0.5
    acq_gamma: 0.5
    seed: 123
    debug_level: DEBUG
    """

    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        self.problem.auto_update_solutions = True

        self.rbf_epoch = int(kwargs.get("rbf_epoch", 5))
        self.rbf_optimizer = kwargs.get("rbf_optimizer", "sgd")
        self.rbf_lr = float(kwargs.get("rbf_lr", 0.06))
        self.rbf_alpha = float(kwargs.get("rbf_alpha", 1.0))
        self.candidate_random = int(kwargs.get("candidate_random", 64))
        self.candidate_perturb = int(kwargs.get("candidate_perturb", 32))
        self.candidate_generate = int(kwargs.get("candidate_generate", 64))
        self.pseudo_top_quantile = float(kwargs.get("pseudo_top_quantile", 0.2))
        self.trust_alpha_u = float(kwargs.get("trust_alpha_u", 2.0))
        self.trust_alpha_d = float(kwargs.get("trust_alpha_d", 2.0))
        self.trust_min = float(kwargs.get("trust_min", 0.05))
        self.use_bootstrap = bool(kwargs.get("use_bootstrap", True))
        self.bootstrap_num = int(kwargs.get("bootstrap_num", 3))
        self.perturb_sigma = float(kwargs.get("perturb_sigma", 0.1))
        self.embed_dim = int(kwargs.get("embed_dim", 8))
        self.generator_latent = int(kwargs.get("generator_latent", 8))
        self.generator_hidden = int(kwargs.get("generator_hidden", 64))
        self.generator_epochs = int(kwargs.get("generator_epochs", 50))
        self.generator_lr = float(kwargs.get("generator_lr", 1e-3))
        self.generator_batch_size = int(kwargs.get("generator_batch_size", 32))
        self.generator_backend = kwargs.get("generator_backend", "auto")
        self.selector_hidden = int(kwargs.get("selector_hidden", 64))
        self.selector_epochs = int(kwargs.get("selector_epochs", 50))
        self.selector_lr = float(kwargs.get("selector_lr", 1e-3))
        self.selector_batch_size = int(kwargs.get("selector_batch_size", 32))
        self.selector_backend = kwargs.get("selector_backend", "auto")
        self.selector_pool = int(kwargs.get("selector_pool", 64))
        self.selector_duels = int(kwargs.get("selector_duels", 12))
        self.selector_pairs = int(kwargs.get("selector_pairs", 2048))
        self.history_window = int(kwargs.get("history_window", 32))
        self.generator_update_interval = int(kwargs.get("generator_update_interval", 5))
        self.selector_update_interval = int(kwargs.get("selector_update_interval", 3))
        self.generator_max_true = int(kwargs.get("generator_max_true", 16))
        self.generator_max_pseudo = int(kwargs.get("generator_max_pseudo", 24))
        self.q_targets = kwargs.get("q_targets", [0.2, 0.1, 0.05])
        self.acq_lambda = float(kwargs.get("acq_lambda", 2.0))
        self.acq_beta = float(kwargs.get("acq_beta", 0.5))
        self.acq_gamma = float(kwargs.get("acq_gamma", 0.5))
        self.seed = kwargs.get("seed", None)
        self.debug_level = str(kwargs.get("debug_level", "DEBUG")).upper()
        self.pull_repeat = int(kwargs.get("pull_repeat", 300))
        self.pull_interval = float(kwargs.get("pull_interval", 1.0))

        self.rng = np.random.default_rng(self.seed)
        self.lb_vec = self.lb if isinstance(self.lb, np.ndarray) else np.ones(self.dim) * self.lb
        self.ub_vec = self.ub if isinstance(self.ub, np.ndarray) else np.ones(self.dim) * self.ub

        self.local_model: RBFNetwork = create_trainable_rbf(
            self.dim, self.obj,
            epoch=self.rbf_epoch, optimizer=self.rbf_optimizer, lr=self.rbf_lr, alpha=self.rbf_alpha, rng=self.rng
        )
        self.transfer_model: RBFNetwork = RBFNetwork(dim=self.dim, obj=self.obj, kernel_size=2 * self.dim + 1)
        self.bootstrap_models: list[RBFNetwork] = [
            create_trainable_rbf(
                self.dim, self.obj,
                epoch=self.rbf_epoch, optimizer=self.rbf_optimizer, lr=self.rbf_lr, alpha=self.rbf_alpha, rng=self.rng
            ) for _ in range(max(1, self.bootstrap_num))
        ]
        self.generator = TrustConditionedGenerator(
            dim=self.dim,
            embed_dim=self.embed_dim,
            latent_dim=self.generator_latent,
            hidden_dim=self.generator_hidden,
            lr=self.generator_lr,
            epochs=self.generator_epochs,
            batch_size=self.generator_batch_size,
            backend=self.generator_backend,
            seed=self.seed,
        )
        self.selector = MetaFeatureSelector(
            feat_dim=7,
            hidden_dim=self.selector_hidden,
            lr=self.selector_lr,
            epochs=self.selector_epochs,
            batch_size=self.selector_batch_size,
            backend=self.selector_backend,
            seed=self.seed,
        )

        self.version = 1
        self.d_share = None
        self.transfer_payload = None
        self.progress_history: list[float] = []
        self.file_logger = None
        self._last_generator_update_ver = 0
        self._last_selector_update_ver = 0

    def optimize(self):
        if self.d_share is None:
            self._initializing()
        else:
            self._optimizing()

    def _initializing(self):
        pkg = ClientPackage(
            cid=self.id,
            action=Actions.PUSH_INIT,
            data={"dim": self.dim, "bound": (self.lb_vec, self.ub_vec), "problem_name": self.problem.name},
        )
        self.request_server(pkg)
        resp = self.request_server(
            ClientPackage(cid=self.id, action=Actions.PULL_INIT),
            repeat=100,
            interval=0.1,
        )
        if resp is None:
            return
        self.d_share = np.asarray(resp["d_share"])
        if self.file_logger is None:
            debug_dir = resp.get("debug_dir")
            log_name = f"client_{self.id:02d}_{self.problem.name}.log"
            self.file_logger = RunFileLogger(Path(debug_dir) / log_name, level=self.debug_level)
            self.file_logger.info(f"client init dim={self.dim} fe_init={self.fe_init} fe_max={self.fe_max}")

    @record_runtime("Total")
    def _optimizing(self):
        self._fit_surrogates()
        self.push()
        self.pull()
        self._sync_transfer_payload()

        pseudo = self._build_pseudo_pool()
        if self._should_update_generator():
            self._train_generator(pseudo)
            self._last_generator_update_ver = self.version
        if self._should_update_selector():
            self._train_selector(pseudo)
            self._last_selector_update_ver = self.version
        prev_best = float(self.y_min)
        next_x = self._select_next_candidate(pseudo)
        next_y = self.problem.evaluate(next_x)
        improvement = prev_best - float(np.min(np.asarray(next_y).reshape(-1)))
        self.progress_history.append(improvement)
        self.progress_history = self.progress_history[-4:]
        self.record_round_info("FE", self.solutions.size)
        self.record_round_info("Best", f"{self.solutions.y_min:.4f}")
        self.record_round_info("Trust", f"{float(np.mean(pseudo['trust'])):.3f}")
        self.record_round_info("Pseudo", int(np.sum(pseudo["positive_mask"])))
        if self.file_logger is not None:
            self.file_logger.info(
                f"ver={self.version} fe={self.solutions.size} best={self.solutions.y_min:.6f} "
                f"trust_mean={float(np.mean(pseudo['trust'])):.4f} pseudo_pos={int(np.sum(pseudo['positive_mask']))} "
                f"improvement={improvement:.6f} "
                f"gen_upd={self._last_generator_update_ver == self.version} "
                f"sel_upd={self._last_selector_update_ver == self.version}"
            )
        self.version += 1

    def _should_update_generator(self) -> bool:
        interval = max(1, self.generator_update_interval)
        return self.version == 1 or (self.version - self._last_generator_update_ver) >= interval

    def _should_update_selector(self) -> bool:
        interval = max(1, self.selector_update_interval)
        return self.version == 1 or (self.version - self._last_selector_update_ver) >= interval

    def _fit_surrogates(self):
        self.local_model.train(self.x, self.y)
        if not self.transfer_model.predictable:
            self.transfer_model.sync_auto(self.local_model.params)
        if self.use_bootstrap:
            for boot_model in self.bootstrap_models:
                try:
                    xu, idx = np.unique(self.x, axis=0, return_index=True)
                    yu = self.y[idx]
                    if xu.shape[0] <= 3:
                        boot_model.sync_auto(self.local_model.params)
                        continue
                    bidx = self.rng.integers(0, xu.shape[0], size=xu.shape[0])
                    xb, yb = xu[bidx], yu[bidx]
                    # dynamic kernel reduction to avoid repeated-cluster failures
                    kernel_size = max(2, min(2 * self.dim + 1, xb.shape[0] - 1))
                    boot_model._kernel_size = kernel_size
                    boot_model.train(xb, yb)
                except Exception:
                    boot_model.sync_auto(self.local_model.params)
                    if self.file_logger is not None:
                        self.file_logger.warning("bootstrap train failed, fallback to local params")
        else:
            for boot_model in self.bootstrap_models:
                boot_model.sync_auto(self.local_model.params)

    @record_runtime("Push")
    def push(self):
        data = {
            "network": self.local_model.params,
            "generator": self.generator.get_transfer_state(),
            "selector": self.selector.get_state(),
            "rank_signature": self._rank_signature(),
            "unc_signature": self._unc_signature(),
            "progress_signature": progress_to_signature(self.progress_history, 4),
        }
        pkg = ClientPackage(cid=self.id, action=Actions.PUSH_UPDATE, version=self.version, data=data)
        self.request_server(pkg)
        if self.file_logger is not None:
            self.file_logger.debug(f"push ver={self.version}")

    @record_runtime("Pull")
    def pull(self):
        pkg = ClientPackage(cid=self.id, action=Actions.PULL_UPDATE, version=self.version)
        self.transfer_payload = self.request_server(
            pkg,
            repeat=self.pull_repeat,
            interval=self.pull_interval,
            msg=f"Pull RPCGS payload for version {self.version}",
        )
        if self.file_logger is not None:
            if self.transfer_payload is None:
                self.file_logger.error(f"pull failed ver={self.version}")
            else:
                self.file_logger.debug(f"pull success ver={self.version}")

    def check_pkg(self, pkg) -> bool:
        return pkg is not None

    def _sync_transfer_payload(self):
        payload = self.transfer_payload or {}
        network = payload.get("network")
        if network:
            self.transfer_model.sync_auto(network)
        else:
            self.transfer_model.sync_auto(self.local_model.params)
        self.generator.set_transfer_state(payload.get("generator"))
        self.selector.set_state(payload.get("selector"))
        if self.file_logger is not None:
            self.file_logger.debug(
                f"sync_transfer neighbors={payload.get('neighbors', [])} "
                f"weights={np.round(payload.get('weights', []), 4) if payload.get('weights') else []}"
            )

    def _rank_signature(self) -> np.ndarray:
        pred = self.local_model.predict(self.d_share).reshape(-1)
        rank = pred.argsort().argsort().astype(float)
        return minmax_normalize(rank)

    def _unc_signature(self) -> np.ndarray:
        if self.bootstrap_models and self.bootstrap_models[0].predictable:
            preds = [m.predict(self.d_share).reshape(-1) for m in self.bootstrap_models]
            unc = np.std(np.stack(preds, axis=0), axis=0)
        else:
            local = self.local_model.predict(self.d_share).reshape(-1)
            transfer = self.transfer_model.predict(self.d_share).reshape(-1) if self.transfer_model.predictable else local
            unc = np.abs(local - transfer)
        return minmax_normalize(unc)

    def _task_embedding(self) -> np.ndarray:
        return reduce_signature(self._rank_signature(), self.embed_dim)

    def _perturb_top_points(self, num: int) -> np.ndarray:
        if num <= 0 or self.solutions.size == 0:
            return np.empty((0, self.dim))
        y = self.y.reshape(-1)
        x = self.x
        best_idx = np.argsort(y)[:max(1, min(5, y.size))]
        base = x[best_idx]
        samples = []
        scale = (self.ub_vec - self.lb_vec) * self.perturb_sigma
        for _ in range(num):
            center = base[self.rng.integers(0, base.shape[0])]
            cand = center + self.rng.normal(scale=scale, size=self.dim)
            cand = np.clip(cand, self.lb_vec, self.ub_vec)
            samples.append(cand)
        return np.asarray(samples)

    def _generated_candidates(self, num: int) -> np.ndarray:
        if num <= 0:
            return np.empty((0, self.dim))
        emb = self._task_embedding()
        groups = max(1, len(self.q_targets))
        counts = [num // groups for _ in range(groups)]
        for i in range(num % groups):
            counts[i] += 1
        blocks = []
        for q_target, cnt in zip(self.q_targets, counts):
            blocks.append(self.generator.sample(cnt, float(q_target), emb, self.lb_vec, self.ub_vec))
        return np.vstack([b for b in blocks if b.size > 0]) if blocks else np.empty((0, self.dim))

    def _predict_pool(self, X: np.ndarray):
        local = self.local_model.predict(X).reshape(-1)
        transfer = self.transfer_model.predict(X).reshape(-1) if self.transfer_model.predictable else local.copy()
        if self.bootstrap_models and self.bootstrap_models[0].predictable:
            bpred = np.stack([m.predict(X).reshape(-1) for m in self.bootstrap_models], axis=0)
            unc = np.std(bpred, axis=0)
        else:
            unc = np.abs(local - transfer)
        disagree = np.abs(local - transfer)
        mu = 0.5 * (local + transfer)
        return mu, unc, disagree

    @record_runtime("Pseudo")
    def _build_pseudo_pool(self) -> dict:
        blocks = [
            self.problem.random_uniform_x(self.candidate_random).reshape(-1, self.dim),
            self._perturb_top_points(self.candidate_perturb),
            self._generated_candidates(self.candidate_generate),
        ]
        X = np.vstack([b for b in blocks if b.size > 0])
        mu, unc, disagree = self._predict_pool(X)
        unc_n = minmax_normalize(unc)
        dis_n = minmax_normalize(disagree)
        trust = np.exp(-self.trust_alpha_u * unc_n - self.trust_alpha_d * dis_n)
        trust = np.clip(trust, self.trust_min, 1.0)
        quality = mu - self.acq_lambda * unc_n
        q_hat = empirical_quantiles(quality)
        positive_mask = (q_hat <= self.pseudo_top_quantile) & (trust >= self.trust_min)
        novelty = self._novelty_score(X)
        meta = build_candidate_meta_features(mu, unc, disagree, trust, novelty, q_hat)
        base_score = (
            0.35 * meta[:, 0] +
            0.20 * meta[:, 3] +
            0.15 * meta[:, 1] +
            0.10 * meta[:, 2] +
            0.10 * meta[:, 4] +
            0.10 * meta[:, 5]
        )
        return {
            "X": X,
            "mu": mu,
            "unc": unc,
            "unc_n": unc_n,
            "disagree": disagree,
            "trust": trust,
            "q_hat": q_hat,
            "positive_mask": positive_mask,
            "novelty": novelty,
            "meta": meta,
            "base_score": base_score,
        }

    def _novelty_score(self, X: np.ndarray) -> np.ndarray:
        if self.solutions.size == 0:
            return np.ones(X.shape[0], dtype=float)
        diff = X[:, None, :] - self.x[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist[:, np.argmin(dist, axis=1)] = np.max(dist, axis=1)
        min_other = np.min(dist, axis=1)
        return minmax_normalize(min_other)

    @record_runtime("Gen")
    def _train_generator(self, pseudo: dict):
        X_true = self.x
        y_true = self.y.reshape(-1)
        q_true = empirical_quantiles(y_true)
        true_mask = q_true <= max(self.pseudo_top_quantile, min(self.q_targets))
        X_pseudo = pseudo["X"][pseudo["positive_mask"]]
        q_pseudo = pseudo["q_hat"][pseudo["positive_mask"]]
        w_pseudo = pseudo["trust"][pseudo["positive_mask"]]

        if np.any(true_mask) and np.sum(true_mask) > self.generator_max_true:
            true_idx = np.where(true_mask)[0]
            pick = true_idx[np.argsort(y_true[true_idx])[:self.generator_max_true]]
            true_mask = np.zeros_like(true_mask, dtype=bool)
            true_mask[pick] = True
        if X_pseudo.size > 0 and X_pseudo.shape[0] > self.generator_max_pseudo:
            pseudo_idx = np.argsort(-w_pseudo)[:self.generator_max_pseudo]
            X_pseudo = X_pseudo[pseudo_idx]
            q_pseudo = q_pseudo[pseudo_idx]
            w_pseudo = w_pseudo[pseudo_idx]

        X_train = []
        q_train = []
        w_train = []
        if np.any(true_mask):
            X_train.append(X_true[true_mask])
            q_train.append(q_true[true_mask])
            w_train.append(np.ones(np.sum(true_mask), dtype=float))
        if X_pseudo.size > 0:
            X_train.append(X_pseudo)
            q_train.append(q_pseudo)
            w_train.append(w_pseudo)
        if not X_train:
            return
        X_all = np.vstack(X_train)
        q_all = np.concatenate(q_train)
        w_all = np.concatenate(w_train)
        X_unit = to_unit_cube(X_all, self.lb_vec, self.ub_vec)
        self.generator.fit(X_unit, q_all, self._task_embedding(), w_all)
        if self.file_logger is not None:
            self.file_logger.debug(
                f"train_generator samples={X_all.shape[0]} true_pos={int(np.sum(true_mask))} "
                f"pseudo_pos={X_pseudo.shape[0] if X_pseudo.size > 0 else 0}"
            )

    @record_runtime("SelectTrain")
    def _train_selector(self, pseudo: dict):
        X_hist = self.x[-self.history_window:]
        y_hist = self.y.reshape(-1)[-self.history_window:]
        if X_hist.shape[0] < 4:
            return
        mu_h, unc_h, dis_h = self._predict_pool(X_hist)
        trust_h = np.exp(-self.trust_alpha_u * minmax_normalize(unc_h) - self.trust_alpha_d * minmax_normalize(dis_h))
        novelty_h = self._novelty_score(X_hist)
        qh = empirical_quantiles(mu_h - self.acq_lambda * minmax_normalize(unc_h))
        feats = build_candidate_meta_features(mu_h, unc_h, dis_h, trust_h, novelty_h, qh)
        pair_x, pair_y = build_pairwise_selector_data(feats, y_hist, max_pairs=self.selector_pairs, rng=self.rng)
        self.selector.fit(pair_x, pair_y)
        if self.file_logger is not None:
            self.file_logger.debug(f"train_selector pairs={pair_x.shape[0]} hist={X_hist.shape[0]}")

    @record_runtime("Select")
    def _select_next_candidate(self, pseudo: dict) -> np.ndarray:
        order = np.argsort(-pseudo["base_score"])
        candidate_idx = []
        for idx in order:
            if pseudo["trust"][idx] < self.trust_min:
                continue
            cand = pseudo["X"][idx]
            if not self._is_duplicate(cand):
                candidate_idx.append(int(idx))
            if len(candidate_idx) >= self.selector_pool:
                break
        if not candidate_idx:
            if self.file_logger is not None:
                self.file_logger.warning("no candidate survived trust gate, fallback to random")
            return self.problem.random_uniform_x(1).reshape(-1)
        chosen = self.selector.tournament_select(
            features=pseudo["meta"],
            candidate_idx=np.asarray(candidate_idx, dtype=int),
            duels=self.selector_duels,
            rng=self.rng,
        )
        if self.file_logger is not None:
            self.file_logger.debug(
                f"select idx={chosen} base={float(pseudo['base_score'][chosen]):.6f} "
                f"mu={float(pseudo['mu'][chosen]):.6f} unc={float(pseudo['unc_n'][chosen]):.6f} "
                f"trust={float(pseudo['trust'][chosen]):.6f}"
            )
        return pseudo["X"][chosen].reshape(-1)

    def _is_duplicate(self, x_new: np.ndarray) -> bool:
        if self.solutions.size == 0:
            return False
        x_new = x_new.reshape(1, -1)
        return bool(np.any(np.all(np.isclose(self.x, x_new), axis=1)))

    @property
    def x(self):
        return self.solutions.x

    @property
    def y(self):
        return self.solutions.y

    @property
    def y_min(self):
        return self.solutions.y_min

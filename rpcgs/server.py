from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from .framework import Server, SyncDataManager
from .utils import (
    Actions,
    ClientPackage,
    aggregate_personalized_payloads_v2,
    build_total_distance_matrix,
    ema_blend,
    init_samples,
    RunFileLogger,
    make_run_tag,
)


class RpcgsServer(Server):
    """
    d_share_size: 32
    top_k: 3
    tau: 1.0
    lambda_rank: 0.6
    lambda_unc: 0.3
    lambda_prog: 0.1
    ema_decay: 0.7
    aggregate_generator: true
    aggregate_selector: true
    debug_dir: out/rpcgs-debug
    debug_level: DEBUG
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.d_share_size = int(kwargs.get("d_share_size", 32))
        self.top_k = int(kwargs.get("top_k", 3))
        self.tau = float(kwargs.get("tau", 1.0))
        self.lambda_rank = float(kwargs.get("lambda_rank", 0.6))
        self.lambda_unc = float(kwargs.get("lambda_unc", 0.3))
        self.lambda_prog = float(kwargs.get("lambda_prog", 0.1))
        self.ema_decay = float(kwargs.get("ema_decay", 0.7))
        self.aggregate_generator = bool(kwargs.get("aggregate_generator", True))
        self.aggregate_selector = bool(kwargs.get("aggregate_selector", True))
        self.debug_root = Path(kwargs.get("debug_dir", "out/rpcgs-debug"))
        self.debug_level = str(kwargs.get("debug_level", "DEBUG")).upper()
        self.run_tag = make_run_tag()
        self.session_dir = self.debug_root / self.run_tag
        self.file_logger = RunFileLogger(self.session_dir / "server.log", level=self.debug_level)

        self.clients_data = SyncDataManager()
        self.client_bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.signature_cache: dict[int, dict[str, np.ndarray]] = {}
        self.d_share = None
        self.dim = None
        self.done_vers: set[int] = {-1}
        self.problem_name = "unknown"
        self.file_logger.info(f"RPCGS server logger initialized at {self.session_dir}")

    def handle_request(self, pkg: ClientPackage) -> Any:
        if pkg.action == Actions.PUSH_INIT:
            return self._handle_push_init(pkg)
        if pkg.action == Actions.PULL_INIT:
            return self._handle_pull_init()
        if pkg.action == Actions.PUSH_UPDATE:
            return self._handle_push_update(pkg)
        if pkg.action == Actions.PULL_UPDATE:
            return self._handle_pull_update(pkg)
        raise ValueError(f"Unknown action: {pkg.action}")

    def _handle_push_init(self, pkg: ClientPackage):
        data = pkg.data or {}
        self.dim = int(data["dim"])
        self.problem_name = str(data.get("problem_name", self.problem_name))
        lb, ub = data["bound"]
        self.client_bounds[pkg.cid] = (np.asarray(lb, dtype=float), np.asarray(ub, dtype=float))
        self._create_d_share_if_ready()
        self.file_logger.info(f"push_init cid={pkg.cid} problem={self.problem_name} dim={self.dim}")
        return "success"

    def _handle_pull_init(self):
        if self.d_share is None:
            return None
        return {"d_share": self.d_share, "run_tag": self.run_tag, "debug_dir": str(self.session_dir)}

    def _handle_push_update(self, pkg: ClientPackage):
        self.clients_data.update_src(pkg.cid, pkg.version, pkg)
        self.file_logger.debug(f"push_update cid={pkg.cid} ver={pkg.version}")
        return "success"

    def _handle_pull_update(self, pkg: ClientPackage):
        return self.clients_data.get_res(pkg.cid, pkg.version)

    def _create_d_share_if_ready(self):
        if self.d_share is not None:
            return
        if self.num_clients <= 0 or len(self.client_bounds) < self.num_clients:
            return
        x_lb = None
        x_ub = None
        for lb, ub in self.client_bounds.values():
            x_lb = lb if x_lb is None else np.minimum(x_lb, lb)
            x_ub = ub if x_ub is None else np.maximum(x_ub, ub)
        self.d_share = init_samples(lb=x_lb, ub=x_ub, dim=self.dim, size=self.d_share_size)

    @property
    def should_agg(self) -> bool:
        clients_ready = self.clients_data.num_clients == self.num_clients
        ver_ready = self.clients_data.available_src_ver not in self.done_vers
        return clients_ready and ver_ready

    def aggregate(self):
        if not self.should_agg:
            return
        version = self.clients_data.available_src_ver
        self.file_logger.info(f"aggregate start version={version} num_clients={self.num_clients}")
        signatures = {"rank": [], "unc": [], "prog": []}
        networks: list[dict] = []
        generators: list[dict | None] = []
        selectors: list[dict | None] = []

        for cid in self.sorted_ids:
            pkg: ClientPackage = self.clients_data.get_src(cid, version)
            data = pkg.data or {}
            current_signatures = {
                "rank": np.asarray(data.get("rank_signature", np.zeros(self.d_share_size)), dtype=float).reshape(-1),
                "unc": np.asarray(data.get("unc_signature", np.zeros(self.d_share_size)), dtype=float).reshape(-1),
                "prog": np.asarray(data.get("progress_signature", np.zeros(4)), dtype=float).reshape(-1),
            }
            if cid in self.signature_cache:
                cached = self.signature_cache[cid]
                current_signatures = {
                    name: ema_blend(cached[name], current_signatures[name], self.ema_decay)
                    for name in current_signatures
                }
            self.signature_cache[cid] = current_signatures
            signatures["rank"].append(current_signatures["rank"])
            signatures["unc"].append(current_signatures["unc"])
            signatures["prog"].append(current_signatures["prog"])
            networks.append(data.get("network", {}))
            generators.append(data.get("generator"))
            selectors.append(data.get("selector"))

        signature_mat = {name: np.stack(values, axis=0) for name, values in signatures.items()}
        distances = build_total_distance_matrix(
            signatures=signature_mat,
            lambda_rank=self.lambda_rank,
            lambda_unc=self.lambda_unc,
            lambda_prog=self.lambda_prog,
        )
        net_payloads, gen_payloads, sel_payloads, neighbors, weights = aggregate_personalized_payloads_v2(
            distances=distances,
            networks=networks,
            generators=generators if self.aggregate_generator else [None for _ in generators],
            selectors=selectors if self.aggregate_selector else [None for _ in selectors],
            top_k=self.top_k,
            tau=self.tau,
        )

        for idx, cid in enumerate(self.sorted_ids):
            self.clients_data.update_res(
                cid=cid,
                version=version,
                data={
                    "network": net_payloads[idx],
                    "generator": gen_payloads[idx],
                    "selector": sel_payloads[idx],
                    "neighbors": neighbors[idx],
                    "weights": weights[idx],
                    "distance_row": distances[idx],
                },
            )
            self.file_logger.debug(
                f"aggregate result cid={cid} ver={version} neighbors={neighbors[idx]} weights={np.round(weights[idx], 4)}"
            )
        self.done_vers.add(version)
        self.file_logger.info(f"aggregate done version={version}")

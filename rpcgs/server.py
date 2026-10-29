from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from .framework import Server, SyncDataManager

from .utils import Actions, ClientPackage, RunFileLogger, average_generator_states, init_samples, make_run_tag


class RpcgsServer(Server):
    """
    d_share_size: 50
    collaborate: true
    debug_dir: out/rpcgs-debug
    debug_level: DEBUG
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.d_share_size = int(kwargs.get("d_share_size", 50))
        # collaborate: every client receives the equal-weight average of all clients' generator states;
        # false is the no-collaboration ablation (each client keeps its own generator).
        self.collaborate = bool(kwargs.get("collaborate", True))
        self.debug_root = Path(kwargs.get("debug_dir", "out/rpcgs-debug"))
        self.debug_level = str(kwargs.get("debug_level", "DEBUG")).upper()
        self.run_tag = make_run_tag()
        self.session_dir = self.debug_root / self.run_tag
        self.file_logger = RunFileLogger(self.session_dir / "server.log", level=self.debug_level)
        self.clients_data = SyncDataManager()
        self.client_bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.d_share = None
        self.dim = None
        self.done_vers: set[int] = {-1}

    def handle_request(self, pkg: ClientPackage) -> Any:
        if pkg.action == Actions.PUSH_INIT:
            data = pkg.data or {}
            self.dim = int(data["dim"])
            lb, ub = data["bound"]
            self.client_bounds[pkg.cid] = (np.asarray(lb, dtype=float), np.asarray(ub, dtype=float))
            self._create_d_share_if_ready()
            return "success"
        if pkg.action == Actions.PULL_INIT:
            if self.d_share is None:
                return None
            return {"d_share": self.d_share, "debug_dir": str(self.session_dir)}
        if pkg.action == Actions.PUSH_UPDATE:
            self.clients_data.update_src(pkg.cid, pkg.version, pkg)
            return "success"
        if pkg.action == Actions.PULL_UPDATE:
            return self.clients_data.get_res(pkg.cid, pkg.version)
        raise ValueError(f"Unknown action: {pkg.action}")

    def _create_d_share_if_ready(self):
        """Shared reference points (Latin hypercube over the union of the client boxes) for task embeddings."""
        if self.d_share is not None or self.num_clients <= 0 or len(self.client_bounds) < self.num_clients:
            return
        lbs, ubs = zip(*self.client_bounds.values())
        self.d_share = init_samples(lb=np.min(lbs, axis=0), ub=np.max(ubs, axis=0), dim=self.dim, size=self.d_share_size)

    @property
    def should_agg(self) -> bool:
        return (self.clients_data.num_clients == self.num_clients
                and self.clients_data.available_src_ver not in self.done_vers)

    def aggregate(self):
        if not self.should_agg:
            return
        version = self.clients_data.available_src_ver
        uploads = [self.clients_data.get_src(cid, version).data or {} for cid in self.sorted_ids]
        states = [u.get("generator") for u in uploads]
        shared = average_generator_states(states, np.full(len(states), 1.0 / len(states))) if self.collaborate else None
        # elite centers (proposal: centers) are forwarded unchanged with the owner's id; without collaboration a
        # client only gets its own center back
        centers = [dict(u["center"], cid=int(cid)) for cid, u in zip(self.sorted_ids, uploads) if u.get("center")]
        for cid in self.sorted_ids:
            mine = centers if self.collaborate else [c for c in centers if c["cid"] == int(cid)]
            self.clients_data.update_res(cid=cid, version=version, data={"generator": shared, "centers": mine})
        self.done_vers.add(version)
        self.file_logger.info(f"aggregate version={version} shared={'yes' if shared is not None else 'no'}")

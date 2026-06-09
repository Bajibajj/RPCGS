from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpcgs import RpcgsClient, RpcgsServer


class SolutionArchive:
    def __init__(self):
        self.x = np.empty((0, 0), dtype=float)
        self.y = np.empty((0, 1), dtype=float)
        self.num_updated = 0

    def add(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if self.x.size == 0:
            self.x = x.copy()
        else:
            self.x = np.vstack([self.x, x])
        self.y = np.vstack([self.y, y])
        self.num_updated = x.shape[0]

    @property
    def size(self):
        return self.y.shape[0]

    @property
    def y_min(self):
        return float(np.min(self.y))

    @property
    def y_max(self):
        return float(np.max(self.y))


class ShiftedSphereProblem:
    def __init__(self, cid: int, shift: np.ndarray, dim: int = 3, fe_init: int = 12, fe_max: int = 30, seed: int = 0):
        self.id = cid
        self.name = f"shifted_sphere_{cid}"
        self.dim = dim
        self.obj = 1
        self.fe_init = fe_init
        self.fe_max = fe_max
        self.lb = -5.0 * np.ones(dim)
        self.ub = 5.0 * np.ones(dim)
        self.shift = np.asarray(shift, dtype=float).reshape(1, -1)
        self.rng = np.random.default_rng(seed)
        self.auto_update_solutions = True
        self.solutions = SolutionArchive()
        self.evaluate(self.random_uniform_x(fe_init))

    @property
    def fe_available(self):
        return max(0, self.fe_max - self.solutions.size)

    def random_uniform_x(self, n: int):
        return self.rng.uniform(self.lb, self.ub, size=(int(n), self.dim))

    def evaluate(self, x: np.ndarray):
        x = np.asarray(x, dtype=float).reshape(-1, self.dim)
        y = np.sum((x - self.shift) ** 2, axis=1, keepdims=True)
        self.solutions.add(x, y)
        return y


def initialize_clients(clients):
    while any(client.d_share is None for client in clients):
        for client in clients:
            if client.d_share is None:
                client._initializing()


def synchronous_round(clients, server):
    for client in clients:
        client._fit_surrogates()
        client.push()
    server.aggregate()

    for client in clients:
        client.pull()
        client._sync_transfer_payload()
        pseudo = client._build_pseudo_pool()
        if client._should_update_generator():
            client._train_generator(pseudo)
            client._last_generator_update_ver = client.version
        if client._should_update_selector():
            client._train_selector(pseudo)
            client._last_selector_update_ver = client.version
        previous_best = float(client.y_min)
        next_x = client._select_next_candidate(pseudo)
        next_y = client.problem.evaluate(next_x)
        improvement = previous_best - float(np.min(next_y))
        client.progress_history.append(improvement)
        client.progress_history = client.progress_history[-4:]
        client.version += 1


def main():
    server = RpcgsServer(top_k=2, d_share_size=16, debug_dir="out/rpcgs-demo", debug_level="ERROR")
    shifts = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.5]),
        np.array([2.0, -1.0, 1.0]),
    ]
    clients = []
    for cid, shift in enumerate(shifts, start=1):
        problem = ShiftedSphereProblem(cid=cid, shift=shift, seed=100 + cid)
        client = RpcgsClient(
            problem,
            seed=200 + cid,
            candidate_random=12,
            candidate_perturb=8,
            candidate_generate=12,
            generator_epochs=2,
            selector_epochs=2,
            use_bootstrap=False,
            generator_backend="none",
            selector_backend="none",
            pull_repeat=1,
        )
        client.bind_server(server)
        clients.append(client)

    initialize_clients(clients)
    for _ in range(5):
        synchronous_round(clients, server)

    for client in clients:
        print(f"client={client.id} evaluations={client.solutions.size} best={client.y_min:.6f}")


if __name__ == "__main__":
    main()

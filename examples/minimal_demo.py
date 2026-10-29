"""An in-process RPCGS example with four shifted Sphere tasks (one client thread per task)."""
from pathlib import Path
import sys
import tempfile
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rpcgs import RpcgsClient, RpcgsServer


class SolutionArchive:
    def __init__(self, dim):
        self.x = np.empty((0, dim), dtype=float)
        self.y = np.empty((0, 1), dtype=float)

    def append(self, x, y):
        self.x = np.vstack([self.x, np.asarray(x, dtype=float).reshape(-1, self.x.shape[1])])
        self.y = np.vstack([self.y, np.asarray(y, dtype=float).reshape(-1, 1)])

    size = property(lambda self: len(self.y))
    y_min = property(lambda self: float(np.min(self.y)))


class ShiftedSphereProblem:
    """Evaluated points are added to the archive, as the RPCGS client expects (auto_update_solutions)."""

    def __init__(self, cid, shift, dim=4, fe_init=9, fe_max=30, seed=0):
        self.id, self.name = cid, f"shifted_sphere_{cid}"
        self.dim, self.obj = dim, 1
        self.fe_init, self.fe_max = fe_init, fe_max
        self.lb, self.ub = -5.0 * np.ones(dim), 5.0 * np.ones(dim)
        self.shift = np.asarray(shift, dtype=float).reshape(1, dim)
        self.rng = np.random.default_rng(seed)
        self.auto_update_solutions = True
        self.solutions = SolutionArchive(dim)
        self.evaluate(self.random_uniform_x(fe_init))

    def random_uniform_x(self, size, within_partition=True):
        return self.rng.uniform(self.lb, self.ub, size=(int(size), self.dim))

    def evaluate(self, x):
        x = np.clip(np.asarray(x, dtype=float).reshape(-1, self.dim), self.lb, self.ub)
        if self.solutions.size + len(x) > self.fe_max:
            raise ValueError("The evaluation budget would be exceeded.")
        y = np.sum((x - self.shift) ** 2, axis=1, keepdims=True)
        if self.auto_update_solutions:
            self.solutions.append(x, y)
        return y


def run_client(client):
    client.optimize()  # initialization: upload bounds, receive the shared reference points
    while client.solutions.size < client.fe_max:
        client.optimize()  # one round: fit, upload, receive neighbors, evaluate one point


def main():
    shifts = [np.zeros(4), 0.5 * np.ones(4), np.array([2.0, -1.0, 1.0, 0.0]), np.array([2.5, -1.0, 1.0, 0.5])]
    out_dir = tempfile.mkdtemp(prefix="rpcgs_demo_")
    server = RpcgsServer(debug_dir=out_dir, debug_level="ERROR")
    clients = [RpcgsClient(ShiftedSphereProblem(cid, s, seed=100 + cid), seed=200 + cid, debug_level="ERROR",
                           pull_repeat=600, pull_interval=0.01)
               for cid, s in enumerate(shifts, start=1)]
    for client in clients:  # register every client before the first upload
        client.bind_server(server)
    threads = [threading.Thread(target=run_client, args=(c,)) for c in clients]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for client in clients:
        print(f"client={client.id} evaluations={client.solutions.size} best={client.solutions.y_min:.6f}")
    print(f"logs and per-round diagnostics: {out_dir}")


if __name__ == "__main__":
    main()

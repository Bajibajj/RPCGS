"""Minimal in-process interfaces for the RPCGS decision loop.

The algorithm classes only need a client that can send packages to a server and read its problem, a server
that knows its registered clients, and a versioned buffer for client uploads and aggregation results.
Clients run in their own threads (see examples/minimal_demo.py); every request is handled under one lock
and is followed by an aggregation attempt, so a client waiting for an aggregation result is answered as soon
as all clients have uploaded the current version.
"""
from __future__ import annotations

import functools
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class ClientPackage:
    cid: int
    action: Any
    version: int | None = None
    data: Any = None


def record_runtime(name: str | None = None):
    def decorate(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            self.record_round_info(name or func.__name__, f"[{time.time() - start:.2f}s]")
            return result
        return wrapper
    return decorate


class Client:
    def __init__(self, problem):
        self.problem = problem
        self._server = None
        self.rounds_info: dict[str, list[str]] = defaultdict(list)

    def bind_server(self, server: "Server"):
        self._server = server
        server.register_client(self.id)

    def request_server(self, package, repeat: int = 10, interval: float = 1.0, msg: str | None = None):
        if self._server is None:
            raise RuntimeError("Call bind_server(server) before optimizing.")
        for _ in range(max(1, int(repeat))):
            response = self._server.serve(package)
            if self.check_pkg(response):
                return response
            time.sleep(interval)
        return None

    def check_pkg(self, package) -> bool:
        return package is not None

    def record_round_info(self, name: str, value: Any):
        self.rounds_info[name].append(str(value))

    id = property(lambda self: self.problem.id)
    dim = property(lambda self: self.problem.dim)
    obj = property(lambda self: self.problem.obj)
    fe_init = property(lambda self: self.problem.fe_init)
    fe_max = property(lambda self: self.problem.fe_max)
    lb = property(lambda self: self.problem.lb)
    ub = property(lambda self: self.problem.ub)
    solutions = property(lambda self: self.problem.solutions)


class Server:
    def __init__(self):
        self._active_clients: set[int] = set()
        self._lock = threading.Lock()

    def register_client(self, client_id: int):
        self._active_clients.add(int(client_id))

    def serve(self, package):
        """Handle one request, then aggregate if every client has uploaded a new version."""
        with self._lock:
            response = self.handle_request(package)
            self.aggregate()
            return response

    def handle_request(self, package):  # implemented by the algorithm
        raise NotImplementedError

    def aggregate(self):  # implemented by the algorithm
        raise NotImplementedError

    sorted_ids = property(lambda self: sorted(self._active_clients))
    num_clients = property(lambda self: len(self._active_clients))


class SyncDataManager:
    """Versioned buffers of client uploads (source) and aggregation results."""

    def __init__(self):
        self._source: dict[int, dict[int, Any]] = defaultdict(dict)
        self._result: dict[int, dict[int, Any]] = defaultdict(dict)

    def update_src(self, cid: int, version: int, data: Any):
        self._source[int(cid)][int(version)] = data

    def update_res(self, cid: int, version: int, data: Any):
        self._result[int(cid)][int(version)] = data

    def get_src(self, cid: int, version: int):
        return self._source.get(int(cid), {}).get(int(version))

    def get_res(self, cid: int, version: int):
        return self._result.get(int(cid), {}).get(int(version))

    @property
    def available_src_ver(self) -> int:
        """Latest version uploaded by every client (-1 if a client has uploaded nothing)."""
        if not self._source:
            return -1
        return min(max(data) for data in self._source.values())

    @property
    def num_clients(self) -> int:
        return len(self._source)

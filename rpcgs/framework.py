from __future__ import annotations

import functools
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class Actions(Enum):
    REGISTER = auto()
    QUIT = auto()
    PUSH_INIT = auto()
    PULL_INIT = auto()
    PUSH_UPDATE = auto()
    PULL_UPDATE = auto()


@dataclass
class ClientPackage:
    cid: int
    action: Actions
    version: int | None = None
    data: Any = None


def record_runtime(name: str | None = None):
    def decorate(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            elapsed = time.time() - start
            label = name or func.__name__
            if hasattr(self, "record_round_info"):
                self.record_round_info(label, f"[{elapsed:.2f}s]")
            return result

        return wrapper

    return decorate


class Client:
    def __init__(self, problem):
        self.problem = problem
        self._server = None
        self._round_info: dict[str, list[str]] = defaultdict(list)

    def bind_server(self, server: "Server"):
        self._server = server
        server.register_client(self.id)

    def request_server(self, package: ClientPackage, repeat: int = 1, interval: float = 0.0, msg: str | None = None):
        if self._server is None:
            raise RuntimeError("No server is bound. Call bind_server(server) before requesting server state.")
        response = None
        for _ in range(max(1, int(repeat))):
            response = self._server.handle_request(package)
            if self.check_pkg(response):
                return response
            if interval > 0:
                time.sleep(interval)
        return response

    def check_pkg(self, package) -> bool:
        return package is not None

    def record_round_info(self, name: str, value: Any):
        self._round_info[name].append(str(value))

    @property
    def id(self):
        return self.problem.id

    @property
    def dim(self):
        return self.problem.dim

    @property
    def obj(self):
        return self.problem.obj

    @property
    def fe_init(self):
        return self.problem.fe_init
    @property
    def fe_max(self):
        return self.problem.fe_max

    @property
    def lb(self):
        return self.problem.lb

    @property
    def ub(self):
        return self.problem.ub

    @property
    def solutions(self):
        return self.problem.solutions


class Server:
    def __init__(self):
        self._active_clients: set[int] = set()

    def register_client(self, client_id: int):
        self._active_clients.add(int(client_id))

    @property
    def sorted_ids(self):
        return sorted(self._active_clients)

    @property
    def num_clients(self):
        return len(self._active_clients)

class SyncDataManager:
    def __init__(self):
        self._src: dict[int, dict[int, ClientPackage]] = defaultdict(dict)
        self._res: dict[int, dict[int, Any]] = defaultdict(dict)

    def update_src(self, cid: int, version: int, package: ClientPackage):
        self._src[int(cid)][int(version)] = package

    def get_src(self, cid: int, version: int):
        return self._src[int(cid)][int(version)]

    def update_res(self, cid: int, version: int, data: Any):
        self._res[int(cid)][int(version)] = data

    def get_res(self, cid: int, version: int):
        return self._res.get(int(cid), {}).get(int(version))

    @property
    def num_clients(self):
        return len(self._src)

    @property
    def available_src_ver(self):
        if not self._src:
            return -1
        counts = Counter()
        for by_version in self._src.values():
            counts.update(by_version.keys())
        required = len(self._src)
        ready = [version for version, count in counts.items() if count == required]
        return max(ready) if ready else -1

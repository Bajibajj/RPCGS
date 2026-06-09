from .client import RpcgsClient
from .server import RpcgsServer
from .framework import Actions, ClientPackage


class RPCGS:
    client = RpcgsClient
    server = RpcgsServer


__all__ = ["RPCGS", "RpcgsClient", "RpcgsServer", "Actions", "ClientPackage"]

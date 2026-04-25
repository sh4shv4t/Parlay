"""
OpenEnv compatibility layer for Parlay.
openenv-core v0.2.3 provides client-side classes only.
Server side is standard FastAPI — already implemented in parlay_env/server.py.
"""
try:
    from openenv import (
        GenericEnvClient,
        SyncEnvClient,
        GenericAction,
        AutoEnv,
        AutoAction,
    )
    OPENENV_AVAILABLE = True
    OPENENV_VERSION = "0.2.3"
except ImportError:
    OPENENV_AVAILABLE = False
    OPENENV_VERSION = None
    GenericEnvClient = object
    SyncEnvClient = object
    GenericAction = object
    AutoEnv = None
    AutoAction = None

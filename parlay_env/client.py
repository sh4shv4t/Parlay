"""
ParlayEnvClient — OpenEnv-compatible client for the Parlay negotiation environment.
Inherits from openenv.GenericEnvClient.

Quick start:
    from parlay_env.client import ParlayEnvClient, ParlayAction

    with ParlayEnvClient(
        base_url="https://huggingface.co/spaces/sh4shv4t/Parlay"
    ).sync() as client:
        obs = client.reset(scenario_id="saas_enterprise", persona="shark")

        result = client.step(ParlayAction(
            utterance="We propose 150,000 for the annual contract.",
            offer_amount=150000.0
        ))

Or via AutoEnv (if Space is registered with OpenEnv):
    from openenv import AutoEnv
    env = AutoEnv.from_space("sh4shv4t/Parlay")
    with env.sync() as client:
        obs = client.reset()
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from parlay_env.openenv_compat import (
    GenericEnvClient,
    GenericAction,
    OPENENV_AVAILABLE,
)


# ── Action ────────────────────────────────────────────────────────────────────


@dataclass
class ParlayAction(GenericAction if OPENENV_AVAILABLE else object):
    """
    Action for the Parlay negotiation environment.
    Inherits from openenv.GenericAction when openenv-core is installed.

    Fields:
        utterance       Natural language negotiation text (required)
        offer_amount    Numeric offer in scenario currency units (optional)
        tactical_move   One of: anchor_high | batna_reveal | silence (optional)
        accept_deal     True to accept the current standing offer
        walk_away       True to terminate negotiation
    """
    utterance: str = ""
    offer_amount: Optional[float] = None
    tactical_move: Optional[str] = None
    accept_deal: bool = False
    walk_away: bool = False

    def to_dict(self) -> dict:
        return {
            "utterance": self.utterance,
            "offer_amount": self.offer_amount,
            "tactical_move": self.tactical_move,
            "accept_deal": self.accept_deal,
            "walk_away": self.walk_away,
        }


# ── Helpers ─────────────────────────────────────────────────────────────────────


def _hf_space_to_ws_url(base_url: str) -> str:
    """
    https://huggingface.co/spaces/ORG/NAME → wss://org-name.hf.space/env/ws
    For direct *.hf.space URLs, preserve host and ensure path /env/ws.
    """
    u = base_url.rstrip("/")
    if "huggingface.co/spaces/" in u:
        rest = u.split("huggingface.co/spaces/", 1)[-1].split("?")[0]
        if "/" in rest:
            org, name = rest.split("/", 1)
            slug = f"{org}-{name}".replace(" ", "-").lower()
            return f"wss://{slug}.hf.space/env/ws"
    parsed = urlparse(u)
    if parsed.hostname and "hf.space" in parsed.hostname:
        scheme = "wss" if (parsed.scheme or "https") in ("https", "wss", "") else "ws"
        path = (parsed.path or "").rstrip("/")
        if not path.endswith("/env/ws"):
            path = f"{path}/env/ws" if path else "/env/ws"
        return f"{scheme}://{parsed.netloc}{path}"
    out = u.replace("https://", "wss://").replace("http://", "ws://")
    if not out.rstrip("/").endswith("env/ws"):
        out = out.rstrip("/") + "/env/ws"
    return out


def _merge_observation_response(data: dict) -> dict:
    """Flatten {observation, done, session_id?} for callers using obs.get('done')."""
    if not isinstance(data, dict):
        return data
    if "error" in data and "observation" not in data:
        return data
    if "observation" not in data:
        return data
    out = {**data}
    obs = out.pop("observation", None)
    if isinstance(obs, dict):
        merged = {**obs, "done": out.get("done", obs.get("episode_done", False))}
        for key in ("session_id", "error", "_fallback"):
            if key in out:
                merged[key] = out[key]
        return merged
    return out


# ── Client ────────────────────────────────────────────────────────────────────


class ParlayEnvClient(GenericEnvClient if OPENENV_AVAILABLE else object):
    """
    OpenEnv-compatible client for the Parlay negotiation environment.
    Inherits from openenv.GenericEnvClient when openenv-core is installed.

    The server runs at: https://huggingface.co/spaces/sh4shv4t/Parlay
    WebSocket endpoint: wss://sh4shv4t-parlay.hf.space/env/ws

    Usage (sync):
        with ParlayEnvClient(base_url="https://...").sync() as client:
            obs = client.reset(scenario_id="hiring_package", persona="veteran")
            while not obs.get("done", False):
                result = client.step(ParlayAction(
                    utterance="I propose 200,000.",
                    offer_amount=200000.0
                ))
                obs = result

    Usage (async):
        async with ParlayEnvClient(base_url="https://...") as client:
            obs = await client.reset()
            result = await client.step(action)
    """

    DEFAULT_BASE_URL = "https://huggingface.co/spaces/sh4shv4t/Parlay"

    def __init__(self, base_url: str = DEFAULT_BASE_URL, **kwargs):
        if OPENENV_AVAILABLE:
            try:
                super().__init__(base_url=base_url, **kwargs)
            except Exception:
                pass
        self.base_url = base_url.rstrip("/")
        self._ws_url = _hf_space_to_ws_url(self.base_url)
        self._ws = None
        self._session_id: Optional[str] = None

    def sync(self) -> "_SyncParlayClient":
        """Return synchronous wrapper for use as context manager."""
        return _SyncParlayClient(self)

    async def __aenter__(self):
        await self._connect()
        return self

    async def __aexit__(self, *args):
        await self._disconnect()

    async def _connect(self):
        try:
            import websockets

            self._ws = await websockets.connect(self._ws_url)
        except ImportError as exc:
            raise ImportError(
                "pip install websockets  # needed for async ParlayEnvClient"
            ) from exc

    async def _disconnect(self):
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def reset(
        self, scenario_id: str = "saas_enterprise", persona: str = "shark", **kwargs
    ) -> dict:
        """Reset environment, returns initial ParlayObservation dict."""
        if not self._ws:
            await self._connect()
        # Server matches on `cmd` (see parlay_env/server.py _coerce_message_params).
        await self._ws.send(
            json.dumps(
                {
                    "cmd": "reset",
                    "scenario_id": scenario_id,
                    "persona": persona,
                    **{k: v for k, v in kwargs.items() if k in ("seed",)},
                }
            )
        )
        data = json.loads(await self._ws.recv())
        merged = _merge_observation_response(data)
        if isinstance(merged, dict) and "session_id" in merged:
            self._session_id = str(merged["session_id"])
        return merged

    async def step(self, action) -> dict:
        """
        Send action, returns observation dict with reward and done flag.
        action: ParlayAction instance or dict
        """
        if not self._ws:
            raise RuntimeError("Call reset() first")
        if not self._session_id:
            raise RuntimeError("No session_id — call reset() first")
        payload = action.to_dict() if hasattr(action, "to_dict") else action
        body = {
            "cmd": "step",
            "session_id": self._session_id,
            "action": payload,
        }
        await self._ws.send(json.dumps(body))
        data = json.loads(await self._ws.recv())
        return _merge_observation_response(data)

    async def state(self) -> dict:
        """Return full ParlayState including hidden state (god view)."""
        if not self._ws:
            raise RuntimeError("Call reset() first")
        if not self._session_id:
            raise RuntimeError("No session_id — call reset() first")
        await self._ws.send(
            json.dumps({"cmd": "state", "session_id": self._session_id})
        )
        return json.loads(await self._ws.recv())


class _SyncParlayClient:
    """Synchronous wrapper returned by ParlayEnvClient.sync()."""

    def __init__(self, async_client: ParlayEnvClient):
        self._client = async_client
        self._ws = None
        self._session_id: Optional[str] = None

    def __enter__(self):
        try:
            import websocket as ws_lib  # type: ignore  # noqa: I001, pylint: disable=import-outside-toplevel

            self._ws = ws_lib.create_connection(self._client._ws_url)
        except ImportError as exc:
            raise ImportError(
                "pip install websocket-client  # needed for sync ParlayEnvClient"
            ) from exc
        return self

    def __exit__(self, *args):
        if self._ws:
            self._ws.close()
            self._ws = None
        self._session_id = None
        return False

    def reset(
        self, scenario_id: str = "saas_enterprise", persona: str = "shark", **kwargs
    ) -> dict:
        self._ws.send(
            json.dumps(
                {
                    "cmd": "reset",
                    "scenario_id": scenario_id,
                    "persona": persona,
                    **{k: v for k, v in kwargs.items() if k in ("seed",)},
                }
            )
        )
        data = json.loads(self._ws.recv())
        merged = _merge_observation_response(data)
        if isinstance(merged, dict) and "session_id" in merged:
            self._session_id = str(merged["session_id"])
            self._client._session_id = self._session_id
        return merged

    def step(self, action) -> dict:
        if not self._session_id:
            raise RuntimeError("Call reset() first")
        payload = action.to_dict() if hasattr(action, "to_dict") else action
        self._ws.send(
            json.dumps(
                {
                    "cmd": "step",
                    "session_id": self._session_id,
                    "action": payload,
                }
            )
        )
        return _merge_observation_response(json.loads(self._ws.recv()))

    def state(self) -> dict:
        if not self._session_id:
            raise RuntimeError("Call reset() first")
        self._ws.send(
            json.dumps({"cmd": "state", "session_id": self._session_id})
        )
        return json.loads(self._ws.recv())

"""
Parlay OpenEnv WebSocket server.
Implements the standard reset/step/state protocol.
Each connection manages one negotiation episode.
"""
import json
import logging
import os
import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect

from .exceptions import (
    EpisodeAlreadyDoneError,
    InvalidActionError,
    InvalidScenarioError,
    SessionNotFoundError,
)
from .game_theory import compute_nash_bargaining_solution, compute_zopa
from .grader import compute_step_reward
from .models import (
    BeliefState,
    HiddenState,
    ParlayAction,
    ParlayObservation,
    ParlayState,
    PersonaType,
    TacticalMove,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/env", tags=["OpenEnv"])

# In-memory session store (replaced by Redis in prod)
_sessions: dict[str, ParlayState] = {}

MAX_TURNS = int(os.getenv("MAX_TURNS_PER_EPISODE", "20"))
CP_START  = int(os.getenv("CREDIBILITY_POINTS_START", "100"))
CP_REGEN  = int(os.getenv("CREDIBILITY_REGEN_PER_TURN", "5"))

_SCENARIO_DEFAULTS: dict[str, dict[str, Any]] = {
    "saas_enterprise":        dict(budget=165_000,    walk=125_000,    urgency=0.55, alt=True),
    "consulting_retainer":    dict(budget=40_000,     walk=25_000,     urgency=0.45, alt=False),
    "hiring_package":         dict(budget=230_000,    walk=195_000,    urgency=0.60, alt=False),
    "vendor_hardware":        dict(budget=2_200_000,  walk=1_750_000,  urgency=0.50, alt=False),
    "acquisition_term_sheet": dict(budget=16_000_000, walk=10_500_000, urgency=0.65, alt=True),
}

_CP_COSTS: dict[TacticalMove, int] = {
    TacticalMove.ANCHOR_HIGH:      0,
    TacticalMove.BATNA_REVEAL:     20,
    TacticalMove.COALITION_INVITE: 30,
    TacticalMove.TIME_PRESSURE:    15,
    TacticalMove.SWEETENER:        10,
    TacticalMove.SILENCE:          5,
    TacticalMove.REFRAME:          12,
}


def _get_scenario_hidden_state(scenario_id: str, rng_seed: int = 42) -> HiddenState:
    """Return a HiddenState for the given scenario with slight randomisation."""
    if scenario_id not in _SCENARIO_DEFAULTS:
        raise InvalidScenarioError(f"Unknown scenario: {scenario_id}")

    rng = np.random.default_rng(rng_seed)
    d   = _SCENARIO_DEFAULTS[scenario_id]
    noise = rng.uniform(0.95, 1.05)

    return HiddenState(
        budget_ceiling=round(d["budget"] * noise, 2),
        walk_away_price=round(d["walk"] * noise, 2),
        urgency_score=float(np.clip(d["urgency"] + rng.uniform(-0.1, 0.1), 0.0, 1.0)),
        has_alternative=bool(d["alt"]),
        persona_drifted=False,
    )


def _initial_belief(hidden: HiddenState) -> BeliefState:
    """Initial belief state — intentionally imprecise."""
    return BeliefState(
        est_budget=hidden.budget_ceiling * 0.80,
        est_walk_away=hidden.walk_away_price * 1.15,
        est_urgency=0.50,
        est_has_alternative=False,
        confidence=0.30,
    )


def _make_observation(
    state: ParlayState,
    reward: float,
    utterance: str,
    drift_event: str | None = None,
) -> ParlayObservation:
    """Build a ParlayObservation from the current state."""
    zopa       = compute_zopa(state.hidden_state.budget_ceiling, state.hidden_state.walk_away_price)
    zopa_lower = zopa[0] if zopa else 0.0
    zopa_upper = zopa[1] if zopa else 0.0
    nash = compute_nash_bargaining_solution(
        state.hidden_state.budget_ceiling,
        state.hidden_state.walk_away_price,
    )
    current_offer = state.offer_history[-1] if state.offer_history else 0.0
    tension       = min(100.0, max(0.0, 50.0 + (state.step_count / MAX_TURNS) * 50.0))
    belief        = state.belief_history[-1] if state.belief_history else _initial_belief(state.hidden_state)

    return ParlayObservation(
        step_count=state.step_count,
        episode_done=state.episode_done,
        current_offer=current_offer,
        opponent_offer=zopa_upper * 0.9,
        zopa_lower=zopa_lower,
        zopa_upper=zopa_upper,
        nash_point=nash,
        tension_score=tension,
        belief_state=belief,
        last_utterance=utterance,
        available_moves=list(TacticalMove),
        credibility_points=state.credibility_points,
        reward=reward,
        cumulative_reward=state.cumulative_reward,
        drift_event=drift_event,
        act=state.act,
    )


def _get_cp_cost(move: TacticalMove | None) -> int:
    """Return the credibility-point cost for a tactical move."""
    if move is None:
        return 0
    return _CP_COSTS.get(move, 0)


@router.websocket("/ws")
async def env_websocket(websocket: WebSocket) -> None:
    """
    OpenEnv WebSocket endpoint.

    Accepts messages: {"cmd": "reset"|"step"|"state", ...params}
    Responds with observation dict or error dict.
    """
    await websocket.accept()
    logger.info("OpenEnv WebSocket client connected")
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            cmd = msg.get("cmd")
            try:
                match cmd:
                    case "reset":
                        result = await _handle_reset(msg)
                    case "step":
                        result = await _handle_step(msg)
                    case "state":
                        result = await _handle_state(msg)
                    case _:
                        result = {"error": f"Unknown command: {cmd}"}
            except (
                InvalidActionError,
                SessionNotFoundError,
                EpisodeAlreadyDoneError,
                InvalidScenarioError,
            ) as exc:
                result = {"error": str(exc)}
            except Exception:
                logger.exception("Unhandled error in env WebSocket")
                result = {"error": "Internal server error"}

            await websocket.send_json(result)
    except WebSocketDisconnect:
        logger.info("OpenEnv WebSocket client disconnected")


async def _handle_reset(msg: dict) -> dict:
    """Handle the reset command: create a fresh episode."""
    scenario_id = msg.get("scenario_id", "saas_enterprise")
    persona_str = msg.get("persona", "shark")
    seed        = msg.get("seed", 42)

    try:
        persona = PersonaType(persona_str)
    except ValueError:
        raise InvalidScenarioError(f"Unknown persona: {persona_str}")

    hidden         = _get_scenario_hidden_state(scenario_id, seed)
    initial_belief = _initial_belief(hidden)
    session_id     = str(uuid.uuid4())

    state = ParlayState(
        session_id=session_id,
        scenario_id=scenario_id,
        persona=persona,
        act=1,
        step_count=0,
        cumulative_reward=0.0,
        hidden_state=hidden,
        belief_history=[initial_belief],
        offer_history=[],
        drift_events_fired=0,
        episode_done=False,
        credibility_points=CP_START,
    )
    _sessions[session_id] = state

    obs = _make_observation(state, 0.0, "Negotiation started. Make your opening move.")
    logger.info(f"Reset: session={session_id}, scenario={scenario_id}, persona={persona_str}")
    return {"session_id": session_id, "observation": obs.model_dump()}


async def _handle_step(msg: dict) -> dict:
    """Handle the step command: advance the episode by one action."""
    session_id = msg.get("session_id")
    if not session_id or session_id not in _sessions:
        raise SessionNotFoundError(f"Session {session_id} not found")

    state = _sessions[session_id]
    if state.episode_done:
        raise EpisodeAlreadyDoneError(f"Episode {session_id} is already done")

    try:
        action = ParlayAction.model_validate(msg.get("action", {}))
    except Exception as exc:
        raise InvalidActionError(f"Invalid action: {exc}") from exc

    # Credibility point accounting
    cp_cost = _get_cp_cost(action.tactical_move)
    new_cp  = min(CP_START, state.credibility_points + CP_REGEN - cp_cost)
    if new_cp < 0:
        raise InvalidActionError("Insufficient credibility points for that move")

    # Update offer history
    new_offers = list(state.offer_history)
    if action.offer_amount is not None:
        new_offers.append(action.offer_amount)

    # Update belief history (Bayesian update simplified to linear adjustment)
    new_beliefs = list(state.belief_history)
    if new_beliefs:
        last = new_beliefs[-1]
        updated = BeliefState(
            est_budget=last.est_budget * 0.98,
            est_walk_away=last.est_walk_away * 1.01,
            est_urgency=min(1.0, last.est_urgency + 0.02),
            est_has_alternative=last.est_has_alternative,
            confidence=min(1.0, last.confidence + 0.05),
        )
        new_beliefs.append(updated)

    new_step = state.step_count + 1
    done     = new_step >= MAX_TURNS

    next_state = ParlayState(
        session_id=state.session_id,
        scenario_id=state.scenario_id,
        persona=state.persona,
        act=state.act,
        step_count=new_step,
        cumulative_reward=state.cumulative_reward,
        hidden_state=state.hidden_state,
        belief_history=new_beliefs,
        offer_history=new_offers,
        drift_events_fired=state.drift_events_fired,
        episode_done=done,
        termination_reason="max_turns" if done else None,
        credibility_points=new_cp,
    )

    step_reward = compute_step_reward(state, action, next_state)

    # Rebuild with updated cumulative reward (HiddenState is frozen so model_dump is safe)
    next_state = ParlayState(
        **{**next_state.model_dump(), "cumulative_reward": state.cumulative_reward + step_reward}
    )
    _sessions[session_id] = next_state

    obs = _make_observation(next_state, step_reward, action.utterance)
    return {"observation": obs.model_dump(), "done": done}


async def _handle_state(msg: dict) -> dict:
    """Handle the state command: return raw session state."""
    session_id = msg.get("session_id")
    if not session_id or session_id not in _sessions:
        raise SessionNotFoundError(f"Session {session_id} not found")
    return {"state": _sessions[session_id].model_dump()}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    """
    Get session state via REST (mirrors the WebSocket state command).

    Args:
        session_id: UUID of the session to retrieve.

    Returns:
        Dict containing the serialised ParlayState.
    """
    if session_id not in _sessions:
        raise SessionNotFoundError(f"Session {session_id} not found")
    return {"state": _sessions[session_id].model_dump()}


_env_app = FastAPI(title="Parlay OpenEnv", version="1.0.0")
_env_app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    _port = int(os.getenv("ENV_PORT", "8001"))
    uvicorn.run(_env_app, host="0.0.0.0", port=_port)

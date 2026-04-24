"""
Parlay OpenEnv WebSocket server.
Implements the standard reset/step/state protocol.
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
from .reward import (
    ZOPA_EROSION_CONSECUTIVE_TURNS,
    ZOPA_EROSION_RATE,
    ZOPA_EROSION_TENSION_THRESHOLD,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/env", tags=["OpenEnv"])

_FALLBACK_BELIEF = BeliefState(
    est_budget=0.0,
    est_walk_away=0.0,
    est_urgency=0.5,
    est_has_alternative=False,
    confidence=0.1,
)
FALLBACK_OBSERVATION = ParlayObservation(
    step_count=0,
    episode_done=False,
    current_offer=0.0,
    opponent_offer=0.0,
    zopa_lower=0.0,
    zopa_upper=0.0,
    nash_point=0.0,
    tension_score=0.0,
    belief_state=_FALLBACK_BELIEF,
    last_utterance="[Connection issue - AI is thinking]",
    available_moves=list(TacticalMove),
    credibility_points=100,
    reward=0.0,
    cumulative_reward=0.0,
)

_sessions: dict[str, ParlayState] = {}

MAX_TURNS = int(os.getenv("MAX_TURNS_PER_EPISODE", "20"))
CP_START = int(os.getenv("CREDIBILITY_POINTS_START", "100"))
CP_REGEN = int(os.getenv("CREDIBILITY_REGEN_PER_TURN", "5"))

_SCENARIO_DEFAULTS: dict[str, dict[str, Any]] = {
    "saas_enterprise": dict(budget=165_000, walk=125_000, urgency=0.55, alt=True),
    "hiring_package": dict(budget=230_000, walk=195_000, urgency=0.60, alt=False),
    "acquisition_term_sheet": dict(budget=16_000_000, walk=10_500_000, urgency=0.65, alt=True),
}

_CP_COSTS: dict[TacticalMove, int] = {
    TacticalMove.ANCHOR_HIGH: 0,
    TacticalMove.BATNA_REVEAL: 20,
    TacticalMove.SILENCE: 5,
}


def _get_scenario_hidden_state(scenario_id: str, rng_seed: int = 42) -> HiddenState:
    """Return a HiddenState for the given scenario with slight randomisation."""
    if scenario_id not in _SCENARIO_DEFAULTS:
        raise InvalidScenarioError(f"Unknown scenario: {scenario_id}")

    rng = np.random.default_rng(rng_seed)
    defaults = _SCENARIO_DEFAULTS[scenario_id]
    noise = rng.uniform(0.95, 1.05)

    return HiddenState(
        budget_ceiling=round(defaults["budget"] * noise, 2),
        walk_away_price=round(defaults["walk"] * noise, 2),
        urgency_score=float(np.clip(defaults["urgency"] + rng.uniform(-0.1, 0.1), 0.0, 1.0)),
        has_alternative=bool(defaults["alt"]),
        persona_drifted=False,
    )


def _initial_belief(hidden: HiddenState) -> BeliefState:
    """Initial belief state - intentionally imprecise."""
    return BeliefState(
        est_budget=hidden.budget_ceiling * 0.80,
        est_walk_away=hidden.walk_away_price * 1.15,
        est_urgency=0.50,
        est_has_alternative=False,
        confidence=0.30,
    )


def _get_cp_cost(move: TacticalMove | None) -> int:
    if move is None:
        return 0
    return _CP_COSTS.get(move, 0)


def _compute_tension(state: ParlayState, action: ParlayAction) -> float:
    """Compute the current tension score for the turn."""
    base = 20.0 + ((state.step_count + 1) / MAX_TURNS) * 60.0
    if action.tactical_move == TacticalMove.ANCHOR_HIGH:
        base += 15.0
    elif action.tactical_move == TacticalMove.BATNA_REVEAL:
        base += 10.0
    elif action.tactical_move == TacticalMove.SILENCE:
        base += 5.0
    return float(max(0.0, min(100.0, base)))


def _make_observation(
    state: ParlayState,
    reward: float,
    utterance: str,
    drift_event: str | None = None,
) -> ParlayObservation:
    """Build a ParlayObservation from the current state."""
    zopa = compute_zopa(state.hidden_state.budget_ceiling, state.hidden_state.walk_away_price)
    zopa_lower = zopa[0] if zopa else state.hidden_state.walk_away_price
    zopa_upper = zopa[1] if zopa else state.hidden_state.budget_ceiling
    nash = compute_nash_bargaining_solution(
        state.hidden_state.budget_ceiling,
        state.hidden_state.walk_away_price,
    )
    current_offer = state.offer_history[-1] if state.offer_history else 0.0
    belief = state.belief_history[-1] if state.belief_history else _initial_belief(state.hidden_state)

    original_zopa = state.original_zopa_width
    current_zopa = max(0.0, state.hidden_state.budget_ceiling - state.hidden_state.walk_away_price)
    width_pct = current_zopa / original_zopa if original_zopa > 0 else 0.0

    return ParlayObservation(
        step_count=state.step_count,
        episode_done=state.episode_done,
        current_offer=current_offer,
        opponent_offer=zopa_upper * 0.9,
        zopa_lower=zopa_lower,
        zopa_upper=zopa_upper,
        nash_point=nash,
        tension_score=state.tension_score,
        belief_state=belief,
        last_utterance=utterance,
        available_moves=list(TacticalMove),
        credibility_points=state.credibility_points,
        reward=reward,
        cumulative_reward=state.cumulative_reward,
        drift_event=drift_event,
        zopa_erosion_ticks=state.zopa_erosion_ticks,
        zopa_width_pct_remaining=width_pct,
    )


def _coerce_message_params(msg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Accept both {cmd, ...} and {method, params} envelope formats."""
    if "method" in msg:
        return str(msg["method"]), dict(msg.get("params", {}))
    command = str(msg.get("cmd", ""))
    params = {k: v for k, v in msg.items() if k != "cmd"}
    return command, params


@router.websocket("/ws")
async def env_websocket(websocket: WebSocket) -> None:
    """OpenEnv WebSocket endpoint."""
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

            command, params = _coerce_message_params(msg)
            try:
                match command:
                    case "reset":
                        result = await _handle_reset(params)
                    case "step":
                        result = await _handle_step(params)
                    case "state":
                        result = await _handle_state(params)
                    case _:
                        result = {"error": f"Unknown command: {command}"}
            except (
                InvalidActionError,
                SessionNotFoundError,
                EpisodeAlreadyDoneError,
                InvalidScenarioError,
            ) as exc:
                result = {"error": str(exc)}
            except Exception:
                logger.exception("Unhandled error in env WebSocket - returning fallback observation")
                result = {
                    "observation": FALLBACK_OBSERVATION.model_dump(),
                    "done": False,
                    "_fallback": True,
                }

            await websocket.send_json(result)
    except WebSocketDisconnect:
        logger.info("OpenEnv WebSocket client disconnected")


async def _handle_reset(msg: dict[str, Any]) -> dict:
    """Handle reset: create a fresh episode."""
    scenario_id = msg.get("scenario_id", "saas_enterprise")
    persona_str = msg.get("persona", "shark")
    seed = int(msg.get("seed", 42))

    try:
        persona = PersonaType(persona_str)
    except ValueError as exc:
        raise InvalidScenarioError(f"Unknown persona: {persona_str}") from exc

    hidden = _get_scenario_hidden_state(scenario_id, seed)
    initial_belief = _initial_belief(hidden)
    session_id = str(uuid.uuid4())
    original_zopa_width = hidden.budget_ceiling - hidden.walk_away_price

    state = ParlayState(
        session_id=session_id,
        scenario_id=scenario_id,
        persona=persona,
        step_count=0,
        cumulative_reward=0.0,
        hidden_state=hidden,
        belief_history=[initial_belief],
        offer_history=[],
        drift_events_fired=0,
        episode_done=False,
        credibility_points=CP_START,
        original_zopa_width=original_zopa_width,
        zopa_width_pct_remaining=1.0,
    )
    _sessions[session_id] = state

    observation = _make_observation(state, 0.0, "Negotiation started. Make your opening move.")
    logger.info("Reset: session=%s, scenario=%s, persona=%s", session_id, scenario_id, persona_str)
    return {"session_id": session_id, "observation": observation.model_dump(), "done": False}


async def _handle_step(msg: dict[str, Any]) -> dict:
    """Advance the episode by one action."""
    session_id = msg.get("session_id")
    if not session_id or session_id not in _sessions:
        raise SessionNotFoundError(f"Session {session_id} not found")

    state = _sessions[session_id]
    if state.episode_done:
        raise EpisodeAlreadyDoneError(f"Episode {session_id} is already done")

    action_payload = msg.get("action", msg)
    try:
        action = ParlayAction.model_validate(action_payload)
    except Exception as exc:
        raise InvalidActionError(f"Invalid action: {exc}") from exc

    cp_cost = _get_cp_cost(action.tactical_move)
    if state.credibility_points < cp_cost:
        raise InvalidActionError("Insufficient credibility points for that move")

    new_offers = list(state.offer_history)
    if action.offer_amount is not None:
        new_offers.append(action.offer_amount)

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

    next_state = ParlayState(
        **{
            **state.model_dump(),
            "step_count": state.step_count + 1,
            "offer_history": new_offers,
            "belief_history": new_beliefs,
            "credibility_points": min(CP_START, state.credibility_points + CP_REGEN - cp_cost),
            "tension_score": _compute_tension(state, action),
            "hidden_state": HiddenState(**state.hidden_state.model_dump()),
        }
    )

    if action.tactical_move == TacticalMove.BATNA_REVEAL:
        revealed = action.offer_amount if action.offer_amount is not None else next_state.hidden_state.walk_away_price
        next_state.hidden_state.last_stated_batna = float(revealed)

    if next_state.tension_score >= ZOPA_EROSION_TENSION_THRESHOLD:
        next_state.high_tension_streak += 1
    else:
        next_state.high_tension_streak = 0

    if next_state.high_tension_streak >= ZOPA_EROSION_CONSECUTIVE_TURNS:
        zopa_width = next_state.hidden_state.budget_ceiling - next_state.hidden_state.walk_away_price
        base_width = next_state.original_zopa_width or zopa_width
        shift = base_width * ZOPA_EROSION_RATE
        next_state.hidden_state.budget_ceiling -= shift
        next_state.hidden_state.walk_away_price += shift
        next_state.zopa_erosion_ticks += 1
        next_state.high_tension_streak = 0

        if next_state.hidden_state.budget_ceiling <= next_state.hidden_state.walk_away_price:
            next_state.walk_away = True
            next_state.termination_reason = "zopa_collapsed"

    current_zopa = max(0.0, next_state.hidden_state.budget_ceiling - next_state.hidden_state.walk_away_price)
    next_state.zopa_width_pct_remaining = (
        current_zopa / next_state.original_zopa_width if next_state.original_zopa_width > 0 else 0.0
    )

    if action.offer_amount is not None:
        next_state.deal_reached = (
            next_state.hidden_state.walk_away_price
            <= action.offer_amount
            <= next_state.hidden_state.budget_ceiling
        )

    step_reward = compute_step_reward(state, action, next_state)
    next_state.cumulative_reward = state.cumulative_reward + step_reward

    if step_reward >= 0.0 and action.tactical_move is None and state.hidden_state.last_stated_batna is not None:
        from .grader import detect_bluff_challenge  # noqa: PLC0415

        if detect_bluff_challenge(
            utterance=action.utterance,
            opponent_stated_batna=state.hidden_state.last_stated_batna,
            opponent_true_batna=state.hidden_state.budget_ceiling,
        ):
            next_state.bluffs_caught = state.bluffs_caught + 1

    next_state.episode_done = (
        next_state.step_count >= MAX_TURNS
        or step_reward < -100.0
        or next_state.deal_reached
        or next_state.walk_away
    )
    if next_state.episode_done and next_state.termination_reason is None:
        if next_state.deal_reached:
            next_state.termination_reason = "deal_reached"
        elif step_reward < -100.0:
            next_state.termination_reason = "reward_floor"
        elif next_state.walk_away:
            next_state.termination_reason = "walk_away"
        else:
            next_state.termination_reason = "max_turns"

    _sessions[session_id] = next_state
    observation = _make_observation(next_state, step_reward, action.utterance)
    return {"observation": observation.model_dump(), "done": next_state.episode_done}


async def _handle_state(msg: dict[str, Any]) -> dict:
    """Return raw session state."""
    session_id = msg.get("session_id")
    if not session_id or session_id not in _sessions:
        raise SessionNotFoundError(f"Session {session_id} not found")
    return {"state": _sessions[session_id].model_dump()}


def get_session_state(session_id: str) -> ParlayState | None:
    """Return the in-memory session state for SSE and tests."""
    return _sessions.get(session_id)


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    """Get session state via REST."""
    if session_id not in _sessions:
        raise SessionNotFoundError(f"Session {session_id} not found")
    return {"state": _sessions[session_id].model_dump()}


_env_app = FastAPI(title="Parlay OpenEnv", version="1.0.0")
_env_app.include_router(router)


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Parlay OpenEnv server")
    parser.add_argument("--port", type=int, default=int(os.getenv("ENV_PORT", "8001")))
    args = parser.parse_args()

    port = int(args.port)
    uvicorn.run(_env_app, host="0.0.0.0", port=port)

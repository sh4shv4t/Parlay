"""
Dashboard API router for Parlay.
Provides REST endpoints for the frontend game interface.
Mounted at /api in main.py.
"""
import asyncio
import json
import logging
import uuid
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.gemini_client import MODEL_ID_DEMO, call_gemini
from agent.personas import PERSONAS, build_system_prompt
from agent.tom_tracker import ToMTracker
from game.leaderboard import Leaderboard
from game.scenarios import SCENARIOS, get_scenario
from game.tactical_cards import TACTICAL_CARDS, get_card
from parlay_env.exceptions import InvalidPersonaError, InvalidScenarioError
from parlay_env.game_theory import compute_nash_bargaining_solution, compute_zopa
from parlay_env.grader import detect_bluff_challenge, grade_episode
from parlay_env.models import BeliefState, HiddenState, ParlayAction, ParlayState, PersonaType, TacticalMove
from parlay_env.reward import (
    PSI,
    ZOPA_EROSION_CONSECUTIVE_TURNS,
    ZOPA_EROSION_RATE,
    ZOPA_EROSION_TENSION_THRESHOLD,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Dashboard"])

_leaderboard = Leaderboard()

CP_START = 100
CP_REGEN = 5
MAX_TURNS = 20

_sessions: dict[str, dict[str, Any]] = {}
_CP_COSTS: dict[TacticalMove, int] = {
    TacticalMove.ANCHOR_HIGH: 0,
    TacticalMove.BATNA_REVEAL: 20,
    TacticalMove.SILENCE: 5,
}


class StartRequest(BaseModel):
    scenario_id: str
    persona: str
    player_name: str = "Player"


class MoveRequest(BaseModel):
    session_id: str
    amount: float
    message: str
    tactical_move: Optional[str] = None


class AcceptRequest(BaseModel):
    session_id: str


class WalkAwayRequest(BaseModel):
    session_id: str


class GameStepRequest(BaseModel):
    session_id: str
    move: str = "counter"
    offer_amount: Optional[float] = None
    card_id: Optional[str] = None
    message: str = ""


class SessionStartRequest(BaseModel):
    scenario_id: str = "saas_enterprise"
    persona: str = "shark"
    player_name: str = "Player"


class SessionStepRequest(BaseModel):
    amount: float = 145_000.0
    message: str = "I propose this amount."
    tactical_move: Optional[str] = None


def _get_tension(state: ParlayState, player_move: Optional[TacticalMove], opponent_move: Optional[TacticalMove]) -> float:
    base = 20.0 + ((state.step_count + 1) / MAX_TURNS) * 55.0
    if player_move == TacticalMove.ANCHOR_HIGH or opponent_move == TacticalMove.ANCHOR_HIGH:
        base += 15.0
    if player_move == TacticalMove.BATNA_REVEAL or opponent_move == TacticalMove.BATNA_REVEAL:
        base += 10.0
    if player_move == TacticalMove.SILENCE or opponent_move == TacticalMove.SILENCE:
        base += 5.0
    return float(max(0.0, min(100.0, base)))


def _serialise_cards() -> list[dict[str, Any]]:
    return [
        {
            "id": card.id,
            "move": card.id,
            "name": card.name,
            "cp_cost": card.cp_cost,
            "description": card.description,
            "theory": card.theory,
            "game_theory_ref": card.game_theory_ref,
        }
        for card in TACTICAL_CARDS.values()
    ]


def _build_observation(
    state: ParlayState,
    opponent_offer: Optional[float] = None,
    last_utterance: str = "",
    reward: float = 0.0,
    drift_event: Optional[str] = None,
) -> dict[str, Any]:
    zopa = compute_zopa(state.hidden_state.budget_ceiling, state.hidden_state.walk_away_price)
    zopa_lower = zopa[0] if zopa else state.hidden_state.walk_away_price
    zopa_upper = zopa[1] if zopa else state.hidden_state.budget_ceiling
    nash = compute_nash_bargaining_solution(state.hidden_state.budget_ceiling, state.hidden_state.walk_away_price)
    belief = state.belief_history[-1]
    current_offer = state.offer_history[-1] if state.offer_history else 0.0
    return {
        "step_count": state.step_count,
        "episode_done": state.episode_done,
        "current_offer": current_offer,
        "opponent_offer": opponent_offer,
        "zopa_lower": zopa_lower,
        "zopa_upper": zopa_upper,
        "nash_point": nash,
        "tension_score": state.tension_score,
        "belief_state": belief.model_dump(),
        "last_utterance": last_utterance,
        "available_moves": [move.value for move in TacticalMove],
        "credibility_points": state.credibility_points,
        "reward": reward,
        "cumulative_reward": state.cumulative_reward,
        "drift_event": drift_event,
        "zopa_erosion_ticks": state.zopa_erosion_ticks,
        "zopa_width_pct_remaining": state.zopa_width_pct_remaining,
        "session_id": state.session_id,
    }


def _build_session(scenario_id: str, persona_str: str, player_name: str) -> tuple[str, dict[str, Any]]:
    scenario = get_scenario(scenario_id)
    try:
        persona_type = PersonaType(persona_str)
    except ValueError as exc:
        raise InvalidPersonaError(
            f"Invalid persona: {persona_str!r}. Valid: {[p.value for p in PersonaType]}"
        ) from exc

    session_id = str(uuid.uuid4())
    rng = np.random.default_rng(hash(session_id) % 10000)
    noise = float(rng.uniform(0.95, 1.05))
    hidden = HiddenState(
        budget_ceiling=round(scenario.batna_buyer * noise, 2),
        walk_away_price=round(scenario.batna_seller * noise, 2),
        urgency_score=float(np.clip(0.5 + rng.uniform(-0.15, 0.15), 0.0, 1.0)),
        has_alternative=scenario.id in ("saas_enterprise", "acquisition_term_sheet"),
        persona_drifted=False,
    )
    initial_belief = BeliefState(
        est_budget=hidden.budget_ceiling * 0.80,
        est_walk_away=hidden.walk_away_price * 1.15,
        est_urgency=0.50,
        est_has_alternative=False,
        confidence=0.30,
    )
    tom = ToMTracker(initial_belief, persona_type)
    state = ParlayState(
        session_id=session_id,
        scenario_id=scenario_id,
        persona=persona_type,
        step_count=0,
        cumulative_reward=0.0,
        hidden_state=hidden,
        belief_history=[initial_belief],
        offer_history=[],
        drift_events_fired=0,
        episode_done=False,
        credibility_points=CP_START,
        original_zopa_width=scenario.batna_buyer - scenario.batna_seller,
        zopa_width_pct_remaining=1.0,
    )
    system_prompt = build_system_prompt(
        persona=persona_type,
        scenario_title=scenario.title,
        scenario_description=scenario.description,
        batna=hidden.walk_away_price,
        budget=hidden.budget_ceiling,
        urgency=hidden.urgency_score,
    )
    return session_id, {
        "session_id": session_id,
        "player_name": player_name,
        "scenario": scenario,
        "state": state,
        "tom_tracker": tom,
        "system_prompt": system_prompt,
        "conversation": [],
        "drift_turn": None,
        "drift_adapted": False,
        "last_opponent_offer": None,
        "last_opponent_move": None,
    }


def get_session(session_id: str) -> dict[str, Any] | None:
    return _sessions.get(session_id)


def _apply_zopa_erosion(state: ParlayState) -> None:
    if state.tension_score >= ZOPA_EROSION_TENSION_THRESHOLD:
        state.high_tension_streak += 1
    else:
        state.high_tension_streak = 0

    if state.high_tension_streak >= ZOPA_EROSION_CONSECUTIVE_TURNS:
        zopa_width = state.hidden_state.budget_ceiling - state.hidden_state.walk_away_price
        base_width = state.original_zopa_width or zopa_width
        shift = base_width * ZOPA_EROSION_RATE
        state.hidden_state.budget_ceiling -= shift
        state.hidden_state.walk_away_price += shift
        state.zopa_erosion_ticks += 1
        state.high_tension_streak = 0

        if state.hidden_state.budget_ceiling <= state.hidden_state.walk_away_price:
            state.walk_away = True
            state.termination_reason = "zopa_collapsed"

    current_zopa = max(0.0, state.hidden_state.budget_ceiling - state.hidden_state.walk_away_price)
    state.zopa_width_pct_remaining = (
        current_zopa / state.original_zopa_width if state.original_zopa_width > 0 else 0.0
    )


def _apply_drift(session: dict[str, Any]) -> Optional[str]:
    state: ParlayState = session["state"]
    scenario = session["scenario"]
    for event in scenario.drift_events:
        if event.trigger_turn == state.step_count:
            session["drift_turn"] = state.step_count
            state.drift_events_fired += 1
            session["tom_tracker"].drift_event(event.effect_on_urgency, event.effect_on_has_alternative)
            state.belief_history = list(session["tom_tracker"].history)
            return event.event
    return None


@router.get("/scenarios")
async def list_scenarios() -> dict:
    """List all available negotiation scenarios."""
    return {
        "scenarios": [
            {
                "id": s.id,
                "title": s.title,
                "description": s.description,
                "currency": s.currency,
                "unit": s.unit,
                "zopa_lower": s.zopa[0],
                "zopa_upper": s.zopa[1],
                "anchor_seller": s.anchor_seller,
                "anchor_buyer": s.anchor_buyer,
                "difficulty": s.difficulty,
                "drift_count": len(s.drift_events),
            }
            for s in SCENARIOS.values()
        ]
    }


@router.get("/personas")
async def list_personas() -> dict:
    """List all available negotiator personas."""
    return {
        "personas": [
            {
                "id": pt.value,
                "name": cfg.name,
                "symbol": cfg.symbol,
                "emoji": cfg.emoji,
                "aggression": cfg.aggression,
                "patience": cfg.patience,
                "bluff_rate": cfg.bluff_rate,
                "tom_depth": cfg.tom_depth,
                "color_var": cfg.color_var,
                "opening_line": cfg.opening_line,
            }
            for pt, cfg in PERSONAS.items()
        ]
    }


@router.post("/game/start")
async def start_game(req: StartRequest) -> dict:
    """Start a new negotiation session."""
    try:
        session_id, session = _build_session(req.scenario_id, req.persona, req.player_name)
    except (InvalidScenarioError, InvalidPersonaError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _sessions[session_id] = session
    scenario = session["scenario"]
    state: ParlayState = session["state"]
    persona_cfg = PERSONAS[state.persona]

    return {
        "session_id": session_id,
        "scenario": {
            "id": scenario.id,
            "title": scenario.title,
            "description": scenario.description,
            "currency": scenario.currency,
            "unit": scenario.unit,
            "anchor_seller": scenario.anchor_seller,
            "anchor_buyer": scenario.anchor_buyer,
        },
        "observation": _build_observation(state),
        "persona": {
            "id": req.persona,
            "name": persona_cfg.name,
            "symbol": persona_cfg.symbol,
            "emoji": persona_cfg.emoji,
            "color_var": persona_cfg.color_var,
            "opening_line": persona_cfg.opening_line,
        },
        "hand": _serialise_cards(),
        "opening_message": persona_cfg.opening_line,
    }


@router.post("/game/move")
async def make_move(req: MoveRequest) -> dict:
    """Submit a negotiation move and get the opponent's response."""
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state: ParlayState = session["state"]
    if state.episode_done:
        raise HTTPException(status_code=400, detail="Episode already concluded")

    move: Optional[TacticalMove] = None
    if req.tactical_move:
        try:
            move = TacticalMove(req.tactical_move)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tactical_move: {req.tactical_move!r}. Valid: {[m.value for m in TacticalMove]}",
            ) from exc

    cost = _CP_COSTS.get(move, 0)
    if state.credibility_points < cost:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient CP: need {cost}, have {state.credibility_points}",
        )

    drift_event_desc = _apply_drift(session)
    turn = state.step_count
    gemini_messages = []
    for msg in session["conversation"][-8:]:
        role = "user" if msg["role"] == "player" else "model"
        gemini_messages.append({"role": role, "parts": [msg["text"]]})

    player_text = f"Player offer: {req.amount:,.0f}. Message: {req.message}"
    if req.tactical_move:
        player_text += f" [Playing card: {req.tactical_move}]"
    gemini_messages.append({"role": "user", "parts": [player_text]})

    opponent_resp = await call_gemini(
        session["system_prompt"],
        gemini_messages,
        persona=state.persona.value,
        model=MODEL_ID_DEMO,
    )
    opponent_utterance = opponent_resp.get("utterance", "Let me think about that...")
    opponent_offer = opponent_resp.get("offer_amount")
    opponent_move: Optional[TacticalMove] = None
    try:
        if opponent_resp.get("tactical_move"):
            opponent_move = TacticalMove(opponent_resp["tactical_move"])
    except ValueError:
        opponent_move = None

    session["conversation"].append(
        {"role": "player", "text": req.message, "offer": req.amount, "move": req.tactical_move, "turn": turn + 1}
    )
    session["conversation"].append(
        {"role": "opponent", "text": opponent_utterance, "offer": opponent_offer, "move": opponent_resp.get("tactical_move"), "turn": turn + 1}
    )

    if opponent_move == TacticalMove.BATNA_REVEAL:
        state.hidden_state.last_stated_batna = (
            float(opponent_offer) if opponent_offer is not None else state.hidden_state.walk_away_price * 1.2
        )

    updated_belief = session["tom_tracker"].update(
        observed_offer=opponent_offer,
        observed_move=opponent_move,
        utterance=opponent_utterance,
        turn=turn,
    )

    if session["drift_turn"] is not None and not session["drift_adapted"]:
        if turn <= session["drift_turn"] + 2 and any(
            signal in req.message.lower() for signal in ["understand", "noted", "given", "considering", "account"]
        ):
            session["drift_adapted"] = True

    next_state = ParlayState(
        **{
            **state.model_dump(),
            "step_count": state.step_count + 1,
            "offer_history": [*state.offer_history, req.amount],
            "belief_history": list(session["tom_tracker"].history),
            "credibility_points": min(CP_START, state.credibility_points + CP_REGEN - cost),
        }
    )
    next_state.tension_score = _get_tension(state, move, opponent_move)
    _apply_zopa_erosion(next_state)

    if opponent_offer is not None and abs(req.amount - opponent_offer) / max(req.amount, 1.0) < 0.03:
        next_state.deal_reached = True
        next_state.termination_reason = "deal_reached"

    action = ParlayAction(utterance=req.message, offer_amount=req.amount, tactical_move=move)
    step_reward = (
        next_state.credibility_points - state.credibility_points
    ) * 0.0  # placeholder to keep local scope explicit
    from parlay_env.grader import compute_step_reward  # noqa: PLC0415

    step_reward = compute_step_reward(state, action, next_state)
    if (
        state.hidden_state.last_stated_batna is not None
        and move is None
        and detect_bluff_challenge(req.message, state.hidden_state.last_stated_batna, state.hidden_state.budget_ceiling)
    ):
        next_state.bluffs_caught = state.bluffs_caught + 1
        step_reward = max(step_reward, PSI)

    next_state.cumulative_reward = state.cumulative_reward + step_reward
    next_state.episode_done = (
        next_state.step_count >= MAX_TURNS
        or step_reward < -100.0
        or next_state.deal_reached
        or next_state.walk_away
    )
    if next_state.episode_done and next_state.termination_reason is None:
        if next_state.walk_away:
            next_state.termination_reason = "zopa_collapsed"
        elif step_reward < -100.0:
            next_state.termination_reason = "reward_floor"
        else:
            next_state.termination_reason = "max_turns"

    session["state"] = next_state
    session["last_opponent_offer"] = opponent_offer
    session["last_opponent_move"] = opponent_move.value if opponent_move else None
    _sessions[req.session_id] = session

    return {
        "opponent": {
            "utterance": opponent_utterance,
            "offer": opponent_offer,
            "tactical_move": opponent_move.value if opponent_move else None,
        },
        "opponent_message": opponent_utterance,
        "opponent_move": opponent_move.value if opponent_move else None,
        "observation": _build_observation(
            next_state,
            opponent_offer=opponent_offer,
            last_utterance=opponent_utterance,
            reward=step_reward,
            drift_event=drift_event_desc,
        ),
        "drift_event": drift_event_desc,
        "done": next_state.episode_done,
        "turns_remaining": MAX_TURNS - next_state.step_count,
        "hand": _serialise_cards(),
    }


@router.post("/game/accept")
async def accept_deal(req: AcceptRequest) -> dict:
    """Accept the current offer and close the deal."""
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state: ParlayState = session["state"]
    if state.episode_done:
        raise HTTPException(status_code=400, detail="Episode already concluded")
    if not state.offer_history and session["last_opponent_offer"] is None:
        raise HTTPException(status_code=400, detail="No offer to accept. Make an offer first.")

    final_price = session["last_opponent_offer"] or state.offer_history[-1]
    state.deal_reached = True
    state.episode_done = True
    state.termination_reason = "deal_accepted"
    grade = grade_episode(
        state,
        final_price=final_price,
        t_close=state.step_count,
        t_max=MAX_TURNS,
        drift_adapted=session["drift_adapted"],
        bluffs_caught=state.bluffs_caught,
    )

    await _leaderboard.record_result(
        player_name=session["player_name"],
        scenario_id=state.scenario_id,
        persona=state.persona.value,
        total_reward=grade.total_reward,
        deal_efficiency=grade.deal_efficiency,
        acts_completed=1,
        deal_closed=True,
    )

    session["state"] = state
    _sessions[req.session_id] = session
    nash = compute_nash_bargaining_solution(state.hidden_state.budget_ceiling, state.hidden_state.walk_away_price)
    zopa = compute_zopa(state.hidden_state.budget_ceiling, state.hidden_state.walk_away_price)
    zopa_width = (zopa[1] - zopa[0]) if zopa else 1.0

    return {
        "final_price": final_price,
        "total_reward": grade.total_reward,
        "deal_efficiency": grade.deal_efficiency,
        "tom_accuracy_avg": grade.tom_accuracy_avg,
        "nash_comparison": {
            "nash_point": nash,
            "your_deal": final_price,
            "delta_pct": round((final_price - nash) / max(zopa_width, 1) * 100, 1),
        },
        "grade": {
            "total_reward": grade.total_reward,
            "deal_efficiency": grade.deal_efficiency,
            "tom_accuracy_avg": grade.tom_accuracy_avg,
            "bluffs_caught": grade.bluffs_caught,
            "drift_adapted": grade.drift_adapted,
        },
    }


@router.post("/game/walkaway")
async def walk_away(req: WalkAwayRequest) -> dict:
    """Walk away from the negotiation without a deal."""
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state: ParlayState = session["state"]
    if state.episode_done:
        raise HTTPException(status_code=400, detail="Episode already concluded")

    state.walk_away = True
    state.episode_done = True
    state.termination_reason = "walk_away"
    session["state"] = state
    _sessions[req.session_id] = session

    nash = compute_nash_bargaining_solution(state.hidden_state.budget_ceiling, state.hidden_state.walk_away_price)
    zopa = compute_zopa(state.hidden_state.budget_ceiling, state.hidden_state.walk_away_price)
    return {
        "result": "walk_away",
        "message": "You walked away. No deal recorded.",
        "counterfactual_optimal": nash,
        "zopa": {"lower": zopa[0] if zopa else 0, "upper": zopa[1] if zopa else 0},
        "termination_reason": state.termination_reason,
    }


@router.get("/leaderboard")
async def get_leaderboard(scenario_id: Optional[str] = None, limit: int = 10) -> dict:
    """Get the global or per-scenario leaderboard."""
    valid_scenarios = list(SCENARIOS.keys())
    if scenario_id and scenario_id not in valid_scenarios:
        raise HTTPException(status_code=400, detail=f"Invalid scenario_id. Valid: {valid_scenarios}")
    entries = await _leaderboard.get_top(scenario_id=scenario_id, limit=min(limit, 50))
    return {"entries": entries, "scenario_id": scenario_id or "global"}


@router.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "parlay-dashboard"}


@router.post("/game/step")
async def game_step(req: GameStepRequest) -> dict:
    """Unified step endpoint for the browser UI."""
    if req.move == "accept":
        return await accept_deal(AcceptRequest(session_id=req.session_id))
    if req.move == "walk_away":
        return await walk_away(WalkAwayRequest(session_id=req.session_id))
    if req.offer_amount is None:
        raise HTTPException(status_code=400, detail="offer_amount required for counter moves")

    tactical_move = req.card_id
    if req.move == "anchor":
        tactical_move = TacticalMove.ANCHOR_HIGH.value

    return await make_move(
        MoveRequest(
            session_id=req.session_id,
            amount=req.offer_amount,
            message=req.message or req.move,
            tactical_move=tactical_move,
        )
    )


@router.post("/session/start")
async def session_start(req: SessionStartRequest) -> dict:
    """Start a simplified session API flow."""
    try:
        session_id, session = _build_session(req.scenario_id, req.persona, req.player_name)
    except (InvalidScenarioError, InvalidPersonaError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _sessions[session_id] = session
    return {"session_id": session_id, "status": "ok"}


@router.post("/session/{session_id}/step")
async def session_step(session_id: str, req: SessionStepRequest) -> dict:
    """Execute one negotiation step in a simplified session."""
    return await make_move(
        MoveRequest(
            session_id=session_id,
            amount=req.amount,
            message=req.message,
            tactical_move=req.tactical_move,
        )
    )


@router.get("/session/{session_id}/spectate-stream")
async def spectate_stream(session_id: str):
    """
    Server-Sent Events stream of full session state including hidden fields.
    Used by spectate.html for live demo projection.
    """

    async def event_generator():
        while True:
            session = get_session(session_id)
            if session is None:
                yield f"data: {json.dumps({'error': 'session not found'})}\n\n"
                break

            state: ParlayState = session["state"]
            payload = {
                "turn": state.step_count,
                "tension": state.tension_score,
                "zopa_lower": state.hidden_state.walk_away_price,
                "zopa_upper": state.hidden_state.budget_ceiling,
                "true_urgency": state.hidden_state.urgency_score,
                "true_walkaway": state.hidden_state.budget_ceiling,
                "last_stated_batna": state.hidden_state.last_stated_batna,
                "bluffs_caught": state.bluffs_caught,
                "zopa_erosion_ticks": state.zopa_erosion_ticks,
                "zopa_width_pct": state.zopa_width_pct_remaining,
                "tom_accuracy": session["tom_tracker"].accuracy_against(state.hidden_state),
                "active_market_event": None,
                "true_mev_impact": state.hidden_state.event_impacts,
                "cumulative_reward": state.cumulative_reward,
                "conversation_tail": session["conversation"][-3:],
                "is_terminal": state.deal_reached or state.walk_away or state.episode_done,
                "termination_reason": state.termination_reason,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            if payload["is_terminal"]:
                break
            await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

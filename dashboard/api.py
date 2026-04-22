"""
Dashboard API router for Parlay.
Provides REST endpoints for the frontend game interface.
Mounted at /api in main.py.
"""
import logging
import uuid
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from parlay_env.exceptions import InvalidPersonaError, InvalidScenarioError
from parlay_env.models import (
    PersonaType, TacticalMove, BeliefState, HiddenState, ParlayState,
)
from parlay_env.game_theory import compute_zopa, compute_nash_bargaining_solution
from parlay_env.grader import grade_episode
from game.scenarios import SCENARIOS, get_scenario
from game.tactical_cards import TACTICAL_CARDS, draw_hand
from game.leaderboard import Leaderboard
from game.achievements import AchievementSystem
from agent.personas import PERSONAS, build_system_prompt
from agent.gemini_client import call_gemini
from agent.tom_tracker import ToMTracker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Dashboard"])

_leaderboard = Leaderboard()
_achievements = AchievementSystem()

CP_START = 100
CP_REGEN = 5
MAX_TURNS = 20

# In-memory session store
_sessions: dict[str, dict] = {}

# CP costs per tactical move
_CP_COSTS: dict[TacticalMove, int] = {
    TacticalMove.ANCHOR_HIGH:      0,
    TacticalMove.BATNA_REVEAL:    20,
    TacticalMove.COALITION_INVITE: 30,
    TacticalMove.TIME_PRESSURE:   15,
    TacticalMove.SWEETENER:       10,
    TacticalMove.SILENCE:          5,
    TacticalMove.REFRAME:         12,
}


# --- Request Models ---

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


# --- Session Builder ---

def _build_session(scenario_id: str, persona_str: str, player_name: str) -> tuple[str, dict]:
    """Create a new game session and return (session_id, session_dict)."""
    scenario = get_scenario(scenario_id)  # raises InvalidScenarioError if bad

    try:
        persona_type = PersonaType(persona_str)
    except ValueError:
        raise InvalidPersonaError(f"Invalid persona: {persona_str!r}. Valid: {[p.value for p in PersonaType]}")

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
    hand = draw_hand(5, rng_seed=hash(session_id) % 9999)
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
        "scenario_id": scenario_id,
        "persona": persona_str,
        "persona_type": persona_type,
        "hidden": hidden,
        "tom": tom,
        "system_prompt": system_prompt,
        "conversation": [],
        "offer_history": [],
        "step_count": 0,
        "cumulative_reward": 0.0,
        "credibility_points": CP_START,
        "act": 1,
        "done": False,
        "hand": hand,
        "drift_adapted": False,
        "drift_turn": None,
    }


# --- Endpoints ---

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
            for s in SCENARIOS
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
        session_id, sess = _build_session(req.scenario_id, req.persona, req.player_name)
    except (InvalidScenarioError, InvalidPersonaError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected error starting game: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")

    _sessions[session_id] = sess
    scenario = get_scenario(req.scenario_id)
    hidden = sess["hidden"]
    zopa = compute_zopa(hidden.budget_ceiling, hidden.walk_away_price)
    nash = compute_nash_bargaining_solution(hidden.budget_ceiling, hidden.walk_away_price)
    persona_cfg = PERSONAS[sess["persona_type"]]

    logger.info(f"Game started: session={session_id}, {req.scenario_id}/{req.persona}")

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
        "observation": {
            "step_count": 0,
            "zopa_lower": zopa[0] if zopa else 0,
            "zopa_upper": zopa[1] if zopa else 0,
            "nash_point": nash,
            "credibility_points": CP_START,
            "act": 1,
            "tension_score": 10.0,
            "belief_state": sess["tom"].current_belief.model_dump(),
            "available_moves": [m.value for m in TacticalMove],
        },
        "persona": {
            "id": req.persona,
            "name": persona_cfg.name,
            "symbol": persona_cfg.symbol,
            "emoji": persona_cfg.emoji,
            "color_var": persona_cfg.color_var,
            "opening_line": persona_cfg.opening_line,
        },
        "hand": [
            {
                "move": m.value,
                "name": TACTICAL_CARDS[m].name,
                "symbol": TACTICAL_CARDS[m].symbol,
                "cp_cost": TACTICAL_CARDS[m].cp_cost,
                "description": TACTICAL_CARDS[m].description,
                "game_theory_basis": TACTICAL_CARDS[m].game_theory_basis,
            }
            for m in sess["hand"]
        ],
        "opening_message": persona_cfg.opening_line,
    }


@router.post("/game/move")
async def make_move(req: MoveRequest) -> dict:
    """Submit a negotiation move and get the opponent's response."""
    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sess = _sessions[req.session_id]
    if sess["done"]:
        raise HTTPException(status_code=400, detail="Episode already concluded")

    move: Optional[TacticalMove] = None
    if req.tactical_move:
        try:
            move = TacticalMove(req.tactical_move)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tactical_move: {req.tactical_move!r}. Valid: {[m.value for m in TacticalMove]}",
            )

    cost = _CP_COSTS.get(move, 0)
    if sess["credibility_points"] < cost:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient CP: need {cost}, have {sess['credibility_points']}",
        )

    new_cp = min(CP_START, sess["credibility_points"] + CP_REGEN - cost)

    # Check drift events
    turn = sess["step_count"]
    scenario = get_scenario(sess["scenario_id"])
    drift_event_desc: Optional[str] = None
    for event in scenario.drift_events:
        if event.trigger_turn == turn:
            drift_event_desc = event.event
            sess["drift_turn"] = turn
            sess["tom"].drift_event(event.effect_on_urgency, event.effect_on_has_alternative)
            logger.info(f"Drift event triggered: {event.event}")
            break

    # Build Gemini message history (last 8 messages)
    gemini_messages = []
    for msg in sess["conversation"][-8:]:
        role = "user" if msg["role"] == "player" else "model"
        gemini_messages.append({"role": role, "parts": [msg["content"]]})

    player_text = f"Player offer: {req.amount:,.0f}. Message: {req.message}"
    if req.tactical_move:
        player_text += f" [Playing card: {req.tactical_move}]"
    gemini_messages.append({"role": "user", "parts": [player_text]})

    opponent_resp = await call_gemini(sess["system_prompt"], gemini_messages)
    opponent_utterance: str = opponent_resp.get("utterance", "Let me think about that...")
    opponent_offer: Optional[float] = opponent_resp.get("offer_amount")

    sess["conversation"].append({
        "role": "player", "content": req.message,
        "offer": req.amount, "move": req.tactical_move, "turn": turn + 1,
    })
    sess["conversation"].append({
        "role": "opponent", "content": opponent_utterance,
        "offer": opponent_offer, "turn": turn + 1,
    })

    updated_belief = sess["tom"].update(
        observed_offer=opponent_offer,
        observed_move=None,
        utterance=opponent_utterance,
        turn=turn,
    )

    # Check drift adaptation within 2 turns
    if sess["drift_turn"] is not None and not sess["drift_adapted"]:
        if turn <= sess["drift_turn"] + 2:
            if any(s in req.message.lower() for s in ["understand", "noted", "given", "considering", "account"]):
                sess["drift_adapted"] = True

    sess["offer_history"].append(req.amount)
    sess["step_count"] += 1
    sess["credibility_points"] = new_cp
    sess["cumulative_reward"] += 4.0  # placeholder step reward; full reward computed at grade_episode

    done = sess["step_count"] >= MAX_TURNS
    sess["done"] = done
    _sessions[req.session_id] = sess

    hidden = sess["hidden"]
    zopa = compute_zopa(hidden.budget_ceiling, hidden.walk_away_price)
    nash = compute_nash_bargaining_solution(hidden.budget_ceiling, hidden.walk_away_price)
    tension = min(100.0, 20.0 + (sess["step_count"] / MAX_TURNS) * 80.0)

    logger.debug(
        f"Move: session={req.session_id}, turn={sess['step_count']}, amount={req.amount}"
    )

    return {
        "opponent": {
            "utterance": opponent_utterance,
            "offer": opponent_offer,
        },
        "observation": {
            "step_count": sess["step_count"],
            "zopa_lower": zopa[0] if zopa else 0,
            "zopa_upper": zopa[1] if zopa else 0,
            "nash_point": nash,
            "tension_score": tension,
            "belief_state": updated_belief.model_dump(),
            "credibility_points": new_cp,
            "act": sess["act"],
        },
        "drift_event": drift_event_desc,
        "done": done,
        "turns_remaining": MAX_TURNS - sess["step_count"],
    }


@router.post("/game/accept")
async def accept_deal(req: AcceptRequest) -> dict:
    """Accept the current offer and close the deal."""
    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sess = _sessions[req.session_id]
    if sess["done"]:
        raise HTTPException(status_code=400, detail="Episode already concluded")
    if not sess["offer_history"]:
        raise HTTPException(status_code=400, detail="No offer to accept. Make an offer first.")

    final_price = sess["offer_history"][-1]
    hidden = sess["hidden"]

    state = ParlayState(
        session_id=req.session_id,
        scenario_id=sess["scenario_id"],
        persona=PersonaType(sess["persona"]),
        act=sess["act"],
        step_count=sess["step_count"],
        cumulative_reward=sess["cumulative_reward"],
        hidden_state=hidden,
        belief_history=sess["tom"].history,
        offer_history=sess["offer_history"],
        drift_events_fired=1 if sess["drift_turn"] is not None else 0,
        episode_done=True,
        termination_reason="deal_accepted",
        credibility_points=sess["credibility_points"],
    )

    grade = grade_episode(
        state,
        final_price=final_price,
        t_close=sess["step_count"],
        t_max=MAX_TURNS,
        drift_adapted=sess["drift_adapted"],
        bluffs_caught=sess["tom"].bluffs_detected,
    )

    await _leaderboard.record_result(
        player_name=sess["player_name"],
        scenario_id=sess["scenario_id"],
        persona=sess["persona"],
        total_reward=grade.total_reward,
        deal_efficiency=grade.deal_efficiency,
        acts_completed=grade.acts_completed,
        deal_closed=True,
    )

    earned = _achievements.evaluate(
        deal_closed=True,
        deal_efficiency=grade.deal_efficiency,
        t_close=sess["step_count"],
        tom_accuracy_avg=grade.tom_accuracy_avg,
        drift_adapted=sess["drift_adapted"],
        bluffs_caught=sess["tom"].bluffs_detected,
        acts_completed=grade.acts_completed,
        persona=sess["persona"],
    )

    sess["done"] = True
    _sessions[req.session_id] = sess

    nash = compute_nash_bargaining_solution(hidden.budget_ceiling, hidden.walk_away_price)
    zopa = compute_zopa(hidden.budget_ceiling, hidden.walk_away_price)
    zopa_width = (zopa[1] - zopa[0]) if zopa else 1.0

    logger.info(
        f"Deal accepted: session={req.session_id}, "
        f"price={final_price:,.0f}, efficiency={grade.deal_efficiency:.3f}, "
        f"reward={grade.total_reward:.2f}"
    )

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
        "achievements": [
            {"id": a.id, "name": a.name, "icon": a.icon, "xp": a.xp_reward}
            for a in earned
        ],
        "grade": {
            "total_reward": grade.total_reward,
            "deal_efficiency": grade.deal_efficiency,
            "tom_accuracy_avg": grade.tom_accuracy_avg,
            "bluffs_caught": grade.bluffs_caught,
            "acts_completed": grade.acts_completed,
            "drift_adapted": grade.drift_adapted,
        },
    }


@router.post("/game/walkaway")
async def walk_away(req: WalkAwayRequest) -> dict:
    """Walk away from the negotiation without a deal."""
    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sess = _sessions[req.session_id]
    if sess["done"]:
        raise HTTPException(status_code=400, detail="Episode already concluded")

    sess["done"] = True
    _sessions[req.session_id] = sess

    hidden = sess["hidden"]
    nash = compute_nash_bargaining_solution(hidden.budget_ceiling, hidden.walk_away_price)
    zopa = compute_zopa(hidden.budget_ceiling, hidden.walk_away_price)

    logger.info(f"Walk away: session={req.session_id}")

    return {
        "result": "walk_away",
        "message": "You walked away. No deal recorded.",
        "counterfactual_optimal": nash,
        "zopa": {"lower": zopa[0] if zopa else 0, "upper": zopa[1] if zopa else 0},
    }


@router.get("/leaderboard")
async def get_leaderboard(scenario_id: Optional[str] = None, limit: int = 10) -> dict:
    """Get the global or per-scenario leaderboard."""
    valid_scenarios = [s.id for s in SCENARIOS]
    if scenario_id and scenario_id not in valid_scenarios:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario_id. Valid: {valid_scenarios}",
        )
    entries = await _leaderboard.get_top(scenario_id=scenario_id, limit=min(limit, 50))
    return {"entries": entries, "scenario_id": scenario_id or "global"}


@router.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "parlay-dashboard"}


# ── Unified game step (used by frontend JS) ──────────────────────────────────

class GameStepRequest(BaseModel):
    session_id: str
    move: str = "counter"
    offer_amount: Optional[float] = None
    card_id: Optional[str] = None


@router.post("/game/step")
async def game_step(req: GameStepRequest) -> dict:
    """
    Unified step endpoint — dispatches to move / accept / walkaway.

    Args:
        session_id:   Active session UUID.
        move:         "counter" | "anchor" | "concede" | "package" | "accept" | "walk_away".
        offer_amount: Required for counter / anchor / concede / package moves.
        card_id:      Optional tactical card identifier.

    Returns:
        Opponent response, updated observation, and done flag.
    """
    match req.move:
        case "accept":
            return await accept_deal(AcceptRequest(session_id=req.session_id))
        case "walk_away":
            return await walk_away(WalkAwayRequest(session_id=req.session_id))
        case _:
            if req.offer_amount is None:
                raise HTTPException(status_code=400, detail="offer_amount required for counter moves")
            tactic: Optional[str] = None
            if req.move == "anchor":
                tactic = "anchor_high"
            return await make_move(MoveRequest(
                session_id=req.session_id,
                amount=req.offer_amount,
                message=req.move,
                tactical_move=tactic,
            ))


# ── Session API (simplified, used by tests and external integrations) ─────────

class SessionStartRequest(BaseModel):
    scenario_id: str = "saas_enterprise"
    persona: str = "shark"
    player_name: str = "Player"


class SessionStepRequest(BaseModel):
    amount: float = 145_000.0
    message: str = "I propose this amount."
    tactical_move: Optional[str] = None


@router.post("/session/start")
async def session_start(req: SessionStartRequest) -> dict:
    """
    Start a new session (simplified API).
    Works in both mock and live Gemini mode.

    Args:
        scenario_id: One of the five scenario IDs.
        persona:     One of the five persona IDs.
        player_name: Display name for the player.

    Returns:
        session_id and status.
    """
    try:
        session_id, sess = _build_session(req.scenario_id, req.persona, req.player_name)
    except (InvalidScenarioError, InvalidPersonaError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected error in session/start: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")

    _sessions[session_id] = sess
    logger.info(f"Session started (simplified): {session_id}, {req.scenario_id}/{req.persona}")
    return {"session_id": session_id, "status": "ok"}


@router.post("/session/{session_id}/step")
async def session_step(session_id: str, req: SessionStepRequest) -> dict:
    """
    Execute one negotiation step in a session.
    Returns a mock AI response when GOOGLE_API_KEY is absent.

    Args:
        session_id: Active session UUID from /session/start.
        amount:     Player's offer amount.
        message:    Player's utterance.
        tactical_move: Optional tactical move string.

    Returns:
        observation dict and opponent response.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sess = _sessions[session_id]
    if sess["done"]:
        raise HTTPException(status_code=400, detail="Episode already concluded")

    turn = sess["step_count"]
    gemini_messages = [
        {"role": "user", "parts": [f"Offer: {req.amount:,.0f}. {req.message}"]}
    ]
    opponent_resp = await call_gemini(
        sess["system_prompt"],
        gemini_messages,
        persona=sess["persona"],
    )

    sess["offer_history"].append(req.amount)
    sess["step_count"] += 1
    sess["done"] = sess["step_count"] >= MAX_TURNS
    _sessions[session_id] = sess

    hidden = sess["hidden"]
    zopa = compute_zopa(hidden.budget_ceiling, hidden.walk_away_price)
    nash = compute_nash_bargaining_solution(hidden.budget_ceiling, hidden.walk_away_price)
    tension = min(100.0, 20.0 + (sess["step_count"] / MAX_TURNS) * 80.0)

    return {
        "observation": {
            "step_count": sess["step_count"],
            "zopa_lower": zopa[0] if zopa else 0.0,
            "zopa_upper": zopa[1] if zopa else 0.0,
            "nash_point": nash,
            "tension_score": tension,
            "belief_state": sess["tom"].current_belief.model_dump(),
            "credibility_points": sess["credibility_points"],
            "act": sess["act"],
        },
        "opponent": opponent_resp,
        "done": sess["done"],
        "turns_remaining": MAX_TURNS - sess["step_count"],
    }

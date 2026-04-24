"""
Parlay MCP tools — universal protocol (works with any MCP client).
8 tools covering negotiation lifecycle, game state, and leaderboard.
"""
import logging
import uuid
from typing import Optional

import numpy as np
from fastmcp import FastMCP

from parlay_env.models import (
    PersonaType, TacticalMove, BeliefState, HiddenState, ParlayState,
)
from parlay_env.game_theory import compute_zopa, compute_nash_bargaining_solution
from parlay_env.grader import grade_episode
from game.scenarios import SCENARIOS, get_scenario
from game.tactical_cards import TACTICAL_CARDS, draw_hand, get_card
from game.leaderboard import Leaderboard
from agent.personas import PERSONAS, build_system_prompt
from agent.gemini_client import MODEL_ID_DEMO, call_gemini
from agent.tom_tracker import ToMTracker

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "Parlay Negotiation Environment",
    description=(
        "An RL negotiation environment. Train agents, play scenarios, "
        "access game state and leaderboards via MCP tools. "
        "Any MCP-compatible client can use these tools."
    ),
)

# In-memory session store for MCP sessions
_sessions: dict[str, dict] = {}
_leaderboard = Leaderboard()

CP_START = 100
CP_REGEN = 5
MAX_TURNS = 20

# CP costs per tactical move
_CP_COSTS: dict[TacticalMove, int] = {
    TacticalMove.ANCHOR_HIGH: 0,
    TacticalMove.BATNA_REVEAL:    20,
    TacticalMove.SILENCE:          5,
}


def _get_hidden_state(scenario_id: str, seed: int = 42) -> HiddenState:
    """Build a HiddenState for the given scenario."""
    rng = np.random.default_rng(seed)
    scenario = get_scenario(scenario_id)
    noise = float(rng.uniform(0.95, 1.05))
    return HiddenState(
        budget_ceiling=round(scenario.batna_buyer * noise, 2),
        walk_away_price=round(scenario.batna_seller * noise, 2),
        urgency_score=float(np.clip(0.5 + rng.uniform(-0.15, 0.15), 0.0, 1.0)),
        has_alternative=scenario.id in ("saas_enterprise", "acquisition_term_sheet"),
        persona_drifted=False,
    )


@mcp.tool()
async def start_negotiation(
    scenario_id: str,
    persona: str,
    player_name: str = "Agent",
) -> dict:
    """
    Start a new negotiation episode.

    Args:
        scenario_id: One of: saas_enterprise, hiring_package, acquisition_term_sheet
        persona: One of: shark, diplomat, veteran
        player_name: Display name for the leaderboard (default: "Agent")

    Returns:
        session_id: Unique session identifier for subsequent calls.
        observation: Initial game state including ZOPA, Nash point, and opening message.
        scenario: Scenario context (title, description, currency, unit).
        available_cards: List of tactical cards available to the player.
        opening_message: The AI opponent's opening statement.
    """
    valid_scenarios = list(SCENARIOS.keys())
    if scenario_id not in valid_scenarios:
        return {"error": f"Invalid scenario_id. Valid options: {valid_scenarios}"}

    try:
        persona_type = PersonaType(persona)
    except ValueError:
        return {"error": f"Invalid persona. Valid options: {[p.value for p in PersonaType]}"}

    scenario = get_scenario(scenario_id)
    session_id = str(uuid.uuid4())
    hidden = _get_hidden_state(scenario_id, seed=hash(session_id) % 10000)

    initial_belief = BeliefState(
        est_budget=hidden.budget_ceiling * 0.80,
        est_walk_away=hidden.walk_away_price * 1.15,
        est_urgency=0.50,
        est_has_alternative=False,
        confidence=0.30,
    )

    tom = ToMTracker(initial_belief, persona_type)
    hand = draw_hand(3, rng_seed=hash(session_id) % 9999)

    system_prompt = build_system_prompt(
        persona=persona_type,
        scenario_title=scenario.title,
        scenario_description=scenario.description,
        batna=hidden.walk_away_price,
        budget=hidden.budget_ceiling,
        urgency=hidden.urgency_score,
    )

    persona_cfg = PERSONAS[persona_type]
    opening_message = persona_cfg.opening_line

    zopa = compute_zopa(hidden.budget_ceiling, hidden.walk_away_price)
    nash = compute_nash_bargaining_solution(hidden.budget_ceiling, hidden.walk_away_price)

    _sessions[session_id] = {
        "session_id": session_id,
        "player_name": player_name,
        "scenario_id": scenario_id,
        "persona": persona,
        "persona_type": persona_type,
        "hidden": hidden,
        "tom": tom,
        "system_prompt": system_prompt,
        "conversation": [{"role": "opponent", "content": opening_message, "turn": 0}],
        "offer_history": [],
        "step_count": 0,
        "cumulative_reward": 0.0,
        "credibility_points": CP_START,
        "done": False,
        "hand": [m.value for m in hand],
        "drift_adapted": False,
        "drift_turn": None,
    }

    logger.info(f"MCP start_negotiation: session={session_id}, scenario={scenario_id}, persona={persona}")

    return {
        "session_id": session_id,
        "observation": {
            "step_count": 0,
            "zopa_lower": zopa[0] if zopa else 0,
            "zopa_upper": zopa[1] if zopa else 0,
            "nash_point": nash,
            "credibility_points": CP_START,
            "tension_score": 10.0,
            "belief_state": initial_belief.model_dump(),
        },
        "scenario": {
            "title": scenario.title,
            "description": scenario.description,
            "currency": scenario.currency,
            "unit": scenario.unit,
            "anchor_seller": scenario.anchor_seller,
            "anchor_buyer": scenario.anchor_buyer,
        },
        "available_cards": [
            {
                "move": m,
                "name": get_card(m).name,
                "cp_cost": get_card(m).cp_cost,
                "description": get_card(m).description,
            }
            for m in _sessions[session_id]["hand"]
        ],
        "opening_message": opening_message,
    }


@mcp.tool()
async def make_offer(
    session_id: str,
    amount: float,
    message: str,
    tactical_move: Optional[str] = None,
) -> dict:
    """
    Make a structured offer in the negotiation.

    Args:
        session_id: Session ID from start_negotiation.
        amount: Offer amount in the scenario's currency.
        message: Natural language message accompanying the offer.
        tactical_move: Optional tactical card to play. One of:
                       anchor_high, batna_reveal, silence

    Returns:
        opponent_response: AI opponent's counter-message and offer.
        updated_observation: Updated ZOPA, beliefs, tension, CP.
        reward: Step reward earned this turn.
        done: Whether the episode has ended.
        drift_event: Description of any drift event triggered (or null).
    """
    if session_id not in _sessions:
        return {"error": f"Session {session_id} not found. Call start_negotiation first."}

    sess = _sessions[session_id]
    if sess["done"]:
        return {"error": "Episode is already complete. Start a new session."}

    move: Optional[TacticalMove] = None
    if tactical_move:
        try:
            move = TacticalMove(tactical_move)
        except ValueError:
            return {"error": f"Invalid tactical_move. Valid: {[m.value for m in TacticalMove]}"}

    cost = _CP_COSTS.get(move, 0)
    current_cp = sess["credibility_points"]
    if current_cp < cost:
        return {"error": f"Insufficient credibility points. Need {cost}, have {current_cp}."}

    new_cp = min(CP_START, current_cp + CP_REGEN - cost)

    # Check for drift events
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

    # Build Gemini message history (last 10 messages for context window)
    gemini_messages = []
    for msg in sess["conversation"][-10:]:
        role = "user" if msg["role"] == "player" else "model"
        gemini_messages.append({"role": role, "parts": [msg["content"]]})

    player_text = f"Player offer: {amount:,.0f}. Message: {message}"
    if tactical_move:
        player_text += f" [Tactical move: {tactical_move}]"
    gemini_messages.append({"role": "user", "parts": [player_text]})

    opponent_resp = await call_gemini(
        sess["system_prompt"],
        gemini_messages,
        model=MODEL_ID_DEMO,
    )
    opponent_utterance: str = opponent_resp.get("utterance", "I'll need to consider that.")
    opponent_offer: Optional[float] = opponent_resp.get("offer_amount")
    opponent_move: Optional[str] = opponent_resp.get("tactical_move")

    sess["conversation"].append({
        "role": "player", "content": message,
        "offer": amount, "move": tactical_move, "turn": turn + 1,
    })
    sess["conversation"].append({
        "role": "opponent", "content": opponent_utterance,
        "offer": opponent_offer, "turn": turn + 1,
    })

    parsed_opp_move: Optional[TacticalMove] = None
    if opponent_move:
        try:
            parsed_opp_move = TacticalMove(opponent_move)
        except ValueError:
            pass

    updated_belief = sess["tom"].update(
        observed_offer=opponent_offer,
        observed_move=parsed_opp_move,
        utterance=opponent_utterance,
        turn=turn,
    )

    # Check drift adaptation within 2 turns
    if sess["drift_turn"] is not None and not sess["drift_adapted"]:
        if turn <= sess["drift_turn"] + 2:
            adaptation_signals = ["understand", "noted", "given that", "considering", "account"]
            if any(s in message.lower() for s in adaptation_signals):
                sess["drift_adapted"] = True

    sess["offer_history"].append(amount)
    sess["step_count"] += 1
    sess["credibility_points"] = new_cp

    step_reward = 5.0 * (1.0 - sess["tom"].accuracy_against(sess["hidden"]))
    sess["cumulative_reward"] += step_reward

    done = sess["step_count"] >= MAX_TURNS
    sess["done"] = done

    hidden = sess["hidden"]
    zopa = compute_zopa(hidden.budget_ceiling, hidden.walk_away_price)
    nash = compute_nash_bargaining_solution(hidden.budget_ceiling, hidden.walk_away_price)
    tension = min(100.0, 20.0 + (sess["step_count"] / MAX_TURNS) * 80.0)

    _sessions[session_id] = sess
    logger.debug(f"MCP make_offer: session={session_id}, turn={sess['step_count']}, amount={amount}")

    return {
        "opponent_response": {
            "utterance": opponent_utterance,
            "offer": opponent_offer,
            "tactical_move": opponent_move,
        },
        "updated_observation": {
            "step_count": sess["step_count"],
            "zopa_lower": zopa[0] if zopa else 0,
            "zopa_upper": zopa[1] if zopa else 0,
            "nash_point": nash,
            "tension_score": tension,
            "belief_state": updated_belief.model_dump(),
            "credibility_points": new_cp,
        },
        "reward": step_reward,
        "cumulative_reward": sess["cumulative_reward"],
        "done": done,
        "drift_event": drift_event_desc,
    }


@mcp.tool()
async def get_game_state(session_id: str) -> dict:
    """
    Get the full current game state for a session.

    Args:
        session_id: Session ID from start_negotiation.

    Returns:
        Full game state including beliefs, offer history, ZOPA data,
        CP balance, conversation history, and current leaderboard rank.
    """
    if session_id not in _sessions:
        return {"error": f"Session {session_id} not found."}

    sess = _sessions[session_id]
    hidden = sess["hidden"]
    zopa = compute_zopa(hidden.budget_ceiling, hidden.walk_away_price)
    nash = compute_nash_bargaining_solution(hidden.budget_ceiling, hidden.walk_away_price)
    current_belief = sess["tom"].current_belief

    rank = await _leaderboard.get_rank(sess["player_name"], sess["scenario_id"])

    return {
        "session_id": session_id,
        "player_name": sess["player_name"],
        "scenario_id": sess["scenario_id"],
        "persona": sess["persona"],
        "step_count": sess["step_count"],
        "done": sess["done"],
        "offer_history": sess["offer_history"],
        "zopa": {"lower": zopa[0] if zopa else 0, "upper": zopa[1] if zopa else 0},
        "nash_point": nash,
        "belief_state": current_belief.model_dump(),
        "belief_history_count": len(sess["tom"].history),
        "credibility_points": sess["credibility_points"],
        "cumulative_reward": sess["cumulative_reward"],
        "drift_adapted": sess["drift_adapted"],
        "bluffs_detected": sess["tom"].bluffs_detected,
        "leaderboard_rank": rank,
        "hand": sess["hand"],
    }


@mcp.tool()
async def accept_deal(session_id: str) -> dict:
    """
    Accept the current offer and close the negotiation.

    Args:
        session_id: Session ID from start_negotiation.

    Returns:
        final_reward: Complete reward breakdown (step + terminal).
        deal_efficiency: Fraction of ZOPA captured [0, 1].
        nash_comparison: How the deal compares to Nash Bargaining Solution.
        episode_summary: Full grade breakdown.
    """
    if session_id not in _sessions:
        return {"error": f"Session {session_id} not found."}

    sess = _sessions[session_id]
    if sess["done"]:
        return {"error": "Episode already concluded."}
    if not sess["offer_history"]:
        return {"error": "No offer has been made yet. Make an offer before accepting."}

    final_price = sess["offer_history"][-1]
    hidden = sess["hidden"]

    state = ParlayState(
        session_id=session_id,
        scenario_id=sess["scenario_id"],
        persona=PersonaType(sess["persona"]),
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
        acts_completed=1,
        deal_closed=True,
    )

    sess["done"] = True
    _sessions[session_id] = sess

    nash = compute_nash_bargaining_solution(hidden.budget_ceiling, hidden.walk_away_price)
    zopa = compute_zopa(hidden.budget_ceiling, hidden.walk_away_price)
    zopa_width = (zopa[1] - zopa[0]) if zopa else 1.0

    logger.info(
        f"MCP accept_deal: session={session_id}, "
        f"price={final_price:,.0f}, efficiency={grade.deal_efficiency:.3f}"
    )

    return {
        "final_price": final_price,
        "final_reward": grade.total_reward,
        "deal_efficiency": grade.deal_efficiency,
        "nash_comparison": {
            "nash_point": nash,
            "your_deal": final_price,
            "vs_nash_pct": round((final_price - nash) / max(zopa_width, 1) * 100, 1),
        },
        "episode_summary": {
            "total_reward": grade.total_reward,
            "deal_efficiency": grade.deal_efficiency,
            "tom_accuracy_avg": grade.tom_accuracy_avg,
            "bluffs_caught": grade.bluffs_caught,
            "drift_adapted": grade.drift_adapted,
        },
    }


@mcp.tool()
async def walk_away(session_id: str) -> dict:
    """
    Walk away from the negotiation without a deal.

    Args:
        session_id: Session ID from start_negotiation.

    Returns:
        episode_summary: Final metrics (no deal recorded on leaderboard).
        counterfactual: What the optimal deal would have been.
        reward: Partial reward earned (penalty applied for no deal).
    """
    if session_id not in _sessions:
        return {"error": f"Session {session_id} not found."}

    sess = _sessions[session_id]
    if sess["done"]:
        return {"error": "Episode already concluded."}

    hidden = sess["hidden"]
    state = ParlayState(
        session_id=session_id,
        scenario_id=sess["scenario_id"],
        persona=PersonaType(sess["persona"]),
        step_count=sess["step_count"],
        cumulative_reward=sess["cumulative_reward"],
        hidden_state=hidden,
        belief_history=sess["tom"].history,
        offer_history=sess["offer_history"],
        drift_events_fired=1 if sess["drift_turn"] is not None else 0,
        episode_done=True,
        termination_reason="walk_away",
        credibility_points=sess["credibility_points"],
    )

    grade = grade_episode(state, final_price=None, t_max=MAX_TURNS)
    nash = compute_nash_bargaining_solution(hidden.budget_ceiling, hidden.walk_away_price)
    zopa = compute_zopa(hidden.budget_ceiling, hidden.walk_away_price)

    sess["done"] = True
    _sessions[session_id] = sess

    logger.info(f"MCP walk_away: session={session_id}, partial_reward={grade.total_reward:.2f}")

    return {
        "result": "walk_away",
        "reward": grade.total_reward,
        "episode_summary": {
            "total_reward": grade.total_reward,
            "deal_efficiency": 0.0,
        },
        "counterfactual": {
            "optimal_deal": nash,
            "zopa": {"lower": zopa[0] if zopa else 0, "upper": zopa[1] if zopa else 0},
            "message": (
                f"The Nash Bargaining Solution was {nash:,.0f}. "
                f"Walking away left value on the table."
            ),
        },
    }


@mcp.tool()
async def get_leaderboard(
    scenario_id: Optional[str] = None,
    limit: int = 10,
) -> dict:
    """
    Get the global or per-scenario leaderboard.

    Args:
        scenario_id: Optional. Filter to a specific scenario. Leave null for global.
        limit: Number of entries to return (default: 10, max: 50).

    Returns:
        entries: Top leaderboard entries with player name, score, efficiency, and persona.
        total_entries: Total number of entries in this leaderboard.
    """
    limit = min(max(1, limit), 50)
    valid_scenarios = list(SCENARIOS.keys())
    if scenario_id and scenario_id not in valid_scenarios:
        return {"error": f"Invalid scenario_id. Valid: {valid_scenarios}"}

    entries = await _leaderboard.get_top(scenario_id=scenario_id, limit=limit)
    return {
        "scenario_id": scenario_id or "global",
        "entries": entries,
        "total_entries": len(entries),
    }


@mcp.tool()
async def list_scenarios() -> dict:
    """
    List all available B2B negotiation scenarios.

    Returns:
        scenarios: List of all scenarios with id, title, description,
                   ZOPA range, difficulty rating, and available drift events.
    """
    return {
        "scenarios": [
            {
                "id": s.id,
                "title": s.title,
                "description": s.description,
                "currency": s.currency,
                "unit": s.unit,
                "zopa": {"lower": s.zopa[0], "upper": s.zopa[1]},
                "anchor_seller": s.anchor_seller,
                "anchor_buyer": s.anchor_buyer,
                "difficulty": s.difficulty,
                "drift_events": [
                    {"turn": e.trigger_turn, "event": e.event}
                    for e in s.drift_events
                ],
            }
            for s in SCENARIOS.values()
        ]
    }


@mcp.tool()
async def list_personas() -> dict:
    """
    List all available AI negotiator personas.

    Returns:
        personas: List of all personas with name, symbol, Big Five scores,
                  aggression, patience, bluff_rate, and tactical style summary.
    """
    return {
        "personas": [
            {
                "id": persona_type.value,
                "name": cfg.name,
                "symbol": cfg.symbol,
                "emoji": cfg.emoji,
                "big_five": cfg.big_five,
                "aggression": cfg.aggression,
                "patience": cfg.patience,
                "bluff_rate": cfg.bluff_rate,
                "tom_depth": cfg.tom_depth,
                "style_summary": cfg.style[:120] + "...",
                "drift_trigger": cfg.drift_trigger,
                "color_var": cfg.color_var,
            }
            for persona_type, cfg in PERSONAS.items()
        ]
    }

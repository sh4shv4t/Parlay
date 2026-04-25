"""
Self-play episode runner for Parlay.
CLI: python -m agent.runner --steps N --persona shark --scenario saas_enterprise
"""
import argparse
import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np

from parlay_env.grader import EpisodeGrade, compute_step_reward, grade_episode
from parlay_env.models import (
    BeliefState,
    HiddenState,
    ParlayAction,
    ParlayState,
    PersonaType,
    TacticalMove,
)

from agent.gemini_client import call_gemini, call_gemini_tom
from agent.personas import PERSONAS, build_system_prompt
from agent.tom_tracker import ToMTracker
from game.scenarios import get_scenario

TOM_DIAGNOSTIC = False  # Set False before full training run

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """Result from a single self-play episode."""

    session: ParlayState
    system_prompt: str
    conversation: list[dict]
    grade: EpisodeGrade
    final_price: Optional[float]


async def run_episode(
    persona: PersonaType = PersonaType.SHARK,
    scenario_id: str = "saas_enterprise",
    inject_noise: bool = False,
    force_drift: bool = False,
    seed: int = 42,
    max_turns: int = 20,
) -> EpisodeResult:
    """
    Run a single self-play negotiation episode.

    Per-turn step rewards are computed via compute_step_reward() and
    accumulated into state.cumulative_reward each turn so the terminal
    grader sees the full dense-reward signal.

    Args:
        persona:      Opponent persona.
        scenario_id:  Scenario to play.
        inject_noise: If True, inject random early moves for data diversity.
        force_drift:  If True, force a drift event to fire.
        seed:         Random seed for reproducibility.
        max_turns:    Maximum number of turns.

    Returns:
        EpisodeResult with session state, conversation, and grade.
    """
    rng = np.random.default_rng(seed)
    scenario = get_scenario(scenario_id)
    persona_cfg = PERSONAS[persona]

    noise = float(rng.uniform(0.95, 1.05))
    hidden = HiddenState(
        budget_ceiling=round(scenario.batna_buyer * noise, 2),
        walk_away_price=round(scenario.batna_seller * noise, 2),
        urgency_score=float(np.clip(0.5 + rng.uniform(-0.2, 0.2), 0.0, 1.0)),
        has_alternative=bool(rng.choice([True, False])),
        persona_drifted=False,
    )

    initial_belief = BeliefState(
        est_budget=hidden.budget_ceiling * 0.75,
        est_walk_away=hidden.walk_away_price * 1.20,
        est_urgency=0.50,
        est_has_alternative=False,
        confidence=0.25,
    )

    tom = ToMTracker(initial_belief, persona)
    session_id = str(uuid.uuid4())

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
        credibility_points=100,
    )

    system_prompt = build_system_prompt(
        persona=persona,
        scenario_id=scenario_id,
        scenario_title=scenario.title,
        scenario_description=scenario.description,
        batna=hidden.walk_away_price,
        budget=hidden.budget_ceiling,
        urgency=hidden.urgency_score,
    )

    conversation: list[dict] = []
    drift_adapted = False
    drift_turn: Optional[int] = None
    final_price: Optional[float] = None
    cumulative_reward: float = 0.0

    opening = persona_cfg.opening_line
    conversation.append({"role": "model", "content": opening, "turn": 0})

    forced_drift_turn = int(rng.integers(3, 8)) if force_drift else None

    for turn in range(max_turns):
        for event in scenario.drift_events:
            if event.trigger_turn == turn or (forced_drift_turn == turn):
                drift_turn = turn
                tom.drift_event(
                    event.effect_on_urgency,
                    event.effect_on_has_alternative,
                    event_description=event.event,
                )
                logger.info(f"Drift event at turn {turn}: {event.event!r}")
                break

        if inject_noise and turn < 3 and rng.random() < 0.3:
            random_move: Optional[TacticalMove] = TacticalMove(
                rng.choice([m.value for m in TacticalMove])
            )
        else:
            random_move = None

        agent_messages = [
            {"role": "user" if i % 2 == 0 else "model", "parts": [m["content"]]}
            for i, m in enumerate(conversation)
        ]
        current_offer_str = str(state.offer_history[-1]) if state.offer_history else "None"
        agent_messages.append({
            "role": "user",
            "parts": [
                f"Turn {turn + 1}. Make your move. "
                f"Current offer on table: {current_offer_str}"
            ],
        })

        # Always pass persona explicitly so mock mode uses the right responses
        agent_response = await call_gemini(
            system_prompt,
            agent_messages,
            persona=persona.value,
            scenario_id=scenario_id,
        )
        action = ParlayAction(
            utterance=agent_response.get("utterance", "..."),
            offer_amount=agent_response.get("offer_amount"),
            tactical_move=random_move or _parse_tactical_move(
                agent_response.get("tactical_move")
            ),
        )

        conversation.append({
            "role": "negotiator",
            "content": action.utterance,
            "offer": action.offer_amount,
            "move": action.tactical_move.value if action.tactical_move else None,
            "turn": turn + 1,
        })

        opponent_messages = agent_messages + [
            {"role": "user", "parts": [action.utterance]}
        ]
        opponent_response = await call_gemini(
            (
                f"You are the human buyer in this negotiation. "
                f"Respond naturally to the AI seller.\n"
                f"Scenario: {scenario.title}. "
                f"Your budget ceiling: {hidden.budget_ceiling:,.0f}"
            ),
            opponent_messages,
            persona=persona.value,
            scenario_id=scenario_id,
        )

        conversation.append({
            "role": "opponent",
            "content": opponent_response.get("utterance", "..."),
            "offer": opponent_response.get("offer_amount"),
            "turn": turn + 1,
        })

        tom.update(
            observed_offer=opponent_response.get("offer_amount"),
            observed_move=_parse_tactical_move(opponent_response.get("tactical_move")),
            utterance=opponent_response.get("utterance", ""),
            turn=turn,
        )
        if TOM_DIAGNOSTIC:
            tom.log_belief_snapshot(turn=turn)

        if drift_turn is not None and not drift_adapted and turn <= drift_turn + 2:
            adaptation_signals = ["understand", "noted", "given that", "considering"]
            matched = next(
                (s for s in adaptation_signals if s in action.utterance.lower()), None
            )
            if matched:
                drift_adapted = True
                logger.info(
                    f"drift_adapted=True at turn={turn} "
                    f"matched_phrase={matched!r} "
                    f"utterance_snippet={action.utterance[:80]!r}"
                )

        new_offers = list(state.offer_history)
        if action.offer_amount:
            new_offers.append(action.offer_amount)

        cp_delta = _get_cp_cost(action.tactical_move) if action.tactical_move else 0

        # Build next state (without cumulative_reward — computed below)
        next_state_fields = {
            **state.model_dump(),
            "step_count": turn + 1,
            "offer_history": new_offers,
            "belief_history": tom.history,
            "episode_done": turn + 1 >= max_turns,
            "termination_reason": "max_turns" if turn + 1 >= max_turns else None,
            "credibility_points": max(0, state.credibility_points + 5 - cp_delta),
        }
        next_state_tmp = ParlayState(**next_state_fields)

        # Compute per-step dense reward and accumulate
        step_reward = compute_step_reward(state, action, next_state_tmp)
        cumulative_reward += step_reward
        logger.debug(
            f"Turn {turn + 1}: step_reward={step_reward:.3f}, "
            f"cumulative={cumulative_reward:.3f}"
        )

        # Update state carrying forward the accumulated reward
        state = ParlayState(
            **{**next_state_fields, "cumulative_reward": cumulative_reward}
        )

        # Check for deal close (within 3% of each other)
        if action.offer_amount and opponent_response.get("offer_amount"):
            agent_offer = action.offer_amount
            opp_offer = float(opponent_response["offer_amount"])
            if abs(agent_offer - opp_offer) / max(agent_offer, 1) < 0.03:
                final_price = (agent_offer + opp_offer) / 2
                logger.info(f"Deal reached at {final_price:,.0f} on turn {turn + 1}")
                break

    grade = grade_episode(
        state,
        final_price=final_price,
        t_close=state.step_count if final_price else None,
        t_max=max_turns,
        drift_adapted=drift_adapted,
        bluffs_caught=tom.bluffs_detected,
    )

    logger.info(
        f"Episode done: scenario={scenario_id}, persona={persona.value}, "
        f"reward={grade.total_reward:.2f}, efficiency={grade.deal_efficiency:.3f}, "
        f"cumulative_step_reward={cumulative_reward:.3f}"
    )

    return EpisodeResult(
        session=state,
        system_prompt=system_prompt,
        conversation=conversation,
        grade=grade,
        final_price=final_price,
    )


def _parse_tactical_move(value: Optional[str]) -> Optional[TacticalMove]:
    """Parse tactical move from string, returning None if invalid."""
    if not value:
        return None
    try:
        return TacticalMove(value)
    except ValueError:
        return None


def _get_cp_cost(move: Optional[TacticalMove]) -> int:
    """Return the credibility-point cost for a tactical move."""
    costs: dict[TacticalMove, int] = {
        TacticalMove.ANCHOR_HIGH: 0,
        TacticalMove.BATNA_REVEAL: 20,
        TacticalMove.SILENCE: 5,
    }
    return costs.get(move, 0) if move else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Parlay self-play runner")
    parser.add_argument("--steps", type=int, default=20, help="Max turns per episode")
    parser.add_argument(
        "--persona", default="shark", choices=[p.value for p in PersonaType]
    )
    parser.add_argument("--scenario", default="saas_enterprise")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    async def _run() -> None:
        for i in range(args.episodes):
            result = await run_episode(
                persona=PersonaType(args.persona),
                scenario_id=args.scenario,
                seed=i,
                max_turns=args.steps,
            )
            print(
                f"\nEpisode {i + 1}: reward={result.grade.total_reward:.2f}, "
                f"efficiency={result.grade.deal_efficiency:.3f}, "
                f"deal={'YES' if result.final_price else 'NO'}"
            )

    asyncio.run(_run())


if __name__ == "__main__":
    main()

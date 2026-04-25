"""Random-action baseline for Parlay."""
import argparse
import asyncio
import json
import logging
import random
from pathlib import Path

from parlay_env.grader import grade_episode
from parlay_env.models import TacticalMove
from parlay_env.server import _handle_reset, _handle_step, get_session_state

logger = logging.getLogger(__name__)

PERSONAS = ["shark", "diplomat", "veteran"]
SCENARIOS = ["saas_enterprise", "hiring_package", "acquisition_term_sheet"]
RANDOM_LINES = [
    "Let's keep talking.",
    "I can move a bit.",
    "This is my proposal.",
    "We should find middle ground.",
    "Given that context, here's my number.",
]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


async def _run_single_episode(scenario_id: str, persona: str, seed: int) -> dict:
    random.seed(seed)
    reset = await _handle_reset({"scenario_id": scenario_id, "persona": persona, "seed": seed})
    session_id = str(reset["session_id"])
    final_price = None
    t_close = None
    done = False

    while not done:
        state = get_session_state(session_id)
        if state is None:
            break
        if state.episode_done:
            break

        low = state.hidden_state.walk_away_price
        high = state.hidden_state.budget_ceiling
        offer = round(random.uniform(low, high), 2)

        moves: list[TacticalMove | None] = [None]
        if state.credibility_points >= 0:
            moves.append(TacticalMove.ANCHOR_HIGH)
        if state.credibility_points >= 5:
            moves.append(TacticalMove.SILENCE)
        if state.credibility_points >= 20:
            moves.append(TacticalMove.BATNA_REVEAL)
        move = random.choice(moves)

        payload = {
            "session_id": session_id,
            "action": {
                "utterance": random.choice(RANDOM_LINES),
                "offer_amount": offer,
                "tactical_move": move.value if move else None,
            },
        }
        step = await _handle_step(payload)
        done = bool(step.get("done", False))

        state = get_session_state(session_id)
        if state and state.deal_reached and final_price is None:
            final_price = offer
            t_close = state.step_count

    state = get_session_state(session_id)
    if state is None:
        raise RuntimeError(f"Missing session state for {session_id}")
    grade = grade_episode(state, final_price=final_price, t_close=t_close, t_max=20)
    return {
        "avg_reward": float(grade.total_reward),
        "deal_rate": 1.0 if final_price is not None else 0.0,
        "avg_efficiency": float(grade.deal_efficiency),
        "avg_tom_accuracy": float(grade.tom_accuracy_avg),
        "bluffs_caught": int(grade.bluffs_caught),
    }


async def _run_baseline(episodes: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(episodes):
        persona = PERSONAS[i % len(PERSONAS)]
        scenario = SCENARIOS[(i // len(PERSONAS)) % len(SCENARIOS)]
        try:
            rows.append(await _run_single_episode(scenario, persona, i + 7))
        except Exception as exc:
            logger.warning("Baseline episode %d failed (%s/%s): %s", i + 1, scenario, persona, exc)
    return rows


def _summarise(rows: list[dict], episodes_requested: int) -> dict:
    if not rows:
        return {
            "episodes_requested": episodes_requested,
            "episodes_completed": 0,
            "avg_reward": 0.0,
            "deal_rate": 0.0,
            "avg_efficiency": 0.0,
            "avg_tom_accuracy": 0.0,
            "bluffs_caught": 0,
        }
    return {
        "episodes_requested": episodes_requested,
        "episodes_completed": len(rows),
        "avg_reward": round(_mean([r["avg_reward"] for r in rows]), 4),
        "deal_rate": round(_mean([r["deal_rate"] for r in rows]), 4),
        "avg_efficiency": round(_mean([r["avg_efficiency"] for r in rows]), 4),
        "avg_tom_accuracy": round(_mean([r["avg_tom_accuracy"] for r in rows]), 4),
        "bluffs_caught": int(sum(r["bluffs_caught"] for r in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parlay random baseline")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", default="results/baseline.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    rows = asyncio.run(_run_baseline(args.episodes))
    summary = _summarise(rows, args.episodes)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved random baseline to {out_path.resolve()}")


if __name__ == "__main__":
    main()

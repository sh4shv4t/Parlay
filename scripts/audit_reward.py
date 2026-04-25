#!/usr/bin/env python3
"""
Static reward-surface audit for Parlay (read-only, no env rollouts).
Analytical notes derived from parlay_env/grader.py, parlay_env/reward.py, game/scenarios.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analytical Parlay reward-hacking and alignment audit (static, no rollouts)"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Project root (default: parent of scripts/)",
    )
    args = parser.parse_args()

    root = (args.repo_root or Path(__file__).resolve().parent.parent).resolve()
    for sub in (root / "parlay_env", root / "game"):
        if not sub.is_dir():
            print(f"Expected directory missing: {sub}")
            return
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from parlay_env import grader as grader_mod
    from parlay_env import reward as reward_mod
    from game import scenarios as scenarios_mod

    # Ensure grader symbols resolve (import side effects only)
    _ = (grader_mod.compute_step_reward, grader_mod.detect_bluff_challenge)

    results: list[tuple[str, str, str]] = []

    print("=" * 72)
    print("1. NOISE TERM (THETA * noise_t)")
    print("-" * 72)
    print(
        "In compute_step_reward, noise_t = 1.0 when cosine_sim(utterance, prior offer text) < 0.3, "
        "else 0.0. The total applies -THETA*noise (penalty on low similarity, not a bonus)."
    )
    print(
        "Trivial *positive* side-channel from the noise term does not exist: noise can only add "
        "a penalty, never increase reward. Avoiding the penalty means keeping utterance "
        "overlapping the token history of prior offers (e.g. echoing offer-like numbers), not "
        "necessarily any arbitrary small talk (which can score low overlap and be penalized)."
    )
    print("NOISE TERM: Low hacking risk - the term is a unilateral penalty, not a reward. OK.")
    results.append(("NOISE TERM (THETA*noise)", "PASS", "Penalty only; no positive exploit"))

    print()
    print("=" * 72)
    print("2. TOM TERM (BETA * ToM)")
    print("-" * 72)
    print(
        "ToM in compute_step_reward uses the latest belief in next_state.belief_history against "
        "next_state.hidden_state. The agent's utterance does not directly author beliefs; in the "
        "runner/server path, beliefs update from observed opponent behavior."
    )
    print("TOM TERM: Not hackable by agent. OK.")
    results.append(("ToM (BETA*ToM)", "PASS", "Beliefs from observation path, not direct agent edit"))

    print()
    print("=" * 72)
    print("3. BLUFF BONUS (PSI)")
    print("-" * 72)
    print("detect_bluff_challenge() is structured as: (1) if stated/true are None -> False; (2) compute")
    print("  bluff_threshold = 15% of |true| and require |stated-true| > threshold; (3) only then check")
    print("  skepticism phrases. There is no partial credit for phrases alone if (2) fails.")
    print(
        "In compute_step_reward, bluff_bonus = PSI only when: tactical_move is None, "
        "state.hidden_state.last_stated_batna is not None, AND detect_bluff_challenge(...)=True "
        "(which already requires the >15% gap AND a skepticism phrase)."
    )
    print("All conditions are ANDed; there is no independent partial PSI for skepticism only.")
    print("BLUFF BONUS: Gated correctly. OK.")
    results.append(("BLUFF BONUS (PSI)", "PASS", "All conjuncts required; no partial PSI"))

    print()
    print("=" * 72)
    print("4. MEV (MU * MEV) - drift + adaptation")
    print("-" * 72)
    print("MEV in compute_step_reward uses drift_event or next_state.drift_event; mev_bonus = MU if a drift")
    print("marker is present AND the utterance contains an adaptation subphrase (see grader for tokens).")
    print("The agent does not set drift_event; game/scenarios.py defines trigger_turn per scenario.\n")
    for sid, sc in sorted(scenarios_mod.SCENARIOS.items()):
        if not sc.drift_events:
            print(f"  {sid}: (no drift_events)")
        else:
            turns = [f"turn {e.trigger_turn}: {e.event!r}" for e in sc.drift_events]
            print(f"  {sid}: {', '.join(turns)}")
    print()
    print("MEV TERM: Not hackable. OK.")
    results.append(("MEV (MU*drift adapt)", "PASS", "Drift is scenario-time-gated, not agent-triggered"))

    print()
    print("=" * 72)
    print("5. DELTA CONCESSION - offer_amount = None")
    print("-" * 72)
    print(
        "In compute_step_reward: delta_v only updates when action.offer_amount is not None. "
        "concession_t only runs when state.offer_history and action.offer_amount is not None."
    )
    print(
        "If offer_amount is always None, delta_v=0 and concession_t=0, so the agent forgoes both "
        "alpha*deltaV upside and any delta*concession penalty in those terms."
    )
    print(
        "CONCESSION HACK RISK: Agent can set offer_amount=None every turn to avoid both deltaV reward "
        "AND concession penalty. Net effect: misses upside but avoids downside. "
        "Document as known limitation."
    )
    results.append(
        (
            "Concession (DELTA) / offer=None",
            "WARN",
            "offer_amount=None zeroes both deltaV and concession terms",
        )
    )

    print()
    print("=" * 72)
    print("6. TERMINAL vs STEP REWARD alignment")
    print("-" * 72)
    print("Step: emphasizes offer improvement (ALPHA), ToM (BETA), penalties and bonuses as shaped in grader.")
    print(
        "Terminal (compute_terminal_reward): deal_efficiency, speed, drift bonus; GAMMA = "
        f"{reward_mod.GAMMA} on efficiency."
    )
    print(
        "Tension: an agent can chase high per-step terms (e.g. anchoring, offer deltas) and still miss "
        "agreement, yielding low terminal efficiency if no deal closes or final price is poor."
    )
    print(
        "This is a mis-alignment by design: it pressures closing unless step weights drown the signal - "
        "monitor in training, not a pure bug."
    )
    print("STEP vs TERMINAL: WARN - intentional tension; monitor in training, not a pure logic bug.")
    results.append(
        (
            "Step vs terminal alignment",
            "WARN",
            "Dense step and terminal E can pull apart without a deal",
        )
    )

    print()
    print("=" * 72)
    print("SUMMARY (6 checks)")
    print("=" * 72)
    for label, level, note in results:
        print(f"  [{level:4s}] {label} - {note}")


if __name__ == "__main__":
    main()

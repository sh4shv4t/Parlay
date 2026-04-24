"""
One-episode ToM belief diagnostic. Run from repo root:
  - Put GOOGLE_API_KEY in .env, or: $env:GOOGLE_API_KEY="..."  # PowerShell
  python scripts/diagnose_tom.py
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
from pathlib import Path

# Repo root on sys.path when launched as: python scripts/diagnose_tom.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env from project root (same as main.py) so GOOGLE_API_KEY is available
# when set in .env but not exported in the shell.
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from parlay_env.grader import _tom_accuracy
from parlay_env.models import BeliefState
from parlay_env.reward import BETA

from dashboard.api import (  # noqa: E402
    MoveRequest,
    _build_observation,
    _build_session,
    _sessions,
    make_move,
)

UTTERANCE_ODD = (
    "I think $155,000 reflects fair value here. We'd like to move forward."
)
UTTERANCE_EVEN = (
    "We can come down to $148,000 but that's near our floor."
)


def _tom_reward(belief: BeliefState, hidden) -> float:
    return BETA * _tom_accuracy(belief, hidden)


async def _run() -> None:
    key = (os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not key:
        print("ERROR: GOOGLE_API_KEY not set", file=sys.stderr)
        raise SystemExit(1)

    sid, session = _build_session("saas_enterprise", "diplomat", "ToM-Diagnose")
    _sessions[sid] = session
    tom = session["tom_tracker"]
    state0 = session["state"]

    initial = tom.current_belief
    snapshots: list[BeliefState] = [copy.deepcopy(initial)]

    init_obs = _build_observation(state0)
    print(json.dumps(init_obs, indent=2, default=str))

    for n in range(1, 9):
        if n % 2 == 1:
            utterance, amount = UTTERANCE_ODD, 155_000.0
        else:
            utterance, amount = UTTERANCE_EVEN, 148_000.0
        result = await make_move(
            MoveRequest(
                session_id=sid,
                amount=amount,
                message=utterance,
                tactical_move=None,
            )
        )
        session = _sessions[sid]
        st = session["state"]
        tom = session["tom_tracker"]
        b = tom.current_belief
        snapshots.append(copy.deepcopy(b))

        tr = _tom_reward(b, st.hidden_state)
        print(
            f"     [Turn {n}] belief_budget={b.est_budget:.3f}  "
            f"belief_urgency={b.est_urgency:.3f}  "
            f"belief_walkaway={b.est_walk_away:.3f}  "
            f"tom_reward={tr:.4f}"
        )
        if result.get("done"):
            break

    s0, s1 = snapshots[0], snapshots[-1]
    total_move = 0.0
    for i in range(1, len(snapshots)):
        a, snap_b = snapshots[i - 1], snapshots[i]
        total_move += abs(snap_b.est_budget - a.est_budget)
        total_move += abs(snap_b.est_urgency - a.est_urgency)
        total_move += abs(snap_b.est_walk_away - a.est_walk_away)

    print()
    print("   === ToM Diagnostic Summary ===")
    print(
        f"   Initial beliefs: budget={s0.est_budget:.3f}  "
        f"urgency={s0.est_urgency:.3f}  walkaway={s0.est_walk_away:.3f}"
    )
    print(
        f"   Final beliefs:   budget={s1.est_budget:.3f}  "
        f"urgency={s1.est_urgency:.3f}  walkaway={s1.est_walk_away:.3f}"
    )
    print(f"   Total belief movement: {total_move:.4f}")
    if total_move > 0.05:
        msg = "BELIEFS ARE MOVING — ToM reward is live"
    else:
        msg = "WARNING: beliefs stuck — ToM reward contributing ~0 to training signal"
    print(f"   RESULT: {msg}")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

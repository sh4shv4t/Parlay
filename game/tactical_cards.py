"""Reduced tactical card set for Parlay."""
from dataclasses import dataclass

import numpy as np

from parlay_env.models import TacticalMove


@dataclass(frozen=True)
class TacticalCard:
    id: str
    name: str
    cp_cost: int
    description: str
    theory: str
    game_theory_ref: str


TACTICAL_CARDS: dict[str, TacticalCard] = {
    "anchor_high": TacticalCard(
        id="anchor_high",
        name="Anchor High",
        cp_cost=0,
        description="Open with an extreme position. Sets the reference point for the entire negotiation.",
        theory="Tversky & Kahneman (1974) - anchoring coefficient 0.65",
        game_theory_ref="anchoring_effect",
    ),
    "batna_reveal": TacticalCard(
        id="batna_reveal",
        name="BATNA Reveal",
        cp_cost=20,
        description="Reveal your walk-away price. Truth or bluff. High risk, high reward.",
        theory="Schelling (1960) - credible commitment devices",
        game_theory_ref="batna_commitment",
    ),
    "silence": TacticalCard(
        id="silence",
        name="Silence",
        cp_cost=5,
        description="Say nothing. Signals strong BATNA. The Veteran's signature weapon.",
        theory="Information revelation through strategic inaction",
        game_theory_ref="strategic_silence",
    ),
}


def get_card(move: TacticalMove | str) -> TacticalCard:
    """Get a TacticalCard by move enum or id."""
    key = move.value if isinstance(move, TacticalMove) else move
    return TACTICAL_CARDS[key]


def draw_hand(n: int = 3, rng_seed: int = 42) -> list[TacticalMove]:
    """Draw a subset of the three tactical cards."""
    rng = np.random.default_rng(rng_seed)
    moves = list(TacticalMove)
    indices = rng.choice(len(moves), size=min(n, len(moves)), replace=False)
    return [moves[i] for i in indices]

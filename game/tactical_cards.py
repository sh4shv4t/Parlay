"""Tactical move cards for Parlay. Players hold 5 cards drawn at episode start."""
from dataclasses import dataclass, field

import numpy as np

from parlay_env.models import TacticalMove


@dataclass(frozen=True)
class TacticalCard:
    name: str
    symbol: str
    cp_cost: int
    description: str
    game_theory_basis: str
    persona_effectiveness: dict[str, str] = field(default_factory=dict)


TACTICAL_CARDS: dict[TacticalMove, TacticalCard] = {
    TacticalMove.ANCHOR_HIGH: TacticalCard(
        name="Anchor High",
        symbol="⬆",
        cp_cost=0,
        description="Open with an extreme first offer. Sets the reference point.",
        game_theory_basis=(
            "Anchoring & adjustment heuristic (Tversky & Kahneman, 1974). "
            "First number dominates final settlement by ~35%."
        ),
        persona_effectiveness={
            "shark": "medium", "diplomat": "low", "analyst": "low",
            "wildcard": "high", "veteran": "low",
        },
    ),
    TacticalMove.BATNA_REVEAL: TacticalCard(
        name="BATNA Reveal",
        symbol="⚑",
        cp_cost=20,
        description="Reveal your walk-away point. Can be truth or bluff.",
        game_theory_basis=(
            "Credible commitment device (Schelling, 1960). "
            "Truthful reveal shifts Nash solution; detected bluff costs −25."
        ),
        persona_effectiveness={
            "shark": "medium", "diplomat": "high", "analyst": "high",
            "wildcard": "medium", "veteran": "low",
        },
    ),
    TacticalMove.COALITION_INVITE: TacticalCard(
        name="Coalition Invite",
        symbol="◉",
        cp_cost=30,
        description="Bring a third party. Unlocks Act 3. Changes Shapley values.",
        game_theory_basis=(
            "Coalitional game theory. Shapley value redistribution "
            "opens new Pareto improvements."
        ),
        persona_effectiveness={
            "shark": "low", "diplomat": "high", "analyst": "medium",
            "wildcard": "high", "veteran": "medium",
        },
    ),
    TacticalMove.TIME_PRESSURE: TacticalCard(
        name="Time Pressure",
        symbol="◷",
        cp_cost=15,
        description="Impose a deadline. Works on Wildcards, backfires on Veterans.",
        game_theory_basis=(
            "Deadline effect in alternating-offers bargaining "
            "(Rubinstein, 1982). Impatience asymmetry drives concessions."
        ),
        persona_effectiveness={
            "shark": "medium", "diplomat": "low", "analyst": "low",
            "wildcard": "high", "veteran": "backfires",
        },
    ),
    TacticalMove.SWEETENER: TacticalCard(
        name="Sweetener",
        symbol="◌",
        cp_cost=10,
        description="Add a non-price concession without moving your offer.",
        game_theory_basis=(
            "Integrative bargaining / expanding the pie (Raiffa, 1982). "
            "Pareto improvements that leave both sides better off."
        ),
        persona_effectiveness={
            "shark": "low", "diplomat": "high", "analyst": "medium",
            "wildcard": "high", "veteran": "medium",
        },
    ),
    TacticalMove.SILENCE: TacticalCard(
        name="Silence",
        symbol="—",
        cp_cost=5,
        description="Say nothing. Pressure fills the void. The Veteran's weapon.",
        game_theory_basis=(
            "Information revelation through inaction. "
            "Silence credibly signals a strong BATNA."
        ),
        persona_effectiveness={
            "shark": "medium", "diplomat": "medium", "analyst": "low",
            "wildcard": "high", "veteran": "high",
        },
    ),
    TacticalMove.REFRAME: TacticalCard(
        name="Reframe",
        symbol="↻",
        cp_cost=12,
        description="Change the reference point. Price-per-year → price-per-seat.",
        game_theory_basis=(
            "Prospect theory framing (Kahneman & Tversky, 1979). "
            "Losses loom larger than gains — reframe to gains."
        ),
        persona_effectiveness={
            "shark": "medium", "diplomat": "high", "analyst": "medium",
            "wildcard": "high", "veteran": "medium",
        },
    ),
}


def get_card(move: TacticalMove) -> TacticalCard:
    """Get a TacticalCard by move enum."""
    return TACTICAL_CARDS[move]


def draw_hand(n: int = 5, rng_seed: int = 42) -> list[TacticalMove]:
    """Draw n random tactical cards for a player's starting hand."""
    rng = np.random.default_rng(rng_seed)
    moves = list(TacticalMove)
    indices = rng.choice(len(moves), size=min(n, len(moves)), replace=False)
    return [moves[i] for i in indices]

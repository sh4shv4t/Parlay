"""
Pure game-theoretic computations for Parlay.
All functions are deterministic: same input → same output.
No I/O. No API calls. No side effects.
"""
import itertools
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


def compute_zopa(
    batna_buyer: float,
    batna_seller: float,
) -> Optional[tuple[float, float]]:
    """
    Compute the Zone of Possible Agreement (ZOPA).

    ZOPA exists iff the buyer's maximum willingness to pay exceeds
    the seller's minimum acceptable price.

    Args:
        batna_buyer:  Buyer's BATNA — maximum they will pay.
        batna_seller: Seller's BATNA — minimum they will accept.

    Returns:
        (lower, upper) tuple if ZOPA exists, else None.
    """
    if batna_buyer > batna_seller:
        return (batna_seller, batna_buyer)
    return None


def compute_nash_bargaining_solution(
    batna_buyer: float,
    batna_seller: float,
) -> float:
    """
    Nash Bargaining Solution (closed-form for symmetric case).

    Maximises the product of surplus gains: (p - batna_seller)(batna_buyer - p).
    The unique solution is the midpoint: p* = (batna_buyer + batna_seller) / 2.

    Reference: Nash (1950), "The Bargaining Problem", Econometrica.

    Args:
        batna_buyer:  Buyer's BATNA (upper bound).
        batna_seller: Seller's BATNA (lower bound).

    Returns:
        Nash equilibrium price p*.
    """
    return (batna_buyer + batna_seller) / 2.0


def compute_pareto_frontier(
    offers: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Compute the Pareto-efficient frontier from a list of (buyer_util, seller_util) pairs.

    A point is Pareto-efficient iff no other point dominates it on both dimensions.

    Args:
        offers: List of (buyer_utility, seller_utility) tuples.

    Returns:
        Pareto-efficient subset, sorted by buyer_utility ascending.
    """
    if not offers:
        return []

    pareto: list[tuple[float, float]] = []
    for candidate in offers:
        dominated = any(
            other[0] >= candidate[0] and other[1] >= candidate[1] and other != candidate
            for other in offers
        )
        if not dominated:
            pareto.append(candidate)

    return sorted(pareto, key=lambda x: x[0])


def compute_shapley_value(
    coalition_values: dict[frozenset, float],
) -> dict[str, float]:
    """
    Compute Shapley values for fair division in Act 3 coalition scenarios.

    The Shapley value assigns each player their average marginal contribution
    across all possible orderings. O(2^n) — fine for n <= 4.

    Reference: Shapley (1953), "A Value for n-person Games".

    Args:
        coalition_values: Maps frozenset of player names → coalition value.
                          Must include the empty set: frozenset() → 0.0.

    Returns:
        Dict mapping player name → Shapley value.
    """
    all_players: set[str] = set()
    for coalition in coalition_values:
        all_players.update(coalition)

    n = len(all_players)
    players = sorted(all_players)
    shapley: dict[str, float] = {p: 0.0 for p in players}

    for player in players:
        for perm in itertools.permutations(players):
            idx = perm.index(player)
            predecessors = frozenset(perm[:idx])
            predecessors_with = predecessors | {player}
            marginal = (
                coalition_values.get(predecessors_with, 0.0)
                - coalition_values.get(predecessors, 0.0)
            )
            shapley[player] += marginal / math.factorial(n)

    return shapley


def offer_anchoring_effect(
    anchor: float,
    adjustment: float,
) -> float:
    """
    Model the cognitive anchoring bias on final price estimates.

    Based on Tversky & Kahneman (1974): people adjust insufficiently from an anchor.
    The empirical anchoring coefficient is ~0.65.

    estimate = anchor + adjustment * (1 - 0.65)

    Args:
        anchor:     The initial reference number seen first.
        adjustment: The rational adjustment away from the anchor.

    Returns:
        Estimated final price after anchoring bias.
    """
    ANCHORING_COEFFICIENT = 0.65
    return anchor + adjustment * (1.0 - ANCHORING_COEFFICIENT)


def compute_rubinstein_deadline_advantage(
    turns_remaining: int,
    discount_rate_self: float = 0.95,
    discount_rate_opponent: float = 0.95,
) -> float:
    """
    Rubinstein (1982) alternating-offers model: first-mover share.

    In the infinite-horizon game the first-mover equilibrium share is:
        share = (1 - delta_opponent) / (1 - delta_self * delta_opponent)

    With finite turns_remaining we discount: share * delta_self^(T - turns_remaining).

    Reference: Rubinstein (1982), "Perfect Equilibrium in a Bargaining Model", Econometrica.

    Args:
        turns_remaining:        Number of rounds left.
        discount_rate_self:     Per-turn discount factor for the proposing agent (0 < d < 1).
        discount_rate_opponent: Per-turn discount factor for the responding agent.

    Returns:
        First-mover's equilibrium share of the surplus ∈ (0, 1).
    """
    if turns_remaining <= 0:
        return 0.0
    if discount_rate_self <= 0 or discount_rate_self >= 1:
        raise ValueError(f"discount_rate_self must be in (0,1), got {discount_rate_self}")
    if discount_rate_opponent <= 0 or discount_rate_opponent >= 1:
        raise ValueError(f"discount_rate_opponent must be in (0,1), got {discount_rate_opponent}")

    denom = 1.0 - discount_rate_self * discount_rate_opponent
    if denom == 0:
        return 0.5  # degenerate case

    infinite_share = (1.0 - discount_rate_opponent) / denom
    # Discount for finite turns
    finite_share = infinite_share * (discount_rate_self ** max(0, turns_remaining - 1))
    return max(0.0, min(1.0, finite_share))

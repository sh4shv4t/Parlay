"""Achievement and badge system for Parlay."""
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Achievement:
    id: str
    name: str
    description: str
    icon: str
    condition: str  # human-readable condition description
    xp_reward: int


ACHIEVEMENTS: list[Achievement] = [
    Achievement(
        id="first_deal",
        name="First Close",
        description="Complete your first negotiation deal.",
        icon="🤝",
        condition="deal_closed == True",
        xp_reward=100,
    ),
    Achievement(
        id="perfect_efficiency",
        name="Perfect Capture",
        description="Achieve deal efficiency above 0.90.",
        icon="💎",
        condition="deal_efficiency >= 0.90",
        xp_reward=500,
    ),
    Achievement(
        id="speedster",
        name="Speed Closer",
        description="Close a deal in under 8 turns.",
        icon="⚡",
        condition="t_close <= 8",
        xp_reward=300,
    ),
    Achievement(
        id="mind_reader",
        name="Mind Reader",
        description="Achieve ToM accuracy above 0.85.",
        icon="🔮",
        condition="tom_accuracy_avg >= 0.85",
        xp_reward=400,
    ),
    Achievement(
        id="drift_master",
        name="Drift Master",
        description="Adapt to a drift event within 2 turns.",
        icon="🌊",
        condition="drift_adapted == True",
        xp_reward=250,
    ),
    Achievement(
        id="bluff_catcher",
        name="Bluff Catcher",
        description="Catch 3 or more opponent bluffs in one session.",
        icon="🎭",
        condition="bluffs_caught >= 3",
        xp_reward=350,
    ),
    Achievement(
        id="act3_unlocked",
        name="Coalition Builder",
        description="Reach Act 3 and invoke coalition dynamics.",
        icon="🌐",
        condition="acts_completed >= 3",
        xp_reward=450,
    ),
    Achievement(
        id="shark_slayer",
        name="Shark Slayer",
        description="Beat The Shark persona with efficiency >= 0.75.",
        icon="🦈",
        condition="persona == 'shark' and deal_efficiency >= 0.75",
        xp_reward=600,
    ),
    Achievement(
        id="all_personas",
        name="Full Roster",
        description="Complete a deal against every persona at least once.",
        icon="🏆",
        condition="distinct_personas_beaten == 5",
        xp_reward=1000,
    ),
    Achievement(
        id="veteran_vanquished",
        name="Veteran Vanquished",
        description="Out-negotiate The Veteran with efficiency >= 0.70.",
        icon="🧓",
        condition="persona == 'veteran' and deal_efficiency >= 0.70",
        xp_reward=700,
    ),
]

_ACHIEVEMENT_MAP: dict[str, Achievement] = {a.id: a for a in ACHIEVEMENTS}


class AchievementSystem:
    """Evaluates which achievements a player earned in an episode."""

    def evaluate(
        self,
        deal_closed: bool,
        deal_efficiency: float,
        t_close: Optional[int],
        tom_accuracy_avg: float,
        drift_adapted: bool,
        bluffs_caught: int,
        acts_completed: int,
        persona: str,
    ) -> list[Achievement]:
        """
        Evaluate and return all achievements earned.

        Args:
            deal_closed:      Whether the episode ended in a deal.
            deal_efficiency:  ZOPA capture fraction [0, 1].
            t_close:          Turn the deal closed (None if no deal).
            tom_accuracy_avg: Mean ToM accuracy.
            drift_adapted:    Whether agent adapted to a drift event.
            bluffs_caught:    Number of opponent bluffs caught.
            acts_completed:   Number of acts reached (1-3).
            persona:          Opponent persona string.

        Returns:
            List of Achievement objects earned this episode.
        """
        earned: list[Achievement] = []

        checks: list[tuple[str, bool]] = [
            ("first_deal",         deal_closed),
            ("perfect_efficiency", deal_closed and deal_efficiency >= 0.90),
            ("speedster",          deal_closed and t_close is not None and t_close <= 8),
            ("mind_reader",        tom_accuracy_avg >= 0.85),
            ("drift_master",       drift_adapted),
            ("bluff_catcher",      bluffs_caught >= 3),
            ("act3_unlocked",      acts_completed >= 3),
            ("shark_slayer",       persona == "shark" and deal_closed and deal_efficiency >= 0.75),
            ("veteran_vanquished", persona == "veteran" and deal_closed and deal_efficiency >= 0.70),
        ]

        for achievement_id, condition in checks:
            if condition and achievement_id in _ACHIEVEMENT_MAP:
                earned.append(_ACHIEVEMENT_MAP[achievement_id])
                logger.info(f"Achievement unlocked: {achievement_id}")

        return earned

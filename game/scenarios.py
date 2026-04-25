"""B2B negotiation scenarios for Parlay."""
from dataclasses import dataclass, field

from parlay_env.exceptions import InvalidScenarioError


@dataclass(frozen=True)
class DriftEvent:
    trigger_turn: int
    event: str
    effect_on_urgency: float
    effect_on_has_alternative: bool


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    description: str
    anchor_seller: float
    anchor_buyer: float
    batna_seller: float
    batna_buyer: float
    zopa: tuple[float, float]
    currency: str
    unit: str
    drift_events: list[DriftEvent] = field(default_factory=list)
    difficulty: int = 2  # 1=easy, 2=medium, 3=hard


SCENARIOS: dict[str, Scenario] = {
    "saas_enterprise": Scenario(
        id="saas_enterprise",
        title="Enterprise SaaS Contract",
        description="500-seat analytics platform. Buyer has two competing bids.",
        anchor_seller=180_000, anchor_buyer=110_000,
        batna_seller=125_000, batna_buyer=165_000,
        zopa=(125_000, 165_000), currency="USD", unit="annual contract value",
        difficulty=2,
        drift_events=[
            DriftEvent(trigger_turn=8, event="Competitor drops price 15%",
                       effect_on_urgency=-0.3, effect_on_has_alternative=True),
            DriftEvent(trigger_turn=14, event="Q-end deadline announced",
                       effect_on_urgency=0.4, effect_on_has_alternative=False),
        ],
    ),
    "hiring_package": Scenario(
        id="hiring_package",
        title="Senior Engineer Offer",
        description="Total comp negotiation: base + equity + signing bonus.",
        anchor_seller=240_000, anchor_buyer=180_000,
        # Widened 15% to improve deal rate in self-play data generation
        batna_seller=195_000, batna_buyer=264_500,
        zopa=(195_000, 264_500), currency="USD", unit="total annual comp",
        difficulty=2,
        drift_events=[
            # Delayed from 5 to 8 - early drift was destabilizing pre-anchor phase
            DriftEvent(
                trigger_turn=8,
                event="Competing offer received",
                effect_on_urgency=-0.25,
                effect_on_has_alternative=True,
            ),
        ],
    ),
    "acquisition_term_sheet": Scenario(
        id="acquisition_term_sheet",
        title="Startup Acquisition",
        description="Acqui-hire: valuation, retention packages, earnout structure.",
        anchor_seller=18_000_000, anchor_buyer=9_000_000,
        batna_seller=10_500_000, batna_buyer=16_000_000,
        zopa=(10_500_000, 16_000_000), currency="USD", unit="acquisition value",
        difficulty=3,
        drift_events=[
            DriftEvent(trigger_turn=7, event="Due diligence reveals tech debt",
                       effect_on_urgency=-0.2, effect_on_has_alternative=False),
            DriftEvent(trigger_turn=13, event="Second acquirer enters",
                       effect_on_urgency=-0.4, effect_on_has_alternative=True),
        ],
    ),
}


def get_scenario(scenario_id: str) -> Scenario:
    """Get a scenario by ID. Raises InvalidScenarioError if not found."""
    if scenario_id not in SCENARIOS:
        raise InvalidScenarioError(
            f"Unknown scenario: {scenario_id!r}. Valid: {list(SCENARIOS)}"
        )
    return SCENARIOS[scenario_id]

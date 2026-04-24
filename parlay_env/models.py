from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PersonaType(str, Enum):
    SHARK = "shark"
    DIPLOMAT = "diplomat"
    VETERAN = "veteran"


class TacticalMove(str, Enum):
    ANCHOR_HIGH = "anchor_high"
    BATNA_REVEAL = "batna_reveal"
    SILENCE = "silence"


class HiddenState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    budget_ceiling: float = Field(description="True max the opponent will pay")
    walk_away_price: float = Field(description="Opponent's true BATNA")
    urgency_score: float = Field(ge=0.0, le=1.0, description="How badly they need to close")
    has_alternative: bool = Field(description="Whether opponent has a competing offer")
    persona_drifted: bool = Field(default=False)
    last_stated_batna: Optional[float] = None
    event_impacts: Optional[dict[str, float]] = None


class BeliefState(BaseModel):
    """Agent's current belief about opponent hidden state (Theory of Mind)."""

    est_budget: float
    est_walk_away: float
    est_urgency: float = Field(ge=0.0, le=1.0)
    est_has_alternative: bool
    confidence: float = Field(ge=0.0, le=1.0, description="Overall belief confidence")


class ParlayObservation(BaseModel):
    step_count: int
    episode_done: bool
    current_offer: float
    opponent_offer: float
    zopa_lower: float
    zopa_upper: float
    nash_point: float = Field(description="Nash Bargaining Solution price")
    tension_score: float = Field(ge=0.0, le=100.0)
    belief_state: BeliefState
    last_utterance: str
    available_moves: list[TacticalMove]
    credibility_points: int = Field(ge=0, le=100)
    reward: float
    cumulative_reward: float
    drift_event: Optional[str] = None
    zopa_erosion_ticks: int = 0
    zopa_width_pct_remaining: float = 1.0


class ParlayAction(BaseModel):
    utterance: str
    offer_amount: Optional[float] = None
    tactical_move: Optional[TacticalMove] = None


class ParlayState(BaseModel):
    session_id: str
    scenario_id: str
    persona: PersonaType
    step_count: int
    cumulative_reward: float
    hidden_state: HiddenState
    belief_history: list[BeliefState]
    offer_history: list[float]
    drift_events_fired: int
    episode_done: bool
    termination_reason: Optional[str] = None
    credibility_points: int = 100
    tension_score: float = 0.0
    deal_reached: bool = False
    walk_away: bool = False
    high_tension_streak: int = 0
    zopa_erosion_ticks: int = 0
    zopa_width_pct_remaining: float = 1.0
    original_zopa_width: float = 0.0
    bluffs_caught: int = 0

"""Reward function constants. Import from here everywhere — never hardcode coefficients."""

# Per-step weights
ALPHA: float = 2.0    # offer improvement toward ZOPA midpoint
BETA: float  = 5.0    # theory-of-mind accuracy bonus (per turn)
DELTA: float = 3.0    # concession rate penalty
THETA: float = 10.0   # random/ungrounded offer penalty

# Terminal reward scalars
GAMMA: float   = 100.0  # deal efficiency scalar
EPSILON: float = 20.0   # max speed bonus
ZETA: float    = 15.0   # drift adaptation bonus
ETA: float     = 0.0    # retained for compatibility; single-act env disables it
OMEGA: float   = 200.0  # capitulation cliff (hard discontinuous penalty)
PSI: float     = 12.0   # bluff-caught bonus

# Game config defaults (overridden by env vars at runtime)
MAX_TURNS: int = 20
CP_START: int  = 100
CP_REGEN:  int = 5

# Dynamic ZOPA collapse
ZOPA_EROSION_RATE = 0.02
ZOPA_EROSION_TENSION_THRESHOLD = 75
ZOPA_EROSION_CONSECUTIVE_TURNS = 3

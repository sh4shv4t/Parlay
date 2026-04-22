class ParlayError(Exception):
    """Base exception for all Parlay errors."""


class InvalidActionError(ParlayError):
    """Raised when an invalid action is submitted to the environment."""


class SessionNotFoundError(ParlayError):
    """Raised when a session ID does not exist."""


class InsufficientCredibilityError(ParlayError):
    """Raised when a player tries to use a card they cannot afford."""


class InvalidScenarioError(ParlayError):
    """Raised when an unknown scenario ID is requested."""


class InvalidPersonaError(ParlayError):
    """Raised when an unknown persona is requested."""


class EpisodeAlreadyDoneError(ParlayError):
    """Raised when an action is submitted to a completed episode."""

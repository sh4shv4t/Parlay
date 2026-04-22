"""
Google Gemini 2.0 Flash client for Parlay.
Uses the google-genai SDK. All calls are async (via run_in_executor).
All errors return SYNTHETIC_RESPONSE.
When GOOGLE_API_KEY is absent, MOCK_RESPONSES are returned so the full game
loop works without any API key.
"""
import asyncio
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_ID = "gemini-2.0-flash"

_client = None
_mock_warned: bool = False

# ── Mock responses (keyless dev / CI) ────────────────────────────────────────

MOCK_RESPONSES: dict[str, list[dict]] = {
    "shark": [
        {"utterance": "Let's not waste each other's time. Here's where I stand — this number isn't moving much.", "offer_amount": None, "tactical_move": "anchor_high"},
        {"utterance": "I have three other parties at the table. You'd better sharpen your pencil.", "offer_amount": None, "tactical_move": "batna_reveal"},
        {"utterance": "That's a creative offer. Not good enough, but creative.", "offer_amount": None, "tactical_move": None},
        {"utterance": "I've been in rooms like this before. Come back with something real.", "offer_amount": None, "tactical_move": None},
        {"utterance": "Fine. One last move from me. This is the ceiling.", "offer_amount": None, "tactical_move": "anchor_high"},
    ],
    "diplomat": [
        {"utterance": "I believe we can find something that works for both of us. Let's explore the range.", "offer_amount": None, "tactical_move": None},
        {"utterance": "I appreciate you sharing that. Could we consider packaging some non-price elements?", "offer_amount": None, "tactical_move": "sweetener"},
        {"utterance": "We're actually closer than it seems. I'm willing to move if you can meet me halfway.", "offer_amount": None, "tactical_move": None},
        {"utterance": "Here's a revised proposal that I think reflects your concerns.", "offer_amount": None, "tactical_move": "reframe"},
        {"utterance": "I think we've built enough trust here. Let me share something that might help us close.", "offer_amount": None, "tactical_move": None},
    ],
    "analyst": [
        {"utterance": "I've modeled this extensively. The numbers you're presenting don't align with market benchmarks.", "offer_amount": None, "tactical_move": None},
        {"utterance": "Can you provide the data backing that position? I need to validate before proceeding.", "offer_amount": None, "tactical_move": None},
        {"utterance": "Based on comparable transactions, the fair value range is well-established.", "offer_amount": None, "tactical_move": "reframe"},
        {"utterance": "The variance in your offer exceeds two standard deviations from the median.", "offer_amount": None, "tactical_move": None},
        {"utterance": "I've run the numbers three ways. Here is the only figure that makes sense.", "offer_amount": None, "tactical_move": "anchor_high"},
    ],
    "wildcard": [
        {"utterance": "You know what? Let's just see where this goes. I feel good about today.", "offer_amount": None, "tactical_move": None},
        {"utterance": "Honestly I wasn't expecting that. Actually, you know what — here's a thought.", "offer_amount": None, "tactical_move": None},
        {"utterance": "Something you said changed my thinking entirely. I'm going a different direction.", "offer_amount": None, "tactical_move": "reframe"},
        {"utterance": "This is either a great deal or a terrible one. I genuinely can't tell. Let's find out.", "offer_amount": None, "tactical_move": None},
        {"utterance": "Sure, why not. But I want something extra thrown in — something creative.", "offer_amount": None, "tactical_move": "sweetener"},
    ],
    "veteran": [
        {"utterance": "I've been in this room many times. I know what fair looks like and that isn't it.", "offer_amount": None, "tactical_move": None},
        {"utterance": "…I'll give you a moment to reconsider that position.", "offer_amount": None, "tactical_move": "silence"},
        {"utterance": "I've seen every tactic in the book. Let's cut to what we're both actually thinking.", "offer_amount": None, "tactical_move": None},
        {"utterance": "Experience tells me we're about three moves from a deal.", "offer_amount": None, "tactical_move": "reframe"},
        {"utterance": "You're good. But I've been doing this longer. Here's my considered response.", "offer_amount": None, "tactical_move": None},
    ],
}

SYNTHETIC_RESPONSE: dict = {
    "utterance": "I need a moment to consider your proposal.",
    "offer_amount": None,
    "tactical_move": None,
}


# ── Helper: mock mode detection ───────────────────────────────────────────────

def _is_mock_mode() -> bool:
    """Return True when GOOGLE_API_KEY is absent or empty."""
    return not os.environ.get("GOOGLE_API_KEY", "").strip()


def _get_mock_response(persona: str, turn: int) -> dict:
    """
    Return a canned response for keyless dev.

    Args:
        persona: Persona key (shark / diplomat / analyst / wildcard / veteran).
        turn:    Current turn count — used to cycle through the 5 canned lines.

    Returns:
        Copy of the canned response dict.
    """
    global _mock_warned
    if not _mock_warned:
        logger.warning(
            "GOOGLE_API_KEY not set — using mock responses. "
            "Set key in .env for real AI."
        )
        _mock_warned = True

    responses = MOCK_RESPONSES.get(persona, MOCK_RESPONSES["shark"])
    return dict(responses[turn % len(responses)])


# ── Lazy client ──────────────────────────────────────────────────────────────

def _get_client():
    """Lazily construct the Gemini client (import deferred so tests don't need SDK)."""
    global _client
    if _client is None:
        from google import genai  # noqa: PLC0415
        _client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY") or "")
    return _client


def _legacy_messages_to_history(messages: list[dict]) -> list:
    """Convert legacy {'role','parts'} messages to google-genai Content list."""
    from google.genai import types  # noqa: PLC0415

    contents: list = []
    for m in messages:
        role = m.get("role", "user")
        if role not in ("user", "model"):
            role = "user"
        raw_parts = m.get("parts") or []
        parts: list = []
        for p in raw_parts:
            text = p if isinstance(p, str) else str(p)
            parts.append(types.Part(text=text))
        if not parts:
            parts.append(types.Part(text=""))
        contents.append(types.Content(role=role, parts=parts))
    return contents


# ── Public API ───────────────────────────────────────────────────────────────

async def call_gemini(
    system_prompt: str,
    messages: list[dict],
    max_tokens: int = 500,
    persona: str = "shark",
) -> dict:
    """
    Call Gemini 2.0 Flash with a system prompt and message history.
    Returns mock responses when GOOGLE_API_KEY is not set.

    Args:
        system_prompt: Persona + scenario context string.
        messages:      List of {"role": "user"|"model", "parts": ["..."]} dicts.
        max_tokens:    Maximum output tokens for the response.
        persona:       Persona name used for mock-mode selection.

    Returns:
        Parsed dict with keys: utterance (str), offer_amount (float|None),
        tactical_move (str|None). Returns SYNTHETIC_RESPONSE on any error.
    """
    if _is_mock_mode():
        return _get_mock_response(persona, len(messages))

    text = ""
    try:
        from google.genai import types  # noqa: PLC0415

        history = messages[:-1] if len(messages) > 1 else []
        last_msg = messages[-1]["parts"][0] if messages else "Begin the negotiation."

        full_prompt = (
            f"{system_prompt}\n\n"
            f"Respond ONLY with valid JSON:\n"
            f'{{"utterance": "...", "offer_amount": <number or null>, '
            f'"tactical_move": <string or null>}}'
        )
        user_message = f"{full_prompt}\n\nUser: {last_msg}"

        def _call():
            chat = _get_client().chats.create(
                model=MODEL_ID,
                history=_legacy_messages_to_history(history),
            )
            return chat.send_message(
                user_message,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7,
                ),
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _call)

        text = (response.text or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)

        if "utterance" not in parsed:
            parsed["utterance"] = text[:200]

        parsed.setdefault("offer_amount", None)
        parsed.setdefault("tactical_move", None)

        logger.debug(f"Gemini response: utterance={parsed['utterance'][:60]!r}")
        return parsed

    except json.JSONDecodeError:
        logger.warning("Gemini JSON parse failed — using text fallback")
        raw = text[:300] if text else ""
        if raw:
            return {**SYNTHETIC_RESPONSE, "utterance": raw}
        return SYNTHETIC_RESPONSE
    except Exception as exc:
        logger.error(f"Gemini API error: {exc}")
        return SYNTHETIC_RESPONSE


async def call_gemini_tom(
    system_prompt: str,
    conversation_history: list[dict],
    current_state: dict,
) -> dict:
    """
    Call Gemini to infer Theory of Mind beliefs about the opponent.
    Returns current_state unchanged in mock mode.

    Args:
        system_prompt:        Base persona context.
        conversation_history: List of {"role", "parts"} messages.
        current_state:        Current belief state dict.

    Returns:
        Updated belief dict: {est_budget, est_walk_away, est_urgency,
                               est_has_alternative, confidence}.
    """
    if _is_mock_mode():
        return current_state

    tom_prompt = (
        f"{system_prompt}\n\n"
        f"THEORY OF MIND TASK:\n"
        f"Based on the negotiation so far, estimate your opponent's hidden state.\n"
        f"Current beliefs: {json.dumps(current_state)}\n\n"
        f"Respond ONLY with valid JSON:\n"
        f'{{"est_budget": <float>, "est_walk_away": <float>, '
        f'"est_urgency": <float 0-1>, "est_has_alternative": <bool>, '
        f'"confidence": <float 0-1>}}'
    )

    try:
        from google.genai import types  # noqa: PLC0415

        def _call():
            chat = _get_client().chats.create(
                model=MODEL_ID,
                history=_legacy_messages_to_history(conversation_history),
            )
            return chat.send_message(
                tom_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=200,
                    temperature=0.3,
                ),
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _call)
        text = (response.text or "").strip().replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as exc:
        logger.error(f"Gemini ToM inference error: {exc}")
        return current_state

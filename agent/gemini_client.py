"""
Google Gemini 2.0 Flash client for Parlay.
Uses the google-genai SDK. All calls are async (via run_in_executor). All errors return SYNTHETIC_RESPONSE.
"""
import asyncio
import json
import logging
import os
from typing import Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

MODEL_ID = "gemini-2.0-flash"

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    """Lazily construct API client (empty key is allowed; calls then fail and return synthetic output)."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY") or "")
    return _client


def _legacy_messages_to_history(messages: list[dict]) -> list[types.Content]:
    """Convert legacy {'role','parts'} messages to google-genai Content list."""
    contents: list[types.Content] = []
    for m in messages:
        role = m.get("role", "user")
        if role not in ("user", "model"):
            role = "user"
        raw_parts = m.get("parts") or []
        parts: list[types.Part] = []
        for p in raw_parts:
            text = p if isinstance(p, str) else str(p)
            parts.append(types.Part(text=text))
        if not parts:
            parts.append(types.Part(text=""))
        contents.append(types.Content(role=role, parts=parts))
    return contents


SYNTHETIC_RESPONSE: dict = {
    "utterance": "I need a moment to consider your proposal.",
    "offer_amount": None,
    "tactical_move": None,
}


async def call_gemini(
    system_prompt: str,
    messages: list[dict],
    max_tokens: int = 500,
) -> dict:
    """
    Call Gemini 2.0 Flash with a system prompt and message history.

    Args:
        system_prompt: Persona + scenario context string.
        messages:      List of {"role": "user"|"model", "parts": ["..."]} dicts.
        max_tokens:    Maximum output tokens for the response.

    Returns:
        Parsed dict with keys: utterance (str), offer_amount (float|None),
        tactical_move (str|None). Returns SYNTHETIC_RESPONSE on any error.
    """
    try:
        history = messages[:-1] if len(messages) > 1 else []
        last_msg = messages[-1]["parts"][0] if messages else "Begin the negotiation."

        full_prompt = (
            f"{system_prompt}\n\n"
            f"Respond ONLY with valid JSON:\n"
            f'{{"utterance": "...", "offer_amount": <number or null>, '
            f'"tactical_move": <string or null>}}'
        )
        user_message = f"{full_prompt}\n\nUser: {last_msg}"

        def _call() -> types.GenerateContentResponse:
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

    Args:
        system_prompt:        Base persona context.
        conversation_history: List of {"role", "parts"} messages.
        current_state:        Current belief state dict.

    Returns:
        Updated belief dict: {est_budget, est_walk_away, est_urgency,
                               est_has_alternative, confidence}.
    """
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
        def _call() -> types.GenerateContentResponse:
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

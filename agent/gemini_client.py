"""
Google Gemini 2.0 Flash client for Parlay.
All calls are async (via run_in_executor). All errors return SYNTHETIC_RESPONSE.
"""
import asyncio
import json
import logging
import os
from typing import Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
_model = genai.GenerativeModel("gemini-2.0-flash")

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

        loop = asyncio.get_event_loop()
        chat = _model.start_chat(history=history)

        response = await loop.run_in_executor(
            None,
            lambda: chat.send_message(
                f"{full_prompt}\n\nUser: {last_msg}",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7,
                ),
            ),
        )

        text = response.text.strip()
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
        try:
            return {**SYNTHETIC_RESPONSE, "utterance": response.text[:300]}
        except Exception:
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
        loop = asyncio.get_event_loop()
        chat = _model.start_chat(history=conversation_history)
        response = await loop.run_in_executor(
            None,
            lambda: chat.send_message(
                tom_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=200,
                    temperature=0.3,
                ),
            ),
        )
        text = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as exc:
        logger.error(f"Gemini ToM inference error: {exc}")
        return current_state

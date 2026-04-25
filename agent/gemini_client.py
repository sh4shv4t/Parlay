"""
Google Gemini client for Parlay.
Uses the google-genai SDK. All calls are async (via run_in_executor).
All errors return SYNTHETIC_RESPONSE.
When GOOGLE_API_KEY is absent, MOCK_RESPONSES are returned so the full game
loop works without any API key.

Model routing:
- MODEL_ID_DATA (gemini-2.5-flash-lite) — data generation, self-play, ToM inference.
  Low-latency, high-throughput; used by runner.py and generate_data.py.
- MODEL_ID_DEMO (gemini-2.5-flash) — web UI, dashboard API, MCP tools.
  Higher quality responses for live user interaction.
"""
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Optional

_live_calls: int = 0
_fallback_calls: int = 0
_turn_count: int = 0

logger = logging.getLogger(__name__)


def get_and_reset_counts() -> tuple[int, int]:
    global _live_calls, _fallback_calls
    counts = (_live_calls, _fallback_calls)
    _live_calls = 0
    _fallback_calls = 0
    return counts

# Scenario-aware role: the AI is always the *opponent*; the user is the *player* (buyer
# in SaaS, seller/candidate in hiring, seller in acquisition).
SCENARIO_ROLE_CONTEXT: dict[str, dict[str, str]] = {
    "saas_enterprise": {
        "ai_role": "seller (vendor)",
        "ai_goal": "Get the highest price possible. Push offers UP. Resist low offers.",
        "winning_direction": "higher",
        "currency_unit": "annual contract value in USD",
    },
    "hiring_package": {
        "ai_role": "employer (buyer of labor)",
        "ai_goal": "Hire the candidate at the lowest total comp possible. Push offers DOWN. Resist high asks.",
        "winning_direction": "lower",
        "currency_unit": "total annual compensation in USD",
    },
    "acquisition_term_sheet": {
        "ai_role": "acquirer (buyer)",
        "ai_goal": "Acquire the company at the lowest valuation possible. Push offers DOWN.",
        "winning_direction": "lower",
        "currency_unit": "acquisition valuation in USD",
    },
}

GEMINI_MODEL  = "gemini-2.5-flash-lite"   # kept for backward compat; equals MODEL_ID_DATA
MODEL_ID_DATA = "gemini-2.5-flash-lite"   # data generation, self-play, ToM inference
MODEL_ID_DEMO = "gemini-2.5-flash"        # web UI, dashboard API, MCP tools
MODEL_ID      = MODEL_ID_DATA             # stable alias (runner.py omits model= → flash-lite)

_client = None
_mock_warned: bool = False
_gemini_model_logged: bool = False
_quiet: bool = False   # suppresses per-call [Gemini LIVE] prints when True


def set_quiet(flag: bool) -> None:
    """Suppress [Gemini LIVE] per-call stderr prints (e.g. during test runs)."""
    global _quiet
    _quiet = flag

# ── Mock responses (keyless dev / CI) ────────────────────────────────────────
# Offer amounts are realistic for the default SaaS enterprise scenario
# (batna_seller ~125k, batna_buyer ~165k).  Per-persona sequences simulate
# a realistic negotiation arc: start near anchor, converge toward midpoint.

MOCK_RESPONSES: dict[str, list[dict]] = {
    "shark": [
        {"utterance": "Let's not waste each other's time. Here's where I stand — this number isn't moving much.", "offer_amount": 162000, "tactical_move": "anchor_high"},
        {"utterance": "I have three other parties at the table. You'd better sharpen your pencil.", "offer_amount": 158000, "tactical_move": "batna_reveal"},
        {"utterance": "That's a creative counter. I'll move — but only this far.", "offer_amount": 154000, "tactical_move": None},
        {"utterance": "I've been in rooms like this before. Last move from me.", "offer_amount": 150000, "tactical_move": None},
        {"utterance": "Fine. One last move from me. This is the ceiling.", "offer_amount": 147000, "tactical_move": "anchor_high"},
    ],
    "diplomat": [
        {"utterance": "I believe we can find something that works for both of us. Let's explore the range.", "offer_amount": 148000, "tactical_move": None},
        {"utterance": "I appreciate your position. If you can move a bit, I can narrow the gap too.", "offer_amount": 145000, "tactical_move": None},
        {"utterance": "We're actually closer than it seems. I'm willing to move if you can meet me halfway.", "offer_amount": 142000, "tactical_move": None},
        {"utterance": "Here's a revised proposal that I think reflects your concerns.", "offer_amount": 140000, "tactical_move": None},
        {"utterance": "I think we've built enough trust here. Let me share something that might help us close.", "offer_amount": 138000, "tactical_move": None},
    ],
    "veteran": [
        {"utterance": "I've been in this room many times. I know what fair looks like and that isn't it.", "offer_amount": 155000, "tactical_move": None},
        {"utterance": "…I'll give you a moment to reconsider that position.", "offer_amount": 152000, "tactical_move": "silence"},
        {"utterance": "I've seen every tactic in the book. Let's cut to what we're both actually thinking.", "offer_amount": 149000, "tactical_move": None},
        {"utterance": "Experience tells me we're about three moves from a deal.", "offer_amount": 146000, "tactical_move": None},
        {"utterance": "You're good. But I've been doing this longer. Here's my considered response.", "offer_amount": 144000, "tactical_move": None},
    ],
}

# When the AI is the *buyer* (employer, acquirer), mock offers must move DOWN vs typical asks.
_MOCK_BUYER_SCENARIOS = frozenset({"hiring_package", "acquisition_term_sheet"})

MOCK_RESPONSES_BUYER_AI: dict[str, list[dict[str, Any]]] = {
    "shark": [
        {"utterance": "We need a number that fits our comp band. Our opening position is on the table.", "offer_amount": 200_000, "tactical_move": "anchor_high"},
        {"utterance": "That's above what we can justify internally. We need you closer to our range.", "offer_amount": 205_000, "tactical_move": "batna_reveal"},
        {"utterance": "I can move slightly, but not to that level. Here's our revised cap.", "offer_amount": 208_000, "tactical_move": None},
        {"utterance": "We're at the top of the approved band. Take it or we walk.", "offer_amount": 210_000, "tactical_move": None},
        {"utterance": "Final number from us. This is the ceiling for this level.", "offer_amount": 212_000, "tactical_move": "anchor_high"},
    ],
    "diplomat": [
        {"utterance": "I appreciate the ask. Let me see what we can do within our framework.", "offer_amount": 202_000, "tactical_move": None},
        {"utterance": "We're a bit apart; here's a step toward you that's still responsible for us.", "offer_amount": 206_000, "tactical_move": None},
        {"utterance": "If you can flex on structure, we can improve the base slightly.", "offer_amount": 209_000, "tactical_move": None},
        {"utterance": "I want to get this done. This is the strongest package I can sign off on today.", "offer_amount": 211_000, "tactical_move": None},
        {"utterance": "This is our best and final for this role in this budget.", "offer_amount": 213_000, "tactical_move": None},
    ],
    "veteran": [
        {"utterance": "The ask is out of range for us. Counter:", "offer_amount": 201_000, "tactical_move": None},
        {"utterance": "…We need a lower number. Here's where we are.", "offer_amount": 204_000, "tactical_move": "silence"},
        {"utterance": "Not there yet. This is our position.", "offer_amount": 207_000, "tactical_move": None},
        {"utterance": "We're not going to chase that figure. Our line is here.", "offer_amount": 210_000, "tactical_move": None},
        {"utterance": "Last move on our side.", "offer_amount": 212_000, "tactical_move": None},
    ],
}

# Acquirer: offers in millions, trending down / cautious (still validated vs player amount in API)
MOCK_RESPONSES_BUYER_ACQ: dict[str, list[dict[str, Any]]] = {
    "shark": [
        {"utterance": "We won't lead with your number. Our valuation is firm on this side.", "offer_amount": 10_800_000, "tactical_move": "anchor_high"},
        {"utterance": "DD doesn't support that ask. We need a lower cap.", "offer_amount": 11_200_000, "tactical_move": "batna_reveal"},
        {"utterance": "We can nudge, not leap. Revised figure:", "offer_amount": 11_500_000, "tactical_move": None},
        {"utterance": "This is the range that works for our IC.", "offer_amount": 12_000_000, "tactical_move": None},
        {"utterance": "Last pass from the acquirer.", "offer_amount": 12_200_000, "tactical_move": None},
    ],
    "diplomat": [
        {"utterance": "Let's find a zone that works for both cap tables.", "offer_amount": 11_000_000, "tactical_move": None},
        {"utterance": "We can improve slightly if we align on key terms.", "offer_amount": 11_300_000, "tactical_move": None},
        {"utterance": "Here's a package we can defend internally.", "offer_amount": 11_600_000, "tactical_move": None},
        {"utterance": "I want a signed term sheet. This is our strongest number.", "offer_amount": 11_800_000, "tactical_move": None},
        {"utterance": "We don't have more room on headline price.", "offer_amount": 12_000_000, "tactical_move": None},
    ],
    "veteran": [
        {"utterance": "That number doesn't work for us. Counter:", "offer_amount": 10_900_000, "tactical_move": None},
        {"utterance": "We need a different magnitude.", "offer_amount": 11_200_000, "tactical_move": "silence"},
        {"utterance": "Still too rich. Here's us.", "offer_amount": 11_400_000, "tactical_move": None},
        {"utterance": "No further upside without structure changes.", "offer_amount": 11_700_000, "tactical_move": None},
        {"utterance": "This is the line.", "offer_amount": 12_000_000, "tactical_move": None},
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


def scenario_role_prompt_block(scenario_id: str) -> str:
    """Inject before persona style so the model knows which side it plays."""
    role_ctx = SCENARIO_ROLE_CONTEXT.get(
        scenario_id,
        {
            "ai_role": "negotiator",
            "ai_goal": "Negotiate effectively toward your side's interest.",
            "winning_direction": "appropriate",
            "currency_unit": "USD",
        },
    )
    return (
        "\nCRITICAL ROLE CONTEXT:\n"
        f"You are the {role_ctx['ai_role']} in this negotiation (you are the player's opponent).\n"
        f"Your goal: {role_ctx['ai_goal']}\n"
        f"When countering, move in the {role_ctx['winning_direction']} direction for you.\n"
        f"All amounts are in {role_ctx['currency_unit']}.\n"
        "NEVER offer more than the player's last stated number if you are a buyer (employer, acquirer).\n"
        "NEVER offer less than the player's last stated number if you are a seller (vendor). "
        "If unsure, match within a small margin instead of leaping past their number in the wrong direction.\n"
    )


def validate_ai_offer_direction(
    ai_offer: Optional[float],
    player_offer: Optional[float],
    scenario_id: str,
) -> Optional[float]:
    """
    Prevents the opponent (AI) from countering in the wrong direction.
    Returns corrected offer when invalid, or the original when valid/None.
    """
    if ai_offer is None or player_offer is None:
        return ai_offer
    tol = 0.05
    buyer_scenarios = {"hiring_package", "acquisition_term_sheet"}
    seller_scenarios = {"saas_enterprise"}
    if scenario_id in buyer_scenarios:
        if ai_offer > player_offer * (1.0 + tol):
            return max(0.0, player_offer * 0.97)
    elif scenario_id in seller_scenarios:
        if ai_offer < player_offer * (1.0 - tol):
            return player_offer * 1.03
    return float(ai_offer)


def _get_mock_response(persona: str, turn: int, scenario_id: Optional[str] = None) -> dict:
    """
    Return a canned response for keyless dev.

    Args:
        persona: Persona key (shark / diplomat / veteran).
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

    p = (persona or "shark").lower()
    if scenario_id in _MOCK_BUYER_SCENARIOS:
        if scenario_id == "acquisition_term_sheet":
            table = MOCK_RESPONSES_BUYER_ACQ
        else:
            table = MOCK_RESPONSES_BUYER_AI
        responses = table.get(p, table["shark"])
    else:
        responses = MOCK_RESPONSES.get(p, MOCK_RESPONSES["shark"])
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
    model: str | None = None,
    scenario_id: str | None = None,
) -> dict:
    """
    Call Gemini with a system prompt and message history.
    Returns mock responses when GOOGLE_API_KEY is not set.

    Args:
        system_prompt: Persona + scenario context string.
        messages:      List of {"role": "user"|"model", "parts": ["..."]} dicts.
        max_tokens:    Maximum output tokens for the response.
        persona:       Persona name used for mock-mode selection.
        model:         Model id, or None for GEMINI_MODEL (runner / data gen).

    Returns:
        Parsed dict with keys: utterance (str), offer_amount (float|None),
        tactical_move (str|None). Returns SYNTHETIC_RESPONSE on any error.
    """
    global _gemini_model_logged, _live_calls, _turn_count, _fallback_calls
    if _is_mock_mode():
        return _get_mock_response(persona, len(messages), scenario_id)

    if not _gemini_model_logged:
        logger.info(f"[Gemini] Using model: {GEMINI_MODEL}")
        _gemini_model_logged = True

    mid = model if model is not None else MODEL_ID_DATA
    text = ""
    try:
        from google.genai import types  # noqa: PLC0415
    except ModuleNotFoundError:
        logger.warning("google-genai SDK missing; falling back to mock response")
        return _get_mock_response(persona, len(messages), scenario_id)

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
            model=mid,
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
    backoff_delays = (1, 2, 4)
    for attempt in range(1, 4):
        try:
            response = await loop.run_in_executor(None, _call)

            _turn_count += 1
            _live_calls += 1
            if not _quiet:
                print(
                    f"[Gemini LIVE] model={mid} chars={len(response.text or '')} turn={_turn_count}",
                    file=sys.stderr,
                )

            text = (response.text or "").strip()
            text = text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)

            if "utterance" not in parsed:
                parsed["utterance"] = text[:200]

            parsed.setdefault("offer_amount", None)
            parsed.setdefault("tactical_move", None)

            logger.debug(
                f"Gemini model={mid} response: utterance={parsed['utterance'][:60]!r}"
            )
            return parsed
        except Exception as e:  # noqa: BLE001 — includes API, 429, JSON decode
            if attempt < 3:
                delay = backoff_delays[attempt - 1]
                print(
                    f"[GeminiClient] attempt {attempt} failed: {e}, retrying in {delay}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
            # attempt == 3: fall through to fallback below

    # Consolidated fallback: third failure or unhandled branch above
    print(
        "[GeminiClient] all 3 attempts failed, using fallback",
        file=sys.stderr,
    )
    logger.warning("Gemini API / parse failed after retries — using text fallback")
    _fallback_calls += 1
    if text:
        return {**SYNTHETIC_RESPONSE, "utterance": text[:300]}
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
                model=GEMINI_MODEL,
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

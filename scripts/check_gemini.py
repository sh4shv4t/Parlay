"""
One-shot smoke test: verify GOOGLE_API_KEY against the Gemini API.

Default model matches Parlay (`agent/gemini_client.py`: ``gemini-2.0-flash``).
Some keys (new accounts) get 404 on 2.0 Flash; this script then retries
``gemini-2.5-flash`` once. Override any time: ``set GEMINI_MODEL=gemini-2.5-flash``.

Usage (from repo root, with venv active):
    python scripts/check_gemini.py

Loads .env from the project root if python-dotenv is available.
Exits 0 on success, 1 on configuration/API error.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep in sync with agent/gemini_client.MODEL_ID
_DEFAULT_PARLAY_MODEL = "gemini-2.0-flash"
_FALLBACK_MODEL = "gemini-2.5-flash"


def _load_dotenv() -> None:
    root = Path(__file__).resolve().parent.parent
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]

        load_dotenv(root / ".env")
    except ImportError:
        pass


def main() -> int:
    _load_dotenv()
    key = (os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not key:
        print("error: GOOGLE_API_KEY is empty or not set (add to .env or the environment).", file=sys.stderr)
        return 1

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        print(
            f"error: google-genai is not installed: {exc}\n  pip install google-genai",
            file=sys.stderr,
        )
        return 1

    model = (os.environ.get("GEMINI_MODEL") or "").strip() or _DEFAULT_PARLAY_MODEL
    client = genai.Client(api_key=key)

    def _call(m: str):
        return client.models.generate_content(
            model=m,
            contents="Reply with exactly the single word: ok",
            config=types.GenerateContentConfig(
                max_output_tokens=16,
                temperature=0.0,
            ),
        )

    response = None
    last_err: Exception | None = None
    try:
        response = _call(model)
    except Exception as exc:  # noqa: BLE001 — surface API / auth errors to the user
        last_err = exc
        msg = str(exc).lower()
        if (
            model == _DEFAULT_PARLAY_MODEL
            and ("404" in str(exc) or "not_found" in msg)
            and _FALLBACK_MODEL not in msg
        ):
            print(
                f"note: {_DEFAULT_PARLAY_MODEL} not available for this key; "
                f"retrying {_FALLBACK_MODEL} (set GEMINI_MODEL to pin a model).",
                file=sys.stderr,
            )
            try:
                model = _FALLBACK_MODEL
                response = _call(model)
                last_err = None
            except Exception as exc2:  # noqa: BLE001
                last_err = exc2

    if last_err is not None:
        print(f"error: request failed: {type(last_err).__name__}: {last_err}", file=sys.stderr)
        return 1

    assert response is not None
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        print("error: empty response (check model name and account quota).", file=sys.stderr)
        return 1

    print(f"Model:    {model}")
    print(f"Response: {text!r}")
    print("ok: Gemini API responded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

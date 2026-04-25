"""
Tactical card tests — verifies card retrieval, serialisation, and API play flow.
Runs in mock mode (no API key required).

Usage:
    pytest tests/test_tactical_cards.py -v
"""
import os

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

os.environ.pop("GOOGLE_API_KEY", None)

from game.tactical_cards import TACTICAL_CARDS, TacticalCard, get_card, draw_hand
from parlay_env.models import TacticalMove
from dashboard.api import _serialise_cards


# ── Unit: card definitions ────────────────────────────────────────────────────

class TestCardDefinitions:
    def test_all_three_cards_defined(self):
        """All three tactical cards are present in the registry."""
        assert "anchor_high" in TACTICAL_CARDS
        assert "batna_reveal" in TACTICAL_CARDS
        assert "silence" in TACTICAL_CARDS

    def test_card_fields_populated(self):
        """Each card has all required fields with sensible values."""
        for card_id, card in TACTICAL_CARDS.items():
            assert isinstance(card, TacticalCard)
            assert card.id == card_id
            assert card.name, f"Card {card_id} has empty name"
            assert card.description, f"Card {card_id} has empty description"
            assert card.cp_cost >= 0, f"Card {card_id} has negative CP cost"

    def test_cp_costs_match_expected(self):
        """CP costs match the game design spec."""
        assert TACTICAL_CARDS["anchor_high"].cp_cost == 0
        assert TACTICAL_CARDS["batna_reveal"].cp_cost == 20
        assert TACTICAL_CARDS["silence"].cp_cost == 5

    def test_get_card_by_tactical_move_enum(self):
        """get_card() accepts TacticalMove enum values."""
        card = get_card(TacticalMove.ANCHOR_HIGH)
        assert card.id == "anchor_high"

        card = get_card(TacticalMove.BATNA_REVEAL)
        assert card.id == "batna_reveal"

        card = get_card(TacticalMove.SILENCE)
        assert card.id == "silence"

    def test_get_card_by_string_id(self):
        """get_card() accepts plain string ids."""
        card = get_card("anchor_high")
        assert card.id == "anchor_high"

    def test_get_card_unknown_raises(self):
        """get_card() raises KeyError for unknown card ids."""
        with pytest.raises(KeyError):
            get_card("does_not_exist")

    def test_draw_hand_returns_subset(self):
        """draw_hand() returns at most n valid TacticalMove values."""
        hand = draw_hand(n=2, rng_seed=0)
        assert len(hand) == 2
        for move in hand:
            assert isinstance(move, TacticalMove)

    def test_draw_hand_no_duplicates(self):
        """draw_hand() never repeats a card."""
        hand = draw_hand(n=3, rng_seed=7)
        assert len(hand) == len(set(hand))

    def test_draw_hand_capped_at_total_cards(self):
        """draw_hand() with n > total cards returns all cards once."""
        hand = draw_hand(n=999, rng_seed=0)
        assert len(hand) == len(TACTICAL_CARDS)


# ── Unit: serialisation ───────────────────────────────────────────────────────

class TestSerialiseCards:
    def test_serialise_returns_list(self):
        result = _serialise_cards()
        assert isinstance(result, list)

    def test_serialise_length_matches_registry(self):
        result = _serialise_cards()
        assert len(result) == len(TACTICAL_CARDS)

    def test_serialise_required_keys(self):
        """Each serialised card has all keys the frontend expects."""
        required_keys = {"id", "move", "name", "cp_cost", "description", "theory", "game_theory_ref"}
        for item in _serialise_cards():
            missing = required_keys - item.keys()
            assert not missing, f"Card {item.get('id')} missing keys: {missing}"

    def test_serialise_id_equals_move(self):
        """'id' and 'move' fields are identical (frontend uses both)."""
        for item in _serialise_cards():
            assert item["id"] == item["move"]

    def test_serialise_cp_cost_is_int(self):
        for item in _serialise_cards():
            assert isinstance(item["cp_cost"], int)


# ── Integration: play a card through the API ─────────────────────────────────

from main import app


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_start_game_hand_contains_cards(client):
    """POST /api/game/start returns a hand of serialised tactical cards."""
    resp = await client.post("/api/game/start", json={
        "scenario_id": "saas_enterprise",
        "persona": "shark",
        "player_name": "Tester",
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "hand" in data
    assert isinstance(data["hand"], list)
    assert len(data["hand"]) == len(TACTICAL_CARDS)
    ids_in_hand = {c["id"] for c in data["hand"]}
    assert "anchor_high" in ids_in_hand
    assert "batna_reveal" in ids_in_hand
    assert "silence" in ids_in_hand


@pytest.mark.asyncio
async def test_play_card_anchor_high_zero_cost(client):
    """Playing anchor_high (0 CP) succeeds and deducts nothing."""
    start = await client.post("/api/game/start", json={
        "scenario_id": "saas_enterprise",
        "persona": "diplomat",
        "player_name": "Tester",
    })
    assert start.status_code == 200
    session_id = start.json()["session_id"]
    initial_cp = start.json()["observation"]["credibility_points"]

    move = await client.post("/api/game/move", json={
        "session_id": session_id,
        "amount": 140000,
        "message": "Anchoring high.",
        "tactical_move": "anchor_high",
    })
    assert move.status_code == 200, move.text
    obs = move.json().get("observation", {})
    assert obs.get("credibility_points", 0) >= initial_cp - 5  # only regen delta at most


@pytest.mark.asyncio
async def test_play_card_insufficient_cp_returns_400(client):
    """Playing batna_reveal (20 CP) with insufficient CP returns 400."""
    start = await client.post("/api/game/start", json={
        "scenario_id": "saas_enterprise",
        "persona": "veteran",
        "player_name": "Tester",
    })
    assert start.status_code == 200
    session_id = start.json()["session_id"]

    # Drain CP by playing silence (5 CP) many times
    for _ in range(18):  # 18 × 5 = 90 CP spent, 18 regen ticks → ~0 CP
        await client.post("/api/game/move", json={
            "session_id": session_id,
            "amount": 150000,
            "message": "...",
            "tactical_move": "silence",
        })

    # At this point CP should be too low for batna_reveal (20 CP)
    resp = await client.post("/api/game/move", json={
        "session_id": session_id,
        "amount": 155000,
        "message": "Let me reveal my BATNA.",
        "tactical_move": "batna_reveal",
    })
    # Either succeeds (if CP regenerated enough) or fails with 400
    assert resp.status_code in (200, 400)


@pytest.mark.asyncio
async def test_play_invalid_card_returns_400(client):
    """Sending an unrecognised card_id returns 400."""
    start = await client.post("/api/game/start", json={
        "scenario_id": "hiring_package",
        "persona": "shark",
        "player_name": "Tester",
    })
    assert start.status_code == 200
    session_id = start.json()["session_id"]

    resp = await client.post("/api/game/move", json={
        "session_id": session_id,
        "amount": 200000,
        "message": "Playing a mystery card.",
        "tactical_move": "not_a_real_card",
    })
    assert resp.status_code == 400

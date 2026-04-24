"""
Keyless test suite — runs with zero API keys.
Tests the full stack in mock mode: game theory, grader, DB, and HTTP endpoints.

Usage:
    pytest tests/test_keyless.py -v
    make test-keyless
"""
import asyncio
import os
import sqlite3

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Ensure mock mode — remove any key that might be set in the environment
os.environ.pop("GOOGLE_API_KEY", None)

# ── App import after env is sanitised ────────────────────────────────────────
from main import app
from parlay_env.game_theory import (
    compute_zopa,
    compute_nash_bargaining_solution,
    compute_shapley_value,
)
from parlay_env.grader import compute_step_reward, compute_terminal_reward
from parlay_env.grader import detect_bluff_challenge
from parlay_env.reward import OMEGA, PSI
from parlay_env.models import (
    BeliefState, HiddenState, ParlayAction, ParlayState, PersonaType,
)
from dashboard.api import _apply_zopa_erosion


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def hidden() -> HiddenState:
    return HiddenState(
        budget_ceiling=165_000.0,
        walk_away_price=125_000.0,
        urgency_score=0.5,
        has_alternative=False,
        persona_drifted=False,
    )


@pytest.fixture
def belief() -> BeliefState:
    return BeliefState(
        est_budget=140_000.0,
        est_walk_away=130_000.0,
        est_urgency=0.5,
        est_has_alternative=False,
        confidence=0.5,
    )


@pytest.fixture
def parlay_state(hidden, belief) -> ParlayState:
    return ParlayState(
        session_id="test-session",
        scenario_id="saas_enterprise",
        persona=PersonaType.SHARK,
        step_count=0,
        cumulative_reward=0.0,
        hidden_state=hidden,
        belief_history=[belief],
        offer_history=[],
        drift_events_fired=0,
        episode_done=False,
        credibility_points=100,
    )


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Part 1: HTTP endpoints ────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_endpoint(self):
        """GET /health returns 200 with status ok."""
        async def _run():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/health")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            data = resp.json()
            assert data["status"] == "ok", f"Expected ok, got {data['status']}"
            assert "db" in data, f"Missing 'db' key: {data}"
            assert "gemini" in data, f"Missing 'gemini' key: {data}"
            assert data["gemini"] in (
                "mock",
                "configured",
            ), f"Expected gemini mock|configured, got {data['gemini']!r}"

        asyncio.get_event_loop().run_until_complete(_run())


class TestListScenarios:
    def test_list_scenarios(self):
        """GET /api/scenarios returns exactly 3 scenarios."""
        async def _run():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/api/scenarios")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            data = resp.json()
            scenarios = data.get("scenarios", [])
            assert len(scenarios) == 3, f"Expected 3 scenarios, got {len(scenarios)}"

        asyncio.get_event_loop().run_until_complete(_run())


class TestListPersonas:
    def test_list_personas(self):
        """GET /api/personas returns exactly 3 personas."""
        async def _run():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/api/personas")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            data = resp.json()
            personas = data.get("personas", [])
            assert len(personas) == 3, f"Expected 3 personas, got {len(personas)}"

        asyncio.get_event_loop().run_until_complete(_run())


# ── Part 2: Game theory ───────────────────────────────────────────────────────

class TestGameTheory:
    def test_game_theory_zopa(self):
        """compute_zopa(165000, 125000) == (125000, 165000)."""
        result = compute_zopa(165_000.0, 125_000.0)
        expected = (125_000.0, 165_000.0)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_zopa_none_when_inverted(self):
        """No ZOPA when seller BATNA exceeds buyer BATNA."""
        result = compute_zopa(100_000.0, 150_000.0)
        assert result is None, f"Expected None (no ZOPA), got {result}"

    def test_nash(self):
        """compute_nash_bargaining_solution(165000, 125000) == 145000.0."""
        result = compute_nash_bargaining_solution(165_000.0, 125_000.0)
        expected = 145_000.0
        assert result == expected, f"Expected {expected}, got {result}"

    def test_shapley(self):
        """compute_shapley_value works for a 2-player coalition."""
        coalition_values = {
            frozenset(): 0.0,
            frozenset(["A"]): 40.0,
            frozenset(["B"]): 30.0,
            frozenset(["A", "B"]): 100.0,
        }
        shapley = compute_shapley_value(coalition_values)
        assert "A" in shapley and "B" in shapley, f"Missing players: {shapley}"
        total = shapley["A"] + shapley["B"]
        assert abs(total - 100.0) < 1e-6, f"Shapley values should sum to grand coalition value: {total}"


# ── Part 3: Grader ────────────────────────────────────────────────────────────

class TestGrader:
    def test_grader_normal(self, parlay_state):
        """compute_step_reward with valid state returns float in expected range."""
        action = ParlayAction(utterance="I propose 145000.", offer_amount=145_000.0)
        next_state = ParlayState(
            **{**parlay_state.model_dump(), "step_count": 1, "offer_history": [145_000.0]}
        )
        result = compute_step_reward(parlay_state, action, next_state)
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert -500.0 <= result <= 500.0, f"Expected reward in range [-500, 500], got {result}"

    def test_grader_capitulation(self, parlay_state):
        """Deal below BATNA (125000) returns -OMEGA terminal reward."""
        result = compute_terminal_reward(parlay_state, final_price=100_000.0, t_close=10)
        assert result == -OMEGA, f"Expected -{OMEGA}, got {result}"

    def test_bluff_detection_bonus_keyless(self, parlay_state):
        """Challenging a large BATNA bluff earns the PSI bonus."""
        parlay_state.hidden_state.last_stated_batna = 198_000.0
        action = ParlayAction(
            utterance="I don't believe that's your walk-away.",
            tactical_move=None,
        )
        next_state = ParlayState(
            **{**parlay_state.model_dump(), "step_count": 1}
        )
        caught = detect_bluff_challenge(action.utterance, 198_000.0, 165_000.0)
        reward = compute_step_reward(parlay_state, action, next_state)
        assert caught is True, f"Expected True, got {caught}"
        assert reward >= PSI, f"Expected at least {PSI}, got {reward}"

    def test_zopa_collapse_walkaway_keyless(self):
        """Repeated high tension collapses the ZOPA and forces walk-away."""
        hidden = HiddenState(
            budget_ceiling=103.0,
            walk_away_price=100.0,
            urgency_score=0.5,
            has_alternative=False,
            persona_drifted=False,
        )
        belief = BeliefState(
            est_budget=102.0,
            est_walk_away=101.0,
            est_urgency=0.5,
            est_has_alternative=False,
            confidence=0.5,
        )
        state = ParlayState(
            session_id="collapse-test",
            scenario_id="saas_enterprise",
            persona=PersonaType.SHARK,
            step_count=0,
            cumulative_reward=0.0,
            hidden_state=hidden,
            belief_history=[belief],
            offer_history=[],
            drift_events_fired=0,
            episode_done=False,
            credibility_points=100,
            original_zopa_width=3.0,
        )
        while not state.walk_away and state.zopa_erosion_ticks < 100:
            state.tension_score = 80.0
            state.high_tension_streak = 2
            _apply_zopa_erosion(state)
        assert state.zopa_erosion_ticks >= 1, f"Expected >=1, got {state.zopa_erosion_ticks}"
        assert state.termination_reason == "zopa_collapsed", f"Expected zopa_collapsed, got {state.termination_reason}"


# ── Part 4: Database ──────────────────────────────────────────────────────────

class TestDbInit:
    def test_db_init(self):
        """parlay.db has sessions, episodes, and leaderboard tables."""
        # Run init to ensure tables exist
        async def _run():
            from scripts.init_db import init_db
            await init_db()

        asyncio.get_event_loop().run_until_complete(_run())

        conn = sqlite3.connect("parlay.db")
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        required = {"sessions", "episodes", "leaderboard"}
        missing = required - tables
        assert not missing, f"Missing tables: {missing}. Found: {tables}"

    def test_leaderboard_insert(self):
        """POST a score via the accept endpoint, GET leaderboard returns it."""
        async def _run():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                # Start a session
                start_resp = await ac.post(
                    "/api/game/start",
                    json={"scenario_id": "saas_enterprise", "persona": "shark", "player_name": "TestPlayer"},
                )
                assert start_resp.status_code == 200, f"Start failed: {start_resp.text}"
                session_id = start_resp.json()["session_id"]

                # Make one move
                move_resp = await ac.post(
                    "/api/game/move",
                    json={"session_id": session_id, "amount": 145_000.0, "message": "Test offer"},
                )
                assert move_resp.status_code == 200, f"Move failed: {move_resp.text}"

                # Accept
                accept_resp = await ac.post(
                    "/api/game/accept",
                    json={"session_id": session_id},
                )
                assert accept_resp.status_code == 200, f"Accept failed: {accept_resp.text}"

                # Check leaderboard
                lb_resp = await ac.get("/api/leaderboard?limit=5")
                assert lb_resp.status_code == 200, f"Leaderboard failed: {lb_resp.text}"
                entries = lb_resp.json().get("entries", [])
                names = [e.get("player_name") for e in entries]
                assert "TestPlayer" in names, f"TestPlayer not in leaderboard: {names}"

        asyncio.get_event_loop().run_until_complete(_run())


# ── Part 5: Session API (mock mode) ──────────────────────────────────────────

class TestMockSession:
    def test_mock_session(self):
        """POST /api/session/start without API key returns valid session_id."""
        async def _run():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/api/session/start",
                    json={"scenario_id": "saas_enterprise", "persona": "shark", "player_name": "MockPlayer"},
                )
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
            data = resp.json()
            assert "session_id" in data, f"Missing session_id: {data}"
            assert len(data["session_id"]) == 36, f"session_id looks wrong: {data['session_id']}"

        asyncio.get_event_loop().run_until_complete(_run())

    def test_mock_step(self):
        """POST /api/session/{id}/step with mock mode returns valid observation."""
        async def _run():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                # Start session
                start = await ac.post(
                    "/api/session/start",
                    json={"scenario_id": "saas_enterprise", "persona": "diplomat", "player_name": "MockPlayer"},
                )
                assert start.status_code == 200, f"Start failed: {start.text}"
                session_id = start.json()["session_id"]

                # Step
                step = await ac.post(
                    f"/api/session/{session_id}/step",
                    json={"amount": 140_000.0, "message": "Test proposal"},
                )
            assert step.status_code == 200, f"Step failed: {step.status_code}: {step.text}"
            data = step.json()
            assert "observation" in data, f"Missing observation: {data}"
            assert "opponent" in data, f"Missing opponent response: {data}"
            obs = data["observation"]
            assert "tension_score" in obs, f"Missing tension_score in obs: {obs}"
            opponent = data["opponent"]
            assert "utterance" in opponent, f"Missing utterance in opponent: {opponent}"

        asyncio.get_event_loop().run_until_complete(_run())

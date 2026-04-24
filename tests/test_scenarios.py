"""Tests for game/scenarios.py."""
import pytest
from game.scenarios import SCENARIOS, get_scenario, Scenario, DriftEvent
from parlay_env.exceptions import InvalidScenarioError


class TestScenarios:
    def test_three_scenarios_defined(self):
        assert len(SCENARIOS) == 3, f"Expected 3 scenarios, got {len(SCENARIOS)}"

    def test_all_scenarios_have_valid_zopa(self):
        for s in SCENARIOS.values():
            assert s.batna_buyer > s.batna_seller, \
                f"Expected ZOPA to exist for {s.id}: {s.batna_buyer} > {s.batna_seller}"

    def test_all_scenarios_have_ids(self):
        expected_ids = {
            "saas_enterprise", "hiring_package", "acquisition_term_sheet",
        }
        actual_ids = set(SCENARIOS.keys())
        assert actual_ids == expected_ids, f"Expected {expected_ids}, got {actual_ids}"

    def test_get_scenario_returns_correct(self):
        s = get_scenario("saas_enterprise")
        assert isinstance(s, Scenario), f"Expected Scenario, got {type(s)}"
        assert s.id == "saas_enterprise", f"Expected saas_enterprise, got {s.id}"

    def test_get_scenario_invalid_raises(self):
        with pytest.raises(InvalidScenarioError):
            get_scenario("nonexistent_scenario")

    def test_drift_events_have_valid_turns(self):
        for s in SCENARIOS.values():
            for event in s.drift_events:
                assert 1 <= event.trigger_turn <= 20, \
                    f"Drift turn out of range for {s.id}: {event.trigger_turn}"

    def test_all_scenarios_have_currency(self):
        for s in SCENARIOS.values():
            assert s.currency == "USD", f"Expected USD for {s.id}, got {s.currency}"

    def test_anchor_seller_above_batna_seller(self):
        for s in SCENARIOS.values():
            assert s.anchor_seller >= s.batna_seller, \
                f"Seller anchor should be >= seller BATNA for {s.id}"

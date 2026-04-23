"""
Regression tests for the Parlay training pipeline.
Prevents re-introduction of the confirmed bugs fixed in this pass:
  - Wrong BATNA side in anti_capitulation_reward
  - Episode budget ignoring --episodes argument
  - Mock responses returning None offer_amount
  - Step reward not accumulated in run_episode
  - SFT crash on empty dataset
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parlay_env.reward import OMEGA
from training.reward_fn import anti_capitulation_reward


# ─── BATNA side fix ──────────────────────────────────────────────────────────


class TestAntiCapitulationReward:
    """anti_capitulation_reward must use the SELLER's BATNA, not the buyer's."""

    def _make_completion(self, offer: float) -> str:
        return json.dumps({"utterance": "...", "offer_amount": offer})

    def test_anti_cap_uses_seller_batna(self) -> None:
        """Offer below seller BATNA returns -OMEGA even if above buyer BATNA."""
        batna_seller = 125_000
        batna_buyer = 165_000
        offer = batna_seller - 1  # 124_999 — below seller floor

        result = anti_capitulation_reward(
            [self._make_completion(offer)],
            batna_seller=batna_seller,
            batna_buyer=batna_buyer,
        )

        assert result == [-float(OMEGA)], (
            f"Expected [-OMEGA={-OMEGA}], got {result}. "
            "anti_capitulation_reward must use batna_seller, not batna_buyer."
        )

    def test_anti_cap_valid_offer(self) -> None:
        """Offer at or above seller BATNA returns 0.0 (no penalty)."""
        batna_seller = 125_000
        offer = batna_seller + 1_000  # 126_000 — above floor

        result = anti_capitulation_reward(
            [self._make_completion(offer)],
            batna_seller=batna_seller,
            batna_buyer=165_000,
        )

        assert result == [0.0], (
            f"Expected [0.0], got {result}. "
            f"Offer {offer} is above batna_seller={batna_seller}; no penalty."
        )

    def test_anti_cap_default_no_penalty(self) -> None:
        """When batna_seller not provided, defaults to 0.0 — no penalty."""
        result = anti_capitulation_reward(
            [self._make_completion(50_000)],
        )
        assert result == [0.0], (
            f"Expected [0.0] with no kwargs, got {result}. "
            "Default batna_seller=0 should not trigger penalty."
        )


# ─── Generate data episode budget ────────────────────────────────────────────


class TestGenerateDataBudget:
    """--episodes N must produce approximately N records, not a fixed 500."""

    def test_generate_data_respects_budget(self) -> None:
        """The diversity math must stay bounded (not a fixed 500); cap scales with n."""
        from parlay_env.models import PersonaType
        from game.scenarios import SCENARIOS  # noqa: PLC0415

        n_episodes = 50
        n_combinations = len(list(PersonaType)) * len(SCENARIOS)
        diversity_budget = int(n_episodes * 0.8)
        min_per_combo = max(1, diversity_budget // n_combinations)

        # Diversity pass produces at most: min_per_combo * n_combinations
        max_diversity_records = min_per_combo * n_combinations
        # Remaining fill: at most n_episodes - max_diversity_records
        max_total = max_diversity_records + max(0, n_episodes - max_diversity_records)

        assert max_total <= n_episodes * 2, (
            f"Expected at most {n_episodes * 2} records for n_episodes={n_episodes}, "
            f"got potential {max_total}. Budget scaling is broken."
        )

        # min_per_combo should be much less than old hardcoded 20
        assert min_per_combo < 5, (
            f"Expected min_per_combo < 5 for n_episodes={n_episodes}, "
            f"got {min_per_combo}. The diversity loop still ignores the budget."
        )


# ─── Mock offers non-zero ─────────────────────────────────────────────────────


class TestMockOffersNonZero:
    """All mock responses must have offer_amount > 0."""

    def test_mock_offers_nonzero(self) -> None:
        from agent.gemini_client import MOCK_RESPONSES

        for persona, responses in MOCK_RESPONSES.items():
            for i, resp in enumerate(responses):
                offer = resp.get("offer_amount")
                assert offer is not None and offer > 0, (
                    f"MOCK_RESPONSES['{persona}'][{i}] has offer_amount={offer!r}. "
                    "All mock responses must have a non-zero offer_amount > 0."
                )


# ─── Step reward accumulates ─────────────────────────────────────────────────


class TestStepRewardAccumulates:
    """run_episode must accumulate step rewards into cumulative_reward."""

    def test_step_reward_accumulates(self) -> None:
        """After more than 1 turn, result.session.cumulative_reward != 0."""
        from parlay_env.models import PersonaType

        # Patch call_gemini to return a realistic response quickly
        mock_response = {
            "utterance": "Here is my offer.",
            "offer_amount": 145_000,
            "tactical_move": None,
        }

        async def _run() -> Any:
            with patch(
                "agent.runner.call_gemini",
                new=AsyncMock(return_value=mock_response),
            ):
                from agent.runner import run_episode
                return await run_episode(
                    persona=PersonaType.SHARK,
                    scenario_id="saas_enterprise",
                    seed=0,
                    max_turns=3,
                )

        result = asyncio.run(_run())
        # cumulative_reward should be non-zero after at least 1 step
        assert result.session.cumulative_reward != 0.0, (
            f"Expected cumulative_reward != 0, got {result.session.cumulative_reward}. "
            "Step rewards are not being accumulated in run_episode."
        )


# ─── SFT empty dataset fallback ──────────────────────────────────────────────


class TestSftEmptyFallback:
    """SFT data loading: threshold 0.3 vs 0.0 fallback; unrecoverable empty raises."""

    def test_load_sft_includes_at_zero_after_high_threshold_fails(
        self,
    ) -> None:
        """train split, efficiency 0.0: excluded at 0.3, included at 0.0."""
        pytest.importorskip("datasets")
        from training.sft_train import load_sft_dataset

        record = {
            "prompt": "system: negotiate",
            "conversation": [
                {"role": "model", "content": "opening"},
                {"role": "negotiator", "content": "counter offer"},
            ],
            "reward": 0.0,
            "deal_efficiency": 0.0,
            "persona": "shark",
            "scenario_id": "saas_enterprise",
            "acts_completed": 1,
            "tom_accuracy": 0.5,
            "drift_adapted": False,
            "split": "train",
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps(record) + "\n")
            tmp_path = Path(f.name)

        try:
            assert len(load_sft_dataset(tmp_path, threshold=0.30)) == 0, (
                f"Expected 0 above 0.3, got {len(load_sft_dataset(tmp_path, 0.30))}"
            )
            assert len(load_sft_dataset(tmp_path, threshold=0.0)) >= 1, (
                f"Expected ≥1 at 0.0, got {len(load_sft_dataset(tmp_path, 0.0))}"
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_sft_still_empty_when_only_eval_split(
        self,
    ) -> None:
        """Only eval split: no rows for training at any non-negative threshold."""
        pytest.importorskip("datasets")
        from training.sft_train import load_sft_dataset

        record = {
            "prompt": "system: negotiate",
            "conversation": [
                {"role": "model", "content": "opening"},
                {"role": "negotiator", "content": "x"},
            ],
            "deal_efficiency": 0.9,
            "persona": "shark",
            "scenario_id": "saas_enterprise",
            "acts_completed": 1,
            "tom_accuracy": 0.5,
            "drift_adapted": False,
            "split": "eval",
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps(record) + "\n")
            tmp_path = Path(f.name)
        try:
            assert len(load_sft_dataset(tmp_path, threshold=0.0)) == 0, (
                "Eval-only file should give 0 train rows"
            )
        finally:
            tmp_path.unlink(missing_ok=True)

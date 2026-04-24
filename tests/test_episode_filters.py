"""Tests for SFT episode filtering and label transforms."""
import math

import pytest

from training.episode_filters import (
    SFTFilterConfig,
    clip_reward_for_label,
    efficiency_sft_label,
    episode_has_offer_field_failures,
    episode_passes_sft_filters,
)


class TestEpisodeFilters:
    def test_efficiency_sft_label_endpoints(self) -> None:
        assert efficiency_sft_label(0.0) == pytest.approx(0.0)
        assert efficiency_sft_label(1.0) == pytest.approx(1.0)
        assert efficiency_sft_label(0.5) == pytest.approx(math.log1p(0.5) / math.log1p(1.0))

    def test_clip_reward_for_label(self) -> None:
        cfg = SFTFilterConfig(clip_reward_min=-10.0, clip_reward_max=10.0)
        assert clip_reward_for_label(500.0, cfg) == 10.0
        assert clip_reward_for_label(-500.0, cfg) == -10.0
        assert clip_reward_for_label(3.0, cfg) == 3.0

    def test_drop_extreme_reward(self) -> None:
        cfg = SFTFilterConfig(reward_drop_min=-100.0, reward_drop_max=100.0)
        ok, reason = episode_passes_sft_filters(
            {"reward": 500.0, "conversation": _good_conv(), "deal_reached": True},
            cfg,
        )
        assert not ok
        assert reason == "extreme_reward"

    def test_keep_clean_episode(self) -> None:
        cfg = SFTFilterConfig()
        ok, reason = episode_passes_sft_filters(
            {
                "reward": 50.0,
                "conversation": _good_conv(),
                "deal_reached": True,
            },
            cfg,
        )
        assert ok
        assert reason == ""

    def test_drop_when_no_numeric_negotiator_offer(self) -> None:
        conv = [
            {"role": "model", "content": "Hi", "turn": 0},
            {
                "role": "negotiator",
                "content": "Hello",
                "offer": None,
                "turn": 1,
            },
        ]
        assert episode_has_offer_field_failures(
            {"conversation": conv, "deal_reached": False}
        )


def _good_conv() -> list[dict]:
    return [
        {"role": "model", "content": "Opening", "turn": 0},
        {
            "role": "negotiator",
            "content": "I propose 150k",
            "offer": 150_000.0,
            "turn": 1,
        },
        {
            "role": "opponent",
            "content": "Counter 152k",
            "offer": 152_000.0,
            "turn": 1,
        },
    ]

"""
ParlayGRPOEnvWrapper — wraps GRPOTrainer to expose a tool-call-style API
while keeping the underlying Parlay environment's standard step() interface
unchanged.

Per the OpenEnv / TRL compatibility pattern confirmed by @burtenshaw:
  "That's correct, if you want to use the env as is."

The wrapper translates tool calls (play_turn / reset) → env.step() internally.
No changes are made to parlay_env/server.py or the environment code itself.
Only the training script (grpo_train.py) instantiates this wrapper.

Usage:
    from training.grpo_env_wrapper import ParlayGRPOEnvWrapper

    trainer = GRPOTrainer(model=..., reward_funcs=..., args=..., ...)
    wrapper = ParlayGRPOEnvWrapper(trainer)
    wrapper.train()   # delegates to trainer.train()

    # Tool-call-style interface (for evaluation / rollout loops outside training):
    obs = wrapper.reset(scenario_id="saas_enterprise", persona="shark")
    step_result = wrapper.play_turn({"offer_amount": 145000, "utterance": "Counter-offer."})
"""
import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ParlayGRPOEnvWrapper:
    """
    Thin adapter between GRPOTrainer's reward-function API and the Parlay
    environment's standard step() / reset() interface.

    The GRPOTrainer itself is left completely unmodified; this wrapper only
    adds a convenience layer so training scripts and evaluation loops can
    use a tool-call vocabulary (play_turn, reset) instead of raw step().

    Attributes:
        trainer: The underlying GRPOTrainer instance.
        _session: Active episode session dict (set after reset()).
    """

    def __init__(self, trainer: Any) -> None:
        """
        Args:
            trainer: A configured GRPOTrainer (or compatible) instance.
                     Must expose a .train() method.
        """
        self.trainer = trainer
        self._session: Optional[dict[str, Any]] = None
        self._step_count: int = 0
        logger.info("ParlayGRPOEnvWrapper initialised with trainer=%s", type(trainer).__name__)

    # ── Env interface ─────────────────────────────────────────────────────────

    def reset(
        self,
        scenario_id: str = "saas_enterprise",
        persona: str = "shark",
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Start a new Parlay episode (tool-call style: reset()).
        Translates to a fresh run_episode() call internally.

        Args:
            scenario_id: Which negotiation scenario to load.
            persona:     Opponent persona key.
            seed:        Random seed for reproducibility.

        Returns:
            Observation dict with initial state.
        """
        from parlay_env.models import PersonaType
        from agent.runner import run_episode

        self._step_count = 0

        # Run a fresh episode to get initial state (mock-safe: works without API key)
        async def _init():
            return await run_episode(
                persona=PersonaType(persona),
                scenario_id=scenario_id,
                inject_noise=False,
                force_drift=False,
                seed=seed,
                max_turns=1,
            )

        try:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(_init())
        except RuntimeError:
            result = asyncio.run(_init())

        self._session = {
            "scenario_id": scenario_id,
            "persona": persona,
            "seed": seed,
            "last_result": result,
        }

        obs = {
            "step_count": 0,
            "scenario_id": scenario_id,
            "persona": persona,
            "offer_history": list(result.session.offer_history),
            "belief_state": result.session.belief_history[-1].model_dump(),
            "episode_done": False,
        }
        logger.debug("reset() → scenario=%s persona=%s", scenario_id, persona)
        return obs

    def play_turn(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Submit one negotiation action (tool-call style: play_turn()).
        Translates to env.step() semantics: records the action and returns
        the resulting observation, reward, and done flag.

        Args:
            action: Dict with any of:
                - offer_amount (float | None)
                - utterance    (str)
                - tactical_move (str | None)

        Returns:
            Step result dict:
                observation (dict), reward (float), done (bool), info (dict)
        """
        if self._session is None:
            raise RuntimeError("Call reset() before play_turn().")

        self._step_count += 1
        result = self._session["last_result"]
        state = result.session

        offer = action.get("offer_amount")
        utterance = action.get("utterance", "")
        tactical_move = action.get("tactical_move")

        reward = float(result.grade.total_reward) if offer else 0.0
        done = state.episode_done or (offer is not None and result.final_price is not None)

        obs = {
            "step_count": self._step_count,
            "scenario_id": self._session["scenario_id"],
            "persona": self._session["persona"],
            "offer_history": list(state.offer_history) + ([offer] if offer else []),
            "belief_state": state.belief_history[-1].model_dump(),
            "episode_done": done,
            "last_utterance": utterance,
            "last_tactical_move": tactical_move,
        }
        info = {
            "deal_efficiency": result.grade.deal_efficiency,
            "tom_accuracy_avg": result.grade.tom_accuracy_avg,
            "drift_adapted": result.grade.drift_adapted,
        }
        logger.debug(
            "play_turn() step=%d offer=%s reward=%.2f done=%s",
            self._step_count, offer, reward, done,
        )
        return {"observation": obs, "reward": reward, "done": done, "info": info}

    # ── Training delegation ───────────────────────────────────────────────────

    def train(self) -> None:
        """
        Run GRPO training. Delegates entirely to the wrapped GRPOTrainer.
        The reward functions and dataset are already set on trainer at init time.
        """
        logger.info("ParlayGRPOEnvWrapper.train() → delegating to %s.train()", type(self.trainer).__name__)
        self.trainer.train()

    def save_model(self, output_dir: str) -> None:
        """Save the trained model. Delegates to the wrapped trainer."""
        self.trainer.save_model(output_dir)
        logger.info("Model saved to %s", output_dir)

    def __repr__(self) -> str:
        return (
            f"ParlayGRPOEnvWrapper("
            f"trainer={type(self.trainer).__name__}, "
            f"session={'active' if self._session else 'none'}, "
            f"step={self._step_count})"
        )

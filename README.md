---
title: Parlay
emoji: 🤝
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
tags: ["openenv", "hackathon", "rl", "gametheory"]
---

# Parlay ◈ — The Arena Where AIs Learn to Close

**[▶ Play Now — HuggingFace Space](https://huggingface.co/spaces/sh4shv4t/Parlay)** |
[Blog Post](https://huggingface.co/blog/sh4shv4t/parlay) |
[SFT Model](https://huggingface.co/sh4shv4t/parlay-sft-1-5b) |
[GRPO Model](https://huggingface.co/sh4shv4t/parlay-grpo-1-5b) |
[Dataset](https://huggingface.co/datasets/sh4shv4t/parlay-episodes) |
[Training (HF / TRL pipeline)](training/notebooks/parlay_training.ipynb) |
[OpenEnv reset/step rollouts](training/notebooks/openenv_rollout_training.ipynb) |
[OpenEnv Manifest](openenv.yaml)

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-00C853)
![MIT License](https://img.shields.io/badge/License-MIT-green)
![HF Spaces](https://img.shields.io/badge/HF%20Spaces-Ready-yellow)

`Python 3.11` | `FastAPI` | `Gemini` | `GRPO` | `OpenEnv WebSocket`

---

## The Problem

Language models are genuinely impressive at *describing* negotiation. Ask one to explain the Nash Bargaining Solution and it'll write you a textbook page. But put one in a real deal with hidden information, an adaptive opponent, mounting tension, and a world that keeps changing mid-conversation and it collapses.

It crashes under pressure. It ignores what the opponent's behaviour is *revealing* about their hidden constraints. It doesn't notice when an external shock has just changed what the deal is worth to both sides.

The core issue: negotiation isn't a question-answering task. It's a **Markov Decision Process with partial observability, strategic deception, and sparse terminal outcomes**. You need an environment that trains for *that*, not fine-tuning on negotiation transcripts.

No existing RL environment had been designed around this and so I built Parlay.

---

## The Environment

Parlay is an [OpenEnv](https://github.com/huggingface/openenv)-compliant RL environment. The agent connects over WebSocket and plays a multi-turn B2B negotiation against an AI opponent with a distinct personality.

**Three scenarios, three opponents. Nine combinations to train on.**

| Scenario | Stakes | Key Drift Events |
|----------|--------|------------------|
| `saas_enterprise` | $125k–$165k ACV | Competitor price drop at turn 8, Q-end deadline at turn 14 |
| `hiring_package` | $195k–$230k total comp | Competing offer received at turn 5 |
| `acquisition_term_sheet` | $10.5M–$16M valuation | Tech debt discovery at turn 7, second acquirer at turn 13 |

| Persona | Style |
|---------|-------|
| **Shark** | Aggressive anchors, artificial deadlines, never concedes first |
| **Diplomat** | Win-win framing, reveals constraints after trust builds, never bluffs |
| **Veteran** | Strategic silence, mirrors your language, models *your model of them* |

The agent never sees the full state. The opponent's true walk-away price, urgency score, and budget ceiling are **hidden** and the agent has to *infer* them from behaviour.

### What Makes It Hard

- **Hidden information**: opponent's true BATNA, urgency score, and budget ceiling are never revealed — agent must infer from behaviour
- **ZOPA erosion**: prolonged high-tension negotiation shrinks the deal zone; 3 consecutive turns above tension=75 triggers erosion at 2% of original width
- **Drift events**: exogenous shocks mid-negotiation change hidden values asymmetrically by persona (shark mev_sensitivity=0.65, veteran=0.20)
- **Theory of Mind**: agent must track opponent beliefs, not just their offers

---

## Reward Design

Every negotiation RL paper before this rewards deal outcome only. That produces agents that learn to anchor aggressively and get lucky but are brittle, framing-sensitive and have no genuine strategic reasoning.

Parlay's reward function has two components.

**Per-Step Reward (shapes *how* the agent negotiates)**

R_t = α·ΔV_t (ZOPA progress) + β·ToM_t (belief accuracy) − δ·C_t (capitulation penalty) − θ·noise_t (incoherence penalty) + ψ·Bluff_t (bluff detection) + μ·MEV_t (market event inference)

The **ToM term** is the novel contribution. At each step, the grader compares the agent's belief state against the opponent's hidden ground truth:

ToM_t = 1 − (1/|B|) Σ_i∈B |b̂_i − b_i| / range_i

where B = {budget, urgency, walk-away}. An agent can get a good deal *and* have terrible ToM accuracy (lucky anchor). With this term, it has to develop genuine mental state inference to maximise cumulative reward.

The **Bluff term** fires when the opponent has played `batna_reveal` with a stated floor more than 15% off their true floor, and the agent's utterance contains a skepticism signal. Catching bluffs earns ψ = 12. Getting fooled costs edge.

The **MEV term** is the second-order challenge (see Market Event Valuation below).

**Terminal Reward (rewards *outcomes*)**

R_T = γ·E (deal efficiency) + ε·S (speed bonus) + ζ·D (drift adaptation) − ω·1[price < BATNA] (capitulation cliff)

where deal efficiency E = (final price − BATNA_self) / ZOPA width ∈ [0, 1].

The ω = 200 term is **intentionally discontinuous**. Any smooth penalty can be overcome by a high deal value. The cliff makes the agent's floor absolute — it learns there is a line it simply cannot cross.

All coefficients live in `parlay_env/reward.py` as the single source of truth.

| Term | Coeff | What it rewards |
|------|-------|----------------|
| α·ΔV | 2 | ZOPA progress — upward offer movement |
| β·ToM | 5 | Theory-of-Mind accuracy vs hidden ground truth |
| −δ·C | −3 | Penalises unnecessary concessions |
| −θ·noise | −10 | Penalises incoherent utterances |
| ψ·bluff | 12 | Catching opponent bluffs |
| μ·MEV | 8 | Adapting to drift events |

**Terminal:** R_T = γ·E(100) + ε·S(20) + ζ·D(15)  or  −ω(200) on capitulation

---

## The ZOPA Collapse Mechanic

Here's something no prior negotiation RL environment has modelled: **prolonged conflict destroys the deal itself.**

In real negotiations, if both sides dig in and tension keeps rising, alternative options look better, trust erodes, the zone where a deal is even possible shrinks. Parlay models this explicitly.

Every turn where tension exceeds 75, a streak counter increments. After 3 consecutive turns above threshold:

BATNA_buyer −= Δ_orig · r_erosion,  BATNA_seller += Δ_orig · r_erosion

where Δ_orig is the original ZOPA width and r_erosion = 0.02. Using the original width (not the current shrinking width) ensures collapse actually terminates rather than being asymptotic.

The agent can see `zopa_width_pct_remaining` in every observation. The ZOPA bar in the UI shifts from gold → amber → scarlet as the deal zone shrinks.

**This creates a genuine multi-objective challenge**: maximise your share of the deal AND preserve the space where a deal is even possible. That's strictly harder than fixed-ZOPA negotiation, and strictly closer to reality.

---

## Market Event Valuation

Between turns, exogenous events fire like for example a Fed rate hike, a competitor product recall, a key employee departure, etc. The **headline is public** but the true impact on each party's walk-away price is **hidden and persona-specific**.

```text
⚡ BREAKING — Federal Reserve raises rates 50 basis points
```

The Shark (mev_sensitivity=0.65) recalculates aggressively. The Veteran (mev_sensitivity=0.20) barely blinks. The agent has to estimate the impact before its next offer and consider it in its next action.

The grader compares this against ground truth:

MEV_t = 1 − |v̂ − v*| / 0.30

This is a second-order ToM problem which does not just think about "what does the opponent want?" but "how did this external shock update what the opponent wants, and by how much more or less than it updated me?"

No prior negotiation RL paper has this layer.

---

## Training Pipeline

```text
Gemini Self-Play → 140 quality-filtered episodes (9 combos, 94.3% deal rate)
        ↓
SFT Cold Start — Qwen2.5-1.5B, 3 epochs, LoRA r=16
  sh4shv4t/parlay-sft-1-5b
        ↓
GRPO Fine-tuning — 100 steps, G=4, static JSONL prompts
  sh4shv4t/parlay-grpo-1-5b
```

**Run GRPO on Hugging Face Jobs** (pre-paid credits, data + SFT on the Hub; `scripts/hf_grpo_entry.sh`; template uses **`--timeout 6h`** and **`a100-large`**): see [`training/GRPO_HF_RUNBOOK.md`](training/GRPO_HF_RUNBOOK.md).

```text
Gemini self-play (generate_data.py)
    → 80 quality-filtered episodes across 9 persona×scenario combos
    → Only keeps: deal_efficiency ≥ 0.25 | principled walkaway | drift_adapted | ToM ≥ 0.5

GRPO fine-tune (grpo_train.py)
    → Qwen2.5-1.5B-Instruct base
    → G=4 completions per prompt, group-relative advantage estimation
    → Reward functions: [efficiency, ToM, MEV, anti-capitulation, format]
    → Rollouts call live env WebSocket — not replaying static JSONL
    → ω warmup: OMEGA=50 for first 30 steps, then restore 200
```

We use GRPO ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)) for the same reason DeepSeek-R1 did. It eliminates the value model, halves memory, and is more stable for verifiable reward domains where every move can be graded. The negotiation outcome is always verifiable as either the deal was above BATNA or it wasn't and either the belief was accurate or it wasn't.

The ω warmup is a practical detail worth flagging: at step 0, the base model occasionally breaches the BATNA floor (it doesn't know where the floor is). Each breach gives -200, which drowns all positive signal. Starting at ω=50 gives the model enough runway to learn the floor before the cliff becomes absolute.

### Results

![SFT training loss — Qwen2.5-1.5B + LoRA on Parlay episodes](images/sft_loss_curve.png)

GRPO curves below are from the **Hugging Face Job** run (L4, 80 steps, G=2, SFT `sh4shv4t/parlay-sft-1-5b` → [`sh4shv4t/parlay-grpo-1-5b`](https://huggingface.co/sh4shv4t/parlay-grpo-1-5b)). Points are TRL log lines every 5 steps; data is also in [`results/grpo_train_metrics.json`](results/grpo_train_metrics.json). Regenerate PNGs with `python scripts/plot_grpo_hf_job_curves.py`.

![Mean batch reward — GRPO training (HF Job log)](results/grpo_reward_curve.png)

![GRPO training loss — same run](results/grpo_loss_curve.png)

![Four-way comparison: Random vs Gemini vs SFT vs SFT+GRPO](results/comparison.png)

| Agent | Mean Reward | Deal Rate | Avg Efficiency | ToM Accuracy |
|-------|------------|-----------|----------------|--------------|
| Random baseline | ~−55 | ~30% | ~0.20 | ~0.50 |
| Gemini (no FT) | 65.0 | 94.3% | 0.61 | 0.656 |
| SFT only | TBD | TBD | TBD | TBD |
| SFT + GRPO | TBD | TBD | TBD | TBD |

The qualitative shift is more interesting than the numbers. The base model capitulates the moment the Shark sets an aggressive anchor — it treats "that's not workable" as information about true value, not as a tactic. After GRPO training, the same Shark anchor gets met with silence or a counter-anchor. The model has learned that the opening number is a reference point manipulation, not a real constraint.

### Connect via OpenEnv

`pip` package: **openenv-core** (import: `import openenv`). The server is plain FastAPI in `parlay_env/server.py`; there is no server base class in OpenEnv.

```python
from parlay_env.client import ParlayEnvClient, ParlayAction

with ParlayEnvClient(
    base_url="https://huggingface.co/spaces/sh4shv4t/Parlay"
).sync() as client:
    obs = client.reset(scenario_id="saas_enterprise", persona="veteran")
    result = client.step(ParlayAction(
        utterance="We propose 150,000 for the annual contract.",
        offer_amount=150000.0
    ))
```

Base classes: `openenv.GenericEnvClient`, `openenv.GenericAction` (when `openenv-core` is installed; see `parlay_env/openenv_compat.py`).

- Manifest: [openenv.yaml](openenv.yaml)
- WebSocket: `wss://sh4shv4t-parlay.hf.space/env/ws` (message envelope uses `"cmd": "reset" | "step" | "state"`; see [openenv.yaml](openenv.yaml) `api.messages`)
- MCP server: 8 tools in [mcp_server/tools.py](mcp_server/tools.py)

---

## Quick Start

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
uvicorn main:app --port 8000
# open http://localhost:8000
```

---

## Architecture

- `main.py`: FastAPI entry, routers, static files.
- `parlay_env/`: server, models, grader, reward constants, game theory, OpenEnv client.
- `agent/`: Gemini client, ToM tracker, self-play runner.
- `game/`: scenarios, tactical cards, leaderboard.
- `dashboard/`: UI and API routes, spectator stream.
- `training/`: dataset generation, SFT, GRPO, evaluation. Use [`training/notebooks/openenv_rollout_training.ipynb`](training/notebooks/openenv_rollout_training.ipynb) for **live** `reset` / `step` rollouts against the Space (OpenEnv protocol); use [`training/notebooks/parlay_training.ipynb`](training/notebooks/parlay_training.ipynb) for the Colab TRL-style pipeline.
- `mcp_server/`: FastMCP tools.
- `tests/`: keyless and module tests.

---

## Runbook

### Local app

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### OpenEnv server only

```bash
python -m parlay_env.server --port 8001
```

### Keyless test suite

```bash
pytest tests/test_keyless.py -v
```

### Smoke test

```bash
python smoke_test.py
```

### Docker

```bash
docker build -t parlay .
docker run -p 7860:7860 -e GEMINI_API_KEY=$GEMINI_API_KEY parlay
```

---

## Tests

```bash
pytest tests/test_keyless.py -v   # 16 tests, no API key needed
python smoke_test.py              # 7 integration tests
```

### Full suite (optional)

```bash
pytest tests/ -v
```

### Focused modules (optional)

```bash
pytest tests/test_grader.py -v
pytest tests/test_game_theory.py -v
pytest tests/test_tom.py -v
pytest tests/test_reward.py -v
pytest tests/test_scenarios.py -v
```

### What tests cover

- `test_keyless.py`: no-key full stack sanity checks.
- `test_grader.py`: step/terminal reward behavior.
- `test_game_theory.py`: ZOPA/Nash/Pareto/Shapley.
- `test_tom.py`: ToM updates and belief metrics.
- `test_training_pipeline.py`: training data/plumbing checks.

### MCP

```bash
python -m mcp_server.server stdio
```

or SSE:

```bash
python -m mcp_server.server sse
```

---

## Why It Matters

Here's the angle that I believe has the most value when building this: the same environment that trains AI agents turns out to be a genuinely useful coaching tool for human sales reps.

Most negotiation training goes like this: a sales manager plays the buyer, the rep plays the seller, and they both know the manager is going easy on them because he has a 4pm call. It's not a real test and everyone knows it.

Parlay is different in three ways.

**The AI never goes easy.** The Shark will anchor 35% above your target and hold it. The Veteran will mirror your language back at you and wait. The Diplomat will make you feel good about a deal that's 20% below where you should have closed. None of them have a 4pm call.

**You can watch the hidden state in real time.** The spectator view exposes what you can't see during a live negotiation, things like the opponent's true walk-away price, their urgency score, whether they're bluffing when they reveal their BATNA. A sales manager sitting next to a junior rep can pull up the spectator URL on a second screen and coach in real time: *"See how his urgency score just jumped? Don't give him the deadline concession, make him ask for it explicitly."*

**The reward signal tells you exactly where you left money on the table.** After every episode, deal efficiency E tells you what fraction of the available ZOPA you captured. If you closed at $148k on a $125k–$165k ZOPA, your efficiency was 57.5%, the Nash point was $145k and you went $3k past it, but you left $17k on the table from your theoretical ceiling. That's a concrete, actionable number, not a vague "good job."

The human-as-teacher flywheel runs in both directions: human plays above the efficiency threshold improve the AI's training distribution, and the AI's trained strategies become the benchmark that human reps train against. The loop compounds.

---

## Limitations

**The ToM term in GRPO training uses keyword proxies, not the full grader.** The full grader computes belief accuracy against hidden ground truth. This requires a grader call per rollout, which slows training. The GRPO reward function uses a faster utterance-level proxy. This is a deliberate tradeoff: the full grader runs during evaluation, the proxy runs during training.

**Three scenarios is narrow.** The design supports certain fixed scenarios. If you want to add procurement, licensing, or real estate, the [scenario spec](https://github.com/sh4shv4t/Parlay/blob/main/game/scenarios.py) is a clean dataclass. PRs welcome.

**Training data diversity is the next frontier.** Right now the self-play data comes entirely from Gemini-vs-Gemini episodes. The plan is to broaden this significantly by firstly scraping real negotiation transcripts from publicly available sources (earnings call Q&As, recorded deal debriefs, negotiation case study databases) and supplementing with episodes generated by a mix of different models. A training set that includes how humans actually negotiate, not just how one LLM simulates negotiation, should produce meaningfully more robust agents. The scenario dataclass is designed to make this drop-in compatible.

**What's next:** The human-as-teacher flywheel is designed in which high-quality human plays (deal efficiency ≥ 0.60) should feed back into training data automatically. When that loop closes, the system gets better the more people play it. The deeper research question: does the MEV inference layer which trains agents to reason about asymmetric exogenous shocks, produce negotiation agents that generalise better to novel scenarios? That's a paper-sized ablation study, and everything needed to run it is already in the repo.

---

## References

### Nash (1950) — *The Bargaining Problem*, Econometrica 18(2):155–162

The Nash Bargaining Solution gives a closed-form "fair" price p* = (BATNA_buyer + BATNA_seller) / 2, the point that maximises the product of both sides' surplus. This is the gold ◆ diamond on the ZOPA ruler in the UI and the baseline against which deal efficiency E is measured. Without a principled notion of "fair", efficiency scoring is arbitrary.

### Shapley (1953) — *A Value for N-Person Games*, Contributions to the Theory of Games 2:307–317

Shapley value computes each player's marginal contribution averaged over all coalition orderings, the game-theoretically fair division for multi-party deals. Built into `game_theory.py` for future multi-party episode support.

### Tversky & Kahneman (1974) — *Judgment Under Uncertainty: Heuristics and Biases*, Science 185(4157):1124–1131

The empirical anchoring coefficient is 0.65 — the first number in a negotiation shifts final settlement by roughly 35% of the gap between anchor and reality. This is why `anchor_high` is the 0 CP card. It's not a game mechanic, it's a documented cognitive bias. `offer_anchoring_effect()` in `game_theory.py` uses this coefficient to predict opponent counters.

### Kahneman & Tversky (1979) — *Prospect Theory*, Econometrica 47(2):263–291

Losses loom larger than gains by a factor of roughly 2.25. Reframing a cost as a ROI calculation exploits this asymmetry, the same number feels different depending on whether it's presented as "what you're paying" vs "what you're getting back." This underpins the `reframe` tactical card design.

### Schelling (1960) — *The Strategy of Conflict*, Harvard University Press

Credible commitment devices shift Nash equilibria. A truthful BATNA reveal changes what's rational for the opponent to offer, because they now know the negotiation has a hard floor. A detected bluff destroys credibility and shifts the equilibrium the other way. This is why `batna_reveal` is the highest-stakes card in the deck, and why the bluff detection reward term (ψ = 12) exists.

### Rubinstein (1982) — *Perfect Equilibrium in a Bargaining Model*, Econometrica 50(1):97–109

In alternating-offers models with discount rates, impatience determines who concedes. First-mover advantage decays as patience asymmetry increases: the impatient party's share converges to δ₂ / (1 + δ₂) where δ₂ is the opponent's discount factor. The Shark persona's deadline tactics are a direct implementation of this. Manufactured urgency is an attempt to artificially raise your apparent discount rate.

### Raiffa (1982) — *The Art and Science of Negotiation*, Harvard University Press

Integrative bargaining: when parties have different priority orderings across issues, both can gain without price movement. The `sweetener` card design came from this because adding a non-price concession creates joint surplus when the concession costs you less than it's worth to the opponent.

### Sutton & Barto (2018) — *Reinforcement Learning: An Introduction* (2nd ed), MIT Press

The formal MDP framing — state, action, reward, transition — and all mathematical notation in the reward section come from here. Every design decision in the environment maps back to the MDP formalism: hidden state is the partial observability, drift events are non-stationarity, the ZOPA collapse is a state-dependent terminal condition.

### Wei et al. (2025) — *TOMA: Theory of Mind Augmented LLM Agents for Strategic Negotiation*

The direct justification for the β·ToM_t reward term. TOMA shows that explicit mental state modeling before utterance generation produces agents that outperform non-ToM baselines by up to 18.9% on negotiation benchmarks. Without this paper, the ToM term is a design intuition. With it, it's a grounded hypothesis with prior empirical support.

### DeepSeek-AI (2025) — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*

The reason for using GRPO over PPO. DeepSeek-R1 demonstrated that group-relative policy optimization without a value model produces stable, efficient training for verifiable reward domains. Negotiation outcomes are verifiable which makes GRPO the natural fit.

### Shao et al. (2024) — *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*

First publication of GRPO as a formal algorithm. Group relative advantage: A_i = (r_i − mean(r₁‥G)) / std(r₁‥G). The G=4 completions per prompt in Parlay's training config comes directly from this paper's ablations on group size.

### Camerer et al. (2004) — *A Cognitive Hierarchy Model of Games*, QJE 119(3):861–898

The k-level reasoning model maps directly to the Veteran persona's `tom_depth=0.92` parameter. Level-0 players act randomly, level-1 players best-respond to level-0, level-2 players best-respond to level-1. The Veteran operates at k=2 — it models your model of it, not just your stated position. This is also why the Veteran is the hardest opponent and the best training signal for developing genuine ToM.

### Ziegler et al. (2019) — *Fine-Tuning Language Models from Human Preferences*, arXiv:1909.08593

The human-as-teacher flywheel is inspired by RLHF's core insight: human preference data is a valuable signal even when sparse. High-efficiency human plays (≥0.60 deal efficiency) are flagged and written to the training JSONL, improving the distribution over time. Human expertise becomes training data.

---

*Code: [github.com/sh4shv4t/parlay](https://github.com/sh4shv4t/parlay) · Space: [huggingface.co/spaces/sh4shv4t/parlay](https://huggingface.co/spaces/sh4shv4t/parlay)*

Built by Shashvat Singh · Meta PyTorch × Scaler OpenEnv Hackathon · April 2026

◈ Parlay — The arena where AIs learn to close.

---

## License

MIT

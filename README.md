---
title: Parlay
emoji: ◈
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
---

# Parlay — The RL Negotiation Arena

> **The arena where AIs learn to close.**

`Python 3.11` | `FastAPI` | `Gemini 2.5 Flash` | `GRPO` | `OpenEnv-style WS`

## Overview

Parlay is a negotiation RL environment + browser game + training stack:

- Three negotiation scenarios and three personas.
- OpenEnv-style WebSocket interface (`reset` / `step` / `state`) on `/env/ws`.
- Theory-of-Mind belief tracking with dense reward shaping.
- Dynamic ZOPA erosion under sustained tension.
- Training pipeline from Gemini self-play data to SFT and GRPO.

Gemini model routing:

- `gemini-2.5-flash-lite` for data generation and self-play.
- `gemini-2.5-flash` for demo gameplay and MCP tools.

## Quickstart

Run exactly in this order:

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
uvicorn main:app --port 8000
open http://localhost:8000
```

## Reward Design

Per-step reward:

`R_t = α·ΔV + β·ToM - δ·C - θ·noise + ψ·bluff + μ·MEV`

Terminal reward:

`R_T = γ·E + ε·S + ζ·D`

Capitulation floor:

`R_T = -ω` when final deal breaches BATNA.

Constants (from `parlay_env/reward.py`):

- `ALPHA=2`, `BETA=5`, `DELTA=3`, `THETA=10`
- `PSI=12`, `MU=8`
- `GAMMA=100`, `EPSILON=20`, `ZETA=15`, `OMEGA=200`

## Training Pipeline

```text
Gemini self-play (training/generate_data.py)
                |
                v
SFT warm start (training/sft_train.py)
                |
                v
GRPO fine-tune (training/grpo_train.py)
                |
                v
Evaluation + comparison (training/evaluate.py, scripts/eval_comparison.py)
```

### Data generation

```bash
python -m training.generate_data --episodes 80 --output data/episodes.jsonl
```

### SFT

```bash
python -m training.sft_train --data data/episodes.jsonl --output checkpoints/sft_1.5b/
```

### GRPO

```bash
BASE_MODEL=checkpoints/sft_1.5b/ python -m training.grpo_train --data data/episodes.jsonl --output models/parlay-grpo
```

## Baseline vs GRPO Results

[Run scripts/eval_comparison.py after training to populate this section]

`results/comparison.png`

## HuggingFace Space

[Space URL here]

## OpenEnv

See `openenv.yaml` for environment manifest metadata and reward spec.

WebSocket endpoint:

`ws://<host>:<port>/env/ws`

## Architecture

- `main.py`: FastAPI entry, routers, static files.
- `parlay_env/`: server, models, grader, reward constants, game theory.
- `agent/`: Gemini client, ToM tracker, self-play runner.
- `game/`: scenarios, tactical cards, leaderboard.
- `dashboard/`: UI and API routes, spectator stream.
- `training/`: dataset generation, SFT, GRPO, evaluation.
- `mcp_server/`: FastMCP tools.
- `tests/`: keyless and module tests.

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

## Testing

### Full suite

```bash
pytest tests/ -v
```

### Focused modules

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

## MCP

Run MCP server:

```bash
python -m mcp_server.server stdio
```

or SSE:

```bash
python -m mcp_server.server sse
```

## License

MIT

# Parlay — The RL Negotiation Arena

> **The arena where AIs learn to close.**

`Python 3.11` | `FastAPI` | `Gemini 2.0 Flash` | `GRPO` | `OpenEnv`

---

## Overview

Parlay is a high-fidelity **reinforcement learning negotiation environment** that ships three things at once:

| Audience | What they get |
|---|---|
| **Hackathon Judges** | A fully playable browser game, an OpenEnv-compliant WebSocket server, an MCP integration layer, and a complete GRPO training pipeline — all in one repo |
| **Players** | A real-time negotiation game with five scenarios, five AI personas (Gemini-powered), Theory of Mind tracking, tactical cards, drift events, and a global leaderboard |
| **B2B / Researchers** | A clean OpenEnv protocol implementation for training negotiation agents; plug in your own model, collect episodes, run GRPO fine-tuning, push to HF Hub |

Parlay is built on:
- **Google Gemini 2.0 Flash** — the AI counterpart, generating persona-consistent responses in real time
- **FastAPI + aiosqlite** — async backend, zero ORM overhead, SQLite for portability
- **OpenEnv protocol** — standard `reset/step/state` WebSocket commands for agent interoperability
- **FastMCP** — universal MCP server supporting both `stdio` and SSE transports
- **HF TRL GRPOTrainer** — two-stage SFT → GRPO pipeline fine-tuning Qwen2.5-7B-Instruct
- **Vanilla JS + Three.js r128** — zero npm, zero build step, runs in any browser

---

## Quick Start

### Prerequisites

- Python 3.11+
- A Google AI Studio API key ([get one free](https://aistudio.google.com/app/apikey))
- (Optional) A Hugging Face token for training and model pushing

### 1. Clone and set up environment

```bash
git clone https://github.com/your-username/parlay.git
cd parlay
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your_key_here
```

### 4. Initialize the database

```bash
python -m scripts.init_db
python -m scripts.seed_scenarios   # optional: adds demo leaderboard entries
```

### 5. Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Open in browser

```
http://localhost:8000          # Game dashboard
http://localhost:8000/train    # Training dashboard
http://localhost:8000/docs     # Interactive API docs (Swagger)
```

---

## API Keys

| Key | Required | Where to get it | Used for |
|---|---|---|---|
| `GOOGLE_API_KEY` | **Yes** | [Google AI Studio](https://aistudio.google.com/app/apikey) | Gemini 2.0 Flash (game AI + data gen) |
| `HF_TOKEN` | Only for training | [Hugging Face Settings](https://huggingface.co/settings/tokens) | Model push to HF Hub |

Set these in your `.env` file (never commit `.env`):

```bash
GOOGLE_API_KEY=AIzaSy...
HF_TOKEN=hf_...
```

---

## Project Structure

```
parlay/
├── main.py                    # FastAPI app entry point
│
├── parlay_env/                # Core RL environment (OpenEnv-compliant)
│   ├── __init__.py
│   ├── server.py              # WebSocket router (reset/step/state endpoints)
│   ├── env.py                 # ParlayEnv class — OpenEnv implementation
│   ├── models.py              # Pydantic models: ParlayState, ParlayAction, BeliefState…
│   ├── reward.py              # Reward coefficients (ALPHA, BETA, GAMMA, OMEGA…)
│   ├── grader.py              # Pure reward functions: step reward, terminal reward
│   ├── game_theory.py         # ZOPA, Nash, Pareto, Shapley, Rubinstein computations
│   └── exceptions.py          # Custom exceptions (InvalidScenarioError, CapitulationError…)
│
├── game/                      # Game logic layer
│   ├── __init__.py
│   ├── scenarios.py           # 5 negotiation scenarios with drift events
│   ├── personas.py            # 5 AI personas with Gemini prompt templates
│   └── session.py             # Session management: active games, turn routing
│
├── agent/                     # AI agent components
│   ├── __init__.py
│   ├── gemini_client.py       # Gemini 2.0 Flash async wrapper
│   ├── tom_tracker.py         # Theory of Mind belief tracker
│   └── tactical.py            # Tactical card execution logic
│
├── dashboard/                 # Frontend (zero npm, zero build)
│   ├── index.html             # Main game UI
│   ├── train.html             # Training monitor UI
│   ├── api.py                 # FastAPI router for dashboard REST endpoints
│   └── static/
│       ├── app.js             # Game WebSocket client + UI logic
│       ├── character.js       # Three.js r128 animated persona character
│       ├── chart_utils.js     # Chart.js reward visualization helpers
│       └── style.css          # CSS with --parlay-* custom properties
│
├── mcp_server/                # MCP integration (stdio + SSE)
│   ├── __init__.py
│   └── server.py              # FastMCP tools: negotiate, get_state, get_leaderboard…
│
├── training/                  # Isolated training pipeline (never imported by game)
│   ├── __init__.py
│   ├── generate_data.py       # Gemini self-play episode generation
│   ├── sft_train.py           # SFTTrainer fine-tuning on top episodes
│   ├── grpo_train.py          # GRPOTrainer RL fine-tuning
│   ├── reward_fn.py           # GRPO reward functions (wraps grader.py)
│   ├── evaluate.py            # Three-bar comparison chart: base vs SFT vs GRPO
│   └── push_to_hub.py         # Upload model to HF Hub
│
├── scripts/
│   ├── __init__.py
│   ├── init_db.py             # Create SQLite schema (idempotent)
│   └── seed_scenarios.py      # Insert demo leaderboard entries
│
├── tests/
│   ├── __init__.py
│   ├── test_grader.py         # Reward computation tests
│   ├── test_game_theory.py    # ZOPA/Nash/Pareto/Shapley tests
│   ├── test_tom.py            # Theory of Mind tracker tests
│   ├── test_reward.py         # Reward constants tests
│   └── test_scenarios.py      # Scenario definition tests
│
├── data/                      # Generated episode JSONL files (gitignored)
├── models/                    # Fine-tuned model checkpoints (gitignored)
├── results/                   # Evaluation charts and metrics (gitignored)
│
├── requirements.txt           # Core dependencies
├── requirements-train.txt     # Training-only dependencies (torch, trl, peft…)
├── .env.example               # Environment variable template
├── .gitignore
├── docker-compose.yml         # Multi-service Docker deployment
├── Dockerfile.game            # Game + dashboard service
├── Dockerfile.env             # OpenEnv WebSocket service
└── Dockerfile.train           # GRPO training service (CUDA)
```

---

## Game Guide

### How to Play

1. **Choose a scenario** — five high-stakes deal types, each with unique ZOPA ranges and drift events
2. **Choose your persona style** — affects how aggressively the AI counterpart responds
3. **Negotiate in natural language** — type your offers and arguments in the chat
4. **Use tactical cards** — spend Credibility Points to play power moves (anchor, BATNA reveal, deadline pressure)
5. **Watch for drift events** — the AI's hidden priorities shift mid-negotiation; adapt or lose ground
6. **Close within 20 turns** — speed bonuses reward efficient closers

### Key Concepts

| Term | Definition |
|---|---|
| **ZOPA** | Zone of Possible Agreement — the range between both parties' walk-away prices where a deal is mutually beneficial |
| **BATNA** | Best Alternative to a Negotiated Agreement — your outside option; the floor below which you'd rather walk away |
| **Nash Bargaining Solution** | The game-theoretically "fair" split of the ZOPA surplus — the midpoint of both BATNAs |
| **Anchor** | Your opening offer. The higher you anchor (as seller), the more the counterpart adjusts from that reference point |
| **Rubinstein Deadline** | The advantage of having more time — patient negotiators extract better deals |
| **Capitulation Cliff** | Accepting below your BATNA triggers a hard -150 penalty (OMEGA). Never capitulate |
| **Theory of Mind** | Parlay tracks the AI's inferred beliefs about you — high ToM accuracy gives a step reward bonus |
| **Drift Event** | A mid-game shock (budget cut, competitor offer, urgency spike) that changes the AI's hidden priorities |

### Tactical Cards

| Card | CP Cost | Effect |
|---|---|---|
| **Anchor High** | 10 CP | Lock in a high reference price — reduces AI's willingness to counter aggressively |
| **BATNA Reveal** | 15 CP | Signal your outside option — increases AI urgency if credible |
| **Deadline Pressure** | 20 CP | Introduce artificial urgency — accelerates AI concessions by 15% |
| **Bundle Offer** | 12 CP | Add non-monetary value — expands the ZOPA by shifting AI utility |
| **Silent Close** | 25 CP | Make a final offer with no further negotiation signal — high risk, high reward |
| **Coalition Play** | 30 CP | Invoke Act 3 coalition mechanics — brings in a third party for multi-issue negotiation |

### Scoring

Your final score is computed by the Parlay Grader:

```
Final Score = Terminal Reward + Cumulative Step Rewards
```

Deal Efficiency is displayed as a percentage:
```
Deal Efficiency = (Final Price - Seller BATNA) / (Buyer BATNA - Seller BATNA)
```

A deal efficiency of 1.0 means you captured the full ZOPA surplus. 0.5 is the Nash fair split.

---

## OpenEnv Protocol

Parlay implements the OpenEnv standard for RL environments over WebSocket.

### Connection

```
ws://localhost:8000/env/ws/{session_id}
```

### Commands

#### `reset` — Start a new episode

```json
{
  "command": "reset",
  "scenario_id": "saas_enterprise",
  "persona": "shark",
  "player_name": "MyAgent"
}
```

Response:
```json
{
  "type": "observation",
  "session_id": "abc-123",
  "scenario_id": "saas_enterprise",
  "persona": "shark",
  "act": 1,
  "step_count": 0,
  "belief": {
    "est_budget": 140000,
    "est_walk_away": 125000,
    "est_urgency": 0.4,
    "est_has_alternative": false,
    "confidence": 0.3
  },
  "credibility_points": 100,
  "offer_history": [],
  "episode_done": false
}
```

#### `step` — Take a negotiation turn

```json
{
  "command": "step",
  "utterance": "I propose an annual contract at $155,000 with a 90-day payment term.",
  "offer_amount": 155000,
  "tactical_move": null
}
```

Response:
```json
{
  "type": "step_result",
  "step_reward": 12.4,
  "cumulative_reward": 12.4,
  "ai_response": "That's ambitious. Our budget ceiling won't stretch that far...",
  "belief": { "...updated belief state..." },
  "tension_score": 0.6,
  "drift_fired": false,
  "episode_done": false
}
```

#### `state` — Query current state without acting

```json
{ "command": "state" }
```

#### `close` — Accept final price and end episode

```json
{
  "command": "close",
  "final_price": 148000
}
```

Response:
```json
{
  "type": "episode_done",
  "total_reward": 287.4,
  "deal_efficiency": 0.82,
  "acts_completed": 2,
  "bluffs_caught": 1,
  "drift_adapted": true,
  "deal_closed": true,
  "leaderboard_rank": 3
}
```

### HTTP REST Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/env/scenarios` | List all scenarios |
| `GET` | `/env/personas` | List all personas |
| `GET` | `/dashboard/leaderboard` | Global leaderboard |
| `GET` | `/dashboard/leaderboard/{scenario_id}` | Per-scenario leaderboard |
| `POST` | `/dashboard/submit` | Submit episode result |
| `GET` | `/docs` | Swagger UI |

---

## MCP Setup

Parlay ships a universal MCP server supporting **both** `stdio` and SSE transports.

### Available MCP Tools

| Tool | Description |
|---|---|
| `negotiate` | Send a negotiation message and get the AI's response |
| `get_state` | Retrieve current session state and belief model |
| `reset_session` | Start a new negotiation session |
| `close_deal` | Accept a final price and get episode grade |
| `get_leaderboard` | Fetch top performers globally or by scenario |
| `list_scenarios` | Get all available scenarios with ZOPA ranges |
| `list_personas` | Get all personas with strategy profiles |
| `get_game_theory` | Compute ZOPA, Nash point, Rubinstein advantage for any deal |

### Client 1: Claude Desktop / Claude Code

Add to your `claude_desktop_config.json` (usually at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "parlay": {
      "command": "python",
      "args": ["-m", "mcp_server.server", "stdio"],
      "cwd": "/path/to/parlay",
      "env": {
        "GOOGLE_API_KEY": "your_key_here"
      }
    }
  }
}
```

### Client 2: Continue.dev / Zed / Any SSE Client

First start the SSE server:

```bash
python -m mcp_server.server sse
# Listening on http://localhost:8002/sse
```

Then configure your client to point at:

```
http://localhost:8002/sse
```

In `Continue.dev` (`~/.continue/config.json`):

```json
{
  "experimental": {
    "modelContextProtocolServers": [
      {
        "transport": {
          "type": "sse",
          "url": "http://localhost:8002/sse"
        }
      }
    ]
  }
}
```

### Client 3: Generic stdio (any MCP-compatible agent)

```bash
python -m mcp_server.server stdio
```

Pipe JSON-RPC messages to stdin; responses arrive on stdout. Compatible with any MCP client library.

---

## Training Pipeline

Parlay uses a two-stage pipeline: **SFT warmup → GRPO fine-tuning**. Never skip the SFT stage — GRPO reward curves are noisy without a warm-started model.

### Stage 1: Generate Self-Play Episodes

Uses Gemini 2.0 Flash to simulate full negotiation episodes across all persona × scenario combinations.

```bash
python -m training.generate_data --episodes 2000 --output data/episodes.jsonl
```

Diversity guarantees enforced:
- Minimum 20 episodes per (persona × scenario) pair = 500 baseline
- 30% noise injection for exploration
- 40% forced drift event rate
- 25% Act 3 coalition scenarios

Each episode record:
```json
{
  "prompt": "You are negotiating a SaaS enterprise deal...",
  "conversation": [...],
  "reward": 247.3,
  "deal_efficiency": 0.79,
  "persona": "shark",
  "scenario_id": "saas_enterprise",
  "acts_completed": 2,
  "tom_accuracy": 0.81,
  "drift_adapted": true,
  "split": "train"
}
```

### Stage 2: SFT Fine-Tuning

Train Qwen2.5-7B-Instruct on the top 60% of episodes by reward:

```bash
python -m training.sft_train \
  --data data/episodes.jsonl \
  --output models/parlay-sft \
  --base_model Qwen/Qwen2.5-7B-Instruct
```

Uses LoRA (r=16, alpha=32) on `q_proj` and `v_proj`. Full fine-tuning is never used.

### Stage 3: GRPO Fine-Tuning

Apply Group Relative Policy Optimization with G=8 generations per prompt:

```bash
python -m training.grpo_train \
  --sft_model models/parlay-sft \
  --data data/episodes.jsonl \
  --output models/parlay-grpo \
  --steps 500
```

GRPO hyperparameters:
- `num_generations=8` (G=8 per prompt)
- `beta=0.001` (low KL coefficient — allows exploration)
- `epsilon=0.2` (clipping range)
- `scale_rewards="batch"` (batch-level reward standardization)
- `learning_rate=5e-7`

### Stage 4: Evaluate

```bash
python -m training.evaluate \
  --base Qwen/Qwen2.5-7B-Instruct \
  --sft models/parlay-sft \
  --grpo models/parlay-grpo \
  --output results/eval.png
```

Produces a three-bar comparison chart: **Base vs SFT vs GRPO** across mean reward, deal efficiency, and bluff detection rate.

### Stage 5: Push to Hub

```bash
python -m training.push_to_hub \
  --model models/parlay-grpo \
  --repo your-username/parlay-negotiator
```

Requires `HF_TOKEN` and `HF_REPO_ID` in `.env`.

---

## Personas

Five AI negotiation personas, each powered by a distinct Gemini 2.0 Flash system prompt:

| Persona | Aggression | Patience | Bluff Rate | Strategy |
|---|---|---|---|---|
| **Shark** | 0.90 | 0.20 | 0.45 | Opens high, concedes slowly, uses deadline pressure, willing to walk away |
| **Diplomat** | 0.30 | 0.80 | 0.10 | Relationship-focused, seeks mutual gain, rarely bluffs, prefers bundle deals |
| **Analyst** | 0.50 | 0.70 | 0.15 | Data-driven, requests justification for every number, ZOPA-aware, systematic |
| **Veteran** | 0.65 | 0.85 | 0.30 | Pattern-recognizes anchors, absorbs pressure, uses silence as a tool |
| **Wildcard** | 0.75 | 0.35 | 0.55 | Unpredictable, drift-prone, high bluff rate, can pivot strategy mid-negotiation |

Persona drift events can cause a **Wildcard** to briefly adopt **Shark** tactics, or a **Diplomat** to reveal an unexpected BATNA. Adapt or get caught off guard.

---

## Scenarios

Five negotiation scenarios spanning B2B deal archetypes:

| Scenario ID | Title | ZOPA Range | Complexity | Drift Events |
|---|---|---|---|---|
| `saas_enterprise` | Enterprise SaaS Annual License | $125K – $165K | Medium | Budget cut at turn 7 |
| `consulting_retainer` | Consulting Retainer Contract | $8K – $15K/mo | Medium | Competitor reveal at turn 5 |
| `hiring_package` | Senior Engineering Hire Package | $180K – $240K | Low | Competing offer at turn 6 |
| `vendor_hardware` | Hardware Vendor Bulk Purchase | $2.1M – $3.4M | High | Supply chain shock at turn 8 |
| `acquisition_term_sheet` | Startup Acquisition Term Sheet | $8.5M – $16M | Very High | Board veto threat at turn 10, valuation dispute at turn 14 |

Each scenario defines:
- `batna_buyer`: Buyer's walk-away ceiling
- `batna_seller`: Seller's walk-away floor
- `anchor_buyer`: Typical buyer opening offer
- `anchor_seller`: Typical seller opening ask
- `drift_events`: List of mid-game shocks with trigger turns and effects
- `currency`: Always USD
- `difficulty`: `low | medium | high | very_high`

---

## Reward Function

The Parlay grader computes rewards in two phases:

### Step Reward (per turn)

```
r_step = α · ΔZOPA_position
       + β · ToM_accuracy_improvement
       - δ · concession_magnitude
       - θ · noise_penalty
       + ε · tactical_card_bonus
```

Where:
- **α (ALPHA = 2.0)** — reward for improving your ZOPA position
- **β (BETA = 5.0)** — reward for improving ToM belief accuracy
- **δ (DELTA = 1.5)** — penalty per unit of concession from previous offer
- **θ (THETA = 3.0)** — penalty for low-grounding utterances (noise)
- **ε (EPSILON = 8.0)** — bonus for successful tactical card execution

### Terminal Reward (episode end)

```
r_terminal =
  if final_price < batna_seller:      -Ω        (capitulation cliff: -150)
  elif deal_closed:
    Γ                                            (base close bonus: +100)
    + ζ · deal_efficiency                        (ZOPA capture: up to +50)
    + η · acts_completed                         (multi-act bonus: +10/act)
    + Γ · (1 - t_close/t_max)                   (speed bonus: up to +100)
    + ETA · drift_adapted                        (drift adaptation: +10)
  else (no deal):
    -Γ/2 + β · avg_tom_accuracy                 (partial credit)
```

Where:
- **Γ (GAMMA = 100.0)** — primary close bonus
- **ζ (ZETA = 50.0)** — ZOPA efficiency multiplier
- **η (ETA = 10.0)** — per-act completion bonus (max 3 acts = +30)
- **Ω (OMEGA = 150.0)** — capitulation cliff penalty
- **t_close** — turn at which deal was closed
- **t_max** — maximum turns (default: 20)

All coefficients live exclusively in `parlay_env/reward.py`. Never hardcode them elsewhere.

---

## Docker

### Run all services

```bash
cp .env.example .env
# Set GOOGLE_API_KEY in .env

docker compose up --build
```

Services:
- `game` → `http://localhost:8000` — game dashboard + API
- `env` → `http://localhost:8001` — OpenEnv WebSocket server
- `mcp` → `http://localhost:8002` — MCP SSE server

### Run training (requires GPU)

```bash
docker build -f Dockerfile.train -t parlay-train .
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
  -e GOOGLE_API_KEY=$GOOGLE_API_KEY \
  -e HF_TOKEN=$HF_TOKEN \
  parlay-train python -m training.grpo_train --steps 500
```

### Individual services

```bash
# Game only
docker build -f Dockerfile.game -t parlay-game .
docker run -p 8000:8000 -e GOOGLE_API_KEY=$GOOGLE_API_KEY parlay-game

# OpenEnv only
docker build -f Dockerfile.env -t parlay-env .
docker run -p 8001:8001 -e GOOGLE_API_KEY=$GOOGLE_API_KEY parlay-env
```

---

## Testing

### Run the full test suite

```bash
pytest tests/ -v
```

### Run with coverage

```bash
pytest tests/ -v --tb=short --cov=parlay_env --cov=game --cov=agent --cov-report=term-missing
```

### Run a specific test module

```bash
pytest tests/test_grader.py -v
pytest tests/test_game_theory.py -v
pytest tests/test_tom.py -v
pytest tests/test_reward.py -v
pytest tests/test_scenarios.py -v
```

### Test descriptions

| File | What it tests |
|---|---|
| `test_grader.py` | Step reward, terminal reward, episode grade computation |
| `test_game_theory.py` | ZOPA, Nash bargaining, Pareto frontier, Shapley value, anchoring, Rubinstein |
| `test_tom.py` | Theory of Mind tracker: belief updates, bluff detection, drift events, accuracy |
| `test_reward.py` | Reward coefficient constants and their mathematical constraints |
| `test_scenarios.py` | Scenario definitions: ZOPA validity, drift events, currency, IDs |

All tests follow the pattern: `Test{Module}` class → `test_{scenario}` methods → `assert ... f"Expected {expected}, got {result}"`.

---

## Architecture Decisions

### Why SQLite over Postgres?

Parlay is designed to be a **zero-infrastructure hackathon demo**. SQLite with `aiosqlite` provides full async support, requires no Docker service for the database, and the `parlay.db` file can be committed for demo snapshots. Migrating to Postgres requires only changing the connection string.

### Why Vanilla JS over React/Vue?

The `.cursorrules` mandate: zero npm, zero build step. Three.js r128 from cdnjs gives us 3D animated personas. Chart.js 4.4 gives us reward curves. `fetch()` + `WebSocket` gives us real-time game state. The entire frontend loads from a single HTML file with `<script>` tags. This means anyone can open the dashboard without `node_modules`.

### Why GRPO over PPO?

GRPO (Group Relative Policy Optimization) eliminates the need for a separate critic/value model. With G=8 generations per prompt, GRPO uses within-group reward standardization as its baseline — simpler, more stable, and better suited to the sparse reward structure of negotiation episodes.

### Why Gemini 2.0 Flash?

- Free tier available via Google AI Studio (critical for hackathon accessibility)
- Sub-500ms latency for negotiation turns with `max_output_tokens=500`
- Strong instruction-following for persona-consistent responses
- Async-compatible via `run_in_executor` pattern

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | **Required.** Google AI Studio API key |
| `HF_TOKEN` | — | Hugging Face token (training only) |
| `ENV_PORT` | `8001` | OpenEnv WebSocket server port |
| `DASHBOARD_PORT` | `8000` | Dashboard + game server port |
| `MCP_SSE_PORT` | `8002` | MCP SSE server port |
| `MAX_TURNS_PER_EPISODE` | `20` | Maximum turns before episode ends |
| `MIN_REWARD_THRESHOLD` | `-100` | Minimum reward for SFT data inclusion |
| `TOP_PLAYER_THRESHOLD` | `0.60` | Percentile cutoff for SFT training data |
| `CREDIBILITY_POINTS_START` | `100` | Starting CP for tactical cards |
| `CREDIBILITY_REGEN_PER_TURN` | `5` | CP regenerated each turn |
| `BASE_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | HF model ID for training base |
| `GRPO_GENERATIONS` | `8` | G value for GRPO (generations per prompt) |
| `GRPO_STEPS` | `500` | GRPO training steps |
| `DATA_PATH` | `data/episodes.jsonl` | Episode data for training |
| `SFT_OUTPUT` | `models/parlay-sft` | SFT checkpoint output path |
| `GRPO_OUTPUT` | `models/parlay-grpo` | GRPO checkpoint output path |
| `HF_REPO_ID` | — | HF Hub repo for model push |

---

## Contributing

1. Fork the repo and create a feature branch
2. Follow the module dependency graph: `training/ → parlay_env/ → game/ → agent/`
3. Add type hints and docstrings to all public functions
4. Write at least 2 tests per new function (happy path + edge case)
5. Run `pytest tests/` — all tests must pass
6. Verify `docker compose up --build` completes without errors

---

## License

MIT License

Copyright (c) 2026 Parlay Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

*Built for the Meta Hackathon 2026. Powered by Gemini 2.0 Flash + Qwen2.5-7B.*

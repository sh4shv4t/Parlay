# ◈ Parlay — I Built an AI That Finally Beats Me at Negotiation

<p align="center">
  <img src="images/Parlay_square%20logo.png" alt="Parlay logo" width="220">
</p>

*Teaching language models to close deals under hidden information, bluffing, and a world that doesn't stand still.*

---

I've always loved games with hidden information. In particular the games where you *don't know* what the other side is holding. Poker. Diplomacy. And, as I got deeper into it, I got interested in real negotiation.

I got genuinely obsessed with things like Game Theory. I started beating the negotiation NPCs in games embarrassingly fast.

Then I ran out of opponents worth playing against.

When I saw the themes of the hackathon I knew I had to build one. And then I realised I'd actually built something that could train AI to negotiate and train *people* to close deals against AI that never goes easy. That's Parlay.

---

## The Problem: LLMs Can't Actually Negotiate

Language models are genuinely impressive at *describing* negotiation. Ask one to explain the Nash Bargaining Solution and it'll write you a textbook page. But put one in a real deal with hidden information, an adaptive opponent, mounting tension, and a world that keeps changing mid-conversation and it collapses.

It crashes under pressure. It ignores what the opponent's behaviour is *revealing* about their hidden constraints. It doesn't notice when an external shock has just changed what the deal is worth to both sides.

The core issue: negotiation isn't a question-answering task. It's a **Markov Decision Process with partial observability, strategic deception, and sparse terminal outcomes**. I needed an environment that trains for *that*, not fine-tuning on negotiation transcripts.

No existing RL environment had been designed around this, so I built Parlay.

---

## The Environment: Negotiation as an MDP

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
| **Veteran** | Strategic silence, mirrors your language, models <em>your model of them</em> |

The agent never sees the full state. The opponent's true walk-away price, urgency score, and budget ceiling are **hidden**  and the agent has to *infer* them from behaviour.

---

## The Reward Signal: Where It Gets Interesting

Every negotiation RL paper before this rewards deal outcome only. That produces agents that learn to anchor aggressively and get lucky but are brittle, framing-sensitive and have no genuine strategic reasoning.

Parlay's reward function has two components.

### Per-Step Reward (shapes *how* the agent negotiates)

$$
R_t = \underbrace{\alpha \cdot \Delta V_t}_{\text{ZOPA progress}} + \underbrace{\beta \cdot \text{ToM}_{t}}_{\text{belief accuracy}} - \underbrace{\delta \cdot C_t}_{\text{capitulation penalty}} - \underbrace{\theta \cdot \text{noise}_{t}}_{\text{incoherence penalty}} + \underbrace{\psi \cdot \text{Bluff}_{t}}_{\text{bluff detection}} + \underbrace{\mu \cdot \text{MEV}_{t}}_{\text{market event inference}}
$$

The **ToM term** is the novel contribution. At each step, the grader compares the agent's belief state against the opponent's hidden ground truth:

$$
\text{ToM}_{t} = 1 - \frac{1}{|B|}\sum_{i \in B} \frac{|\hat{b}_i - b_i|}{\text{range}_i}
$$

where $B = \{\text{budget, urgency, walk-away}\}$. An agent can get a good deal *and* have terrible ToM accuracy (lucky anchor). With this term, it has to develop genuine mental state inference to maximise cumulative reward.

The **Bluff term** fires when the opponent has played `batna_reveal` with a stated floor more than 15% off their true floor, and the agent's utterance contains a skepticism signal. Catching bluffs earns $\psi = 12$. Getting fooled costs edge.

The **MEV term** is the second-order challenge (see below).

### Terminal Reward (rewards *outcomes*)

$$
R_T = \underbrace{\gamma \cdot E}_{\text{deal efficiency}} + \underbrace{\epsilon \cdot S}_{\text{speed bonus}} + \underbrace{\zeta \cdot D}_{\text{drift adaptation}} - \underbrace{\omega \cdot \mathbf{1}[\text{price} < \text{BATNA}]}_{\text{capitulation cliff}}
$$

where deal efficiency $E = \frac{\text{final price} - \text{BATNA}_{\text{self}}}{\text{ZOPA width}} \in [0, 1]$.

The $\omega = 200$ term is **intentionally discontinuous**. Any smooth penalty can be overcome by a high deal value. The cliff makes the agent's floor absolute — it learns there is a line it simply cannot cross.

All coefficients live in [`parlay_env/reward.py`](https://github.com/sh4shv4t/Parlay/blob/main/parlay_env/reward.py) as the single source of truth. Nothing is hardcoded elsewhere.

---

## The ZOPA Collapse Mechanic

Here's something no prior negotiation RL environment has modelled: **prolonged conflict destroys the deal itself.**

In real negotiations, if both sides dig in and tension keeps rising, alternative options look better, trust erodes, the zone where a deal is even possible shrinks. Parlay models this explicitly.

Every turn where tension exceeds 75, a streak counter increments. After 3 consecutive turns above threshold:

$$
\text{BATNA}_{\text{buyer}} \mathrel{-}= \Delta_{\text{orig}} \cdot r_{\text{erosion}}, \quad \text{BATNA}_{\text{seller}} \mathrel{+}= \Delta_{\text{orig}} \cdot r_{\text{erosion}}
$$

where $\Delta_{\text{orig}}$ is the original ZOPA width and $r_{\text{erosion}} = 0.02$. Using the original width (not the current shrinking width) ensures collapse actually terminates rather than being asymptotic.

The agent can see `zopa_width_pct_remaining` in every observation. The ZOPA bar in the UI shifts from gold → amber → scarlet as the deal zone shrinks.

**This creates a genuine multi-objective challenge**: maximise your share of the deal AND preserve the space where a deal is even possible. That's strictly harder than fixed-ZOPA negotiation, and strictly closer to reality.

---

## Market Event Valuation: The Second Inference Layer

Between turns, exogenous events fire like for example a Fed rate hike, a competitor product recall, a key employee departure, etc. The **headline is public** but the true impact on each party's walk-away price is **hidden and persona-specific**.

```text
⚡ BREAKING — Federal Reserve raises rates 50 basis points
```

The Shark (mev_sensitivity=0.65) recalculates aggressively. The Veteran (mev_sensitivity=0.20) barely blinks. The agent has to estimate the impact before its next offer and consider it in its next action.

The grader compares this against ground truth:

$$
\text{MEV}_{t} = 1 - \frac{|\hat{v} - v^*|}{0.30}
$$

This is a second-order ToM problem which does not just think about "what does the opponent want?" but "how did this external shock update what the opponent wants, and by how much more or less than it updated me?"

No prior negotiation RL paper has this layer.

---

## Training Pipeline

**Google Colab:** [parlay_sft_colab](https://colab.research.google.com/drive/1x5uZMbdKF7XeDNm-bM5YSPdpd1srgArA?usp=sharing) · [parlay_grpo_hf_job](https://colab.research.google.com/drive/1DNYogmRlR_YJrEO6GN3YC7xj8lfycDuL?usp=sharing) (in-repo: [`training/notebooks/parlay_sft_colab.ipynb`](https://github.com/sh4shv4t/Parlay/blob/main/training/notebooks/parlay_sft_colab.ipynb), plus [`training/GRPO_HF_RUNBOOK.md`](https://github.com/sh4shv4t/Parlay/blob/main/training/GRPO_HF_RUNBOOK.md) / `scripts/hf_grpo_entry.sh` for the job-style GRPO run).

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

I use GRPO ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)) for the same reason DeepSeek-R1 did. It eliminates the value model, halves memory, and is more stable for verifiable reward domains where every move can be graded. The negotiation outcome is always verifiable as either the deal was above BATNA or it wasn't and either the belief was accurate or it wasn't.

The ω warmup is a practical detail worth flagging: at step 0, the base model occasionally breaches the BATNA floor (it doesn't know where the floor is). Each breach gives -200, which drowns all positive signal. Starting at ω=50 gives the model enough runway to learn the floor before the cliff becomes absolute.

---

## Results

The table below is the digest version. The training curves are where you *see* the two-stage story: SFT is **teaching the model to play the game by the rules**; GRPO is **making the grader the teacher**.

### SFT: why the loss dives, then whispers

<p align="center">
  <img src="images/sft_loss_curve.png" alt="SFT training loss — Qwen2.5-1.5B + LoRA on Parlay episodes" width="720">
  <br>
  <em><strong>Figure.</strong> Supervised loss on high-quality, filtered negotiation episodes, not a generic “chat SFT” mix.</em>
</p>

This is next-token loss on a **tight, structured** dataset, Gemini self-play plus filters that keep only episodes that *already* look like the economics I care about, deals that close well, refusals that are principled, responses that track drift. The **fast early drop** is the model nailing the *syntax* of the task, JSON, turn order, the tactical vocabulary, and the broad statistics of *what good play looks like* in the data. It is a **huge, low-entropy move** in loss space because “talk like a negotiator” is still far easier to learn than “negotiate optimally against a live grader”.

The **shallower later segment** is not “I gave up on training” and in fact is the signature of a model that has *mostly* become a good maximum-likelihood imitator on a *fixed* set of trajectories. Imitation has a **ceiling** that is *below* the best achievable policy, because the dataset cannot contain every future shock, persona twist, or bluff. That is the entire reason the pipeline I built does not stop at SFT. I want the policy **on the manifold of valid play**; then I turn on **GRPO** to optimise the **verifiable** reward that the environment actually grades.

**Bottom line:** a steep left and a soft right on this plot is *exactly* what I want before RL: **a cold start that is already fluent**, so the RL stage spends its budget *searching in useful directions* instead of fighting format errors and nonsense moves.

### GRPO: reward and loss, read like a pro

The GRPO run I ship is a **Hugging Face Job** configuration (L4, 80 steps, small group size **G=2** for budget, SFT base [`sh4shv4t/parlay-sft-1-5b`](https://huggingface.co/sh4shv4t/parlay-sft-1-5b) into [`sh4shv4t/parlay-grpo-1-5b`](https://huggingface.co/sh4shv4t/parlay-grpo-1-5b)). The reward curve is mean batch return from the TRL log (sampled every few steps; full series in-repo as `grpo_train_metrics.json` or regenerated with `scripts/plot_grpo_hf_job_curves.py`).

<p align="center">
  <img src="images/grpo_reward_curve.png" alt="Mean batch reward during GRPO on Hugging Face Job" width="720">
  <br>
  <em><strong>Figure.</strong> GRPO mean reward — a stochastic process that should trend upward in expectation, not a line whose every tick must beat the last.</em>
</p>

**If you are waiting for a smooth exponential**, you are reading the wrong domain. Every batch is a *different* roll of personas, scenarios, and opponent noise; the group is only **G=2** completions. Some steps *will* go sideways. The signature of health is not silk-smooth monotonicity; it is a **wobbly** trace whose **envelope** shifts upward as the model learns to land on the high-reward side of the **ZOPA / ToM / format / anti-capitulation** composite. That is what “multi-objective policy gradient under a real environment” *looks* like in the log, honest variance instead of a fake curve fit.

When the curve **flattens**, that is not automatically “stuck” either. It often means the policy has *stopped* doing the *catastrophically* bad moves that used to move the average a lot, so the **advantage** gets smaller, same story as late-stage RLHF, except *my* reward is a **grader** with cliffs and ToM, not a single human thumb up or down.

The **policy loss** in GRPO is a *second* read, and it is **not** a duplicate of the reward plot.

<p align="center">
  <img src="images/grpo_loss_curve.png" alt="GRPO training loss — same run" width="720">
  <br>
  <em><strong>Figure.</strong> The GRPO / PPO-style loss reflects ratios, clipping, and KL, not a single clean CE target — it can and does move in ways that look “wrong” when reward is still doing the right thing on average.</em>
</p>

The objective is **not** “minimise this line as if it were validation cross-entropy.” It is a constrained policy update: the loss can **bump** when the optimiser is **trading** off exploration, KL, and the group-relative return. I still watch it. If the reward trace is *systematically* flat or down while the loss explodes, that is a red flag, but a **divergent pair** in the *healthy* case is: reward asks “*where* in outcome space are the rollouts landing?”, the loss asks “*how much* did the policy *move* from the SFT prior this step?”. Both matter; they are **not the same number**.

**One sentence I would put on a slide:** SFT is **fluency in the world of the deal**; GRPO is **pressure-testing that fluency under the only score that actually counts, the grader in the live loop.**

---

The qualitative shift is more interesting than the numbers. The base model capitulates the moment the Shark sets an aggressive anchor — it treats "that's not workable" as information about true value, not as a tactic. After GRPO training, the same Shark anchor gets met with silence or a counter-anchor. The model has learned that the opening number is a reference point manipulation, not a real constraint.

---

## Try It

The environment is live on [Hugging Face Spaces](https://huggingface.co/spaces/sh4shv4t/Parlay).

You don’t have to stick to API calls and instead you can **interact with the AI models in the web app**: pick a persona and scenario, type offers and tactics, and see how you fare against an opponent shown as a **3D character**. The interface surfaces **ZOPA** (the zone of possible agreement), **Theory-of-Mind**-style belief metrics, and how negotiation moves nudge those bars over time, so the “physics” of the deal is visible alongside the dialogue.

<p align="center">
  <img src="images/Parlay_frontend_interactwithchar.png" alt="Parlay UI: negotiation with a 3D opponent and live ZOPA and ToM metrics" width="920">
  <br>
  <em><strong>Figure.</strong> Playing in the browser: 3D opponent avatar, chat and tactical actions, and live analytics (ZOPA, belief state) as the negotiation evolves.</em>
</p>

You can connect to it directly via OpenEnv:

```python
import openenv

env = openenv.connect("https://huggingface.co/spaces/sh4shv4t/Parlay")
obs = env.reset(persona="veteran", scenario_id="saas_enterprise")

while not obs.done:
    action = your_agent.act(obs)
    obs = env.step(action)
```

Or stay in the Space’s game view and spar there, the 3D personas and metrics make it obvious when you’re winning or eroding the deal zone. I promise the Veteran will make you uncomfortable.

---

## It's Also a Training Ground for Human Negotiators

Here's the angle that I believe has the most value when building this: the same environment that trains AI agents turns out to be a genuinely useful coaching tool for human sales reps.

Most negotiation training goes like this: a sales manager plays the buyer, the rep plays the seller, and they both know the manager is going easy on them because he has a 4pm call. It's not a real test and everyone knows it.

Parlay is different in three ways.

**The AI never goes easy.** The Shark will anchor 35% above your target and hold it. The Veteran will mirror your language back at you and wait. The Diplomat will make you feel good about a deal that's 20% below where you should have closed. None of them have a 4pm call.

**You can watch the hidden state in real time.** The spectator view exposes what you can't see during a live negotiation, things like the opponent's true walk-away price, their urgency score, whether they're bluffing when they reveal their BATNA. A sales manager sitting next to a junior rep can pull up the spectator URL on a second screen and coach in real time: *"See how his urgency score just jumped? Don't give him the deadline concession, make him ask for it explicitly."*

**The reward signal tells you exactly where you left money on the table.** After every episode, deal efficiency $E$ tells you what fraction of the available ZOPA you captured. If you closed at $148k on a $125k–$165k ZOPA, your efficiency was 57.5%, the Nash point was $145k and you went $3k past it, but you left $17k on the table from your theoretical ceiling. That's a concrete, actionable number, not a vague "good job."

The human-as-teacher flywheel runs in both directions: human plays above the efficiency threshold improve the AI's training distribution, and the AI's trained strategies become the benchmark that human reps train against. The loop compounds.

---

## Limitations

**The ToM term in GRPO training uses keyword proxies, not the full grader.** The full grader computes belief accuracy against hidden ground truth. This requires a grader call per rollout, which slows training. The GRPO reward function uses a faster utterance-level proxy. This is a deliberate tradeoff: the full grader runs during evaluation, the proxy runs during training.

**Three scenarios is narrow.** I started with a small fixed set of scenarios on purpose. If you want to add procurement, licensing, or real estate, the [scenario spec](https://github.com/sh4shv4t/parlay/blob/main/game/scenarios.py) is a clean dataclass. PRs welcome.

**Training data diversity is the next frontier.** Right now the self-play data comes entirely from Gemini-vs-Gemini episodes. **My plan** is to broaden this significantly by firstly scraping real negotiation transcripts from publicly available sources (earnings call Q&As, recorded deal debriefs, negotiation case study databases) and supplementing with episodes generated by a mix of different models. A training set that includes how humans actually negotiate, not just how one LLM simulates negotiation, should produce meaningfully more robust agents. I designed the scenario dataclass to make this drop-in compatible.

---

## What's Next

I designed the human-as-teacher flywheel so that high-quality human plays (deal efficiency ≥ 0.60) can feed back into training data automatically. When that loop closes, the system gets better the more people play it.

The deeper research question: does the MEV inference layer which trains agents to reason about asymmetric exogenous shocks, produce negotiation agents that generalise better to novel scenarios? That's a paper-sized ablation study, and everything needed to run it is already in the repo.

---

## References and Design Decisions

Every mechanic in Parlay traces back to a specific paper. Here is the reading list, and why I put each one in the codebase.

### Nash (1950) — *The Bargaining Problem*, Econometrica 18(2):155–162

The Nash Bargaining Solution gives a closed-form "fair" price **p**<sup>*</sup> = (BATNA<sub>buyer</sub> + BATNA<sub>seller</sub>) / 2, the point that maximises the product of both sides' surplus. This is the gold ◆ diamond on the ZOPA ruler in the UI and the baseline against which deal efficiency **E** is measured. Without a principled notion of "fair", efficiency scoring is arbitrary.

### Shapley (1953) — *A Value for N-Person Games*, Contributions to the Theory of Games 2:307–317

Shapley value computes each player's marginal contribution averaged over all coalition orderings, the game-theoretically fair division for multi-party deals. Built into `game_theory.py` for future multi-party episode support.

### Tversky & Kahneman (1974) — *Judgment Under Uncertainty: Heuristics and Biases*, Science 185(4157):1124–1131

The empirical anchoring coefficient is 0.65 — the first number in a negotiation shifts final settlement by roughly 35% of the gap between anchor and reality. This is why `anchor_high` is the 0 CP card. It's not a game mechanic, it's a documented cognitive bias. `offer_anchoring_effect()` in `game_theory.py` uses this coefficient to predict opponent counters.

### Kahneman & Tversky (1979) — *Prospect Theory*, Econometrica 47(2):263–291

Losses loom larger than gains by a factor of roughly 2.25. Reframing a cost as an ROI calculation exploits this asymmetry, the same number feels different depending on whether it's presented as "what you're paying" vs "what you're getting back." This underpins the `reframe` tactical card design.

### Schelling (1960) — *The Strategy of Conflict*, Harvard University Press

Credible commitment devices shift Nash equilibria. A truthful BATNA reveal changes what's rational for the opponent to offer, because they now know the negotiation has a hard floor. A detected bluff destroys credibility and shifts the equilibrium the other way. This is why `batna_reveal` is the highest-stakes card in the deck, and why the bluff detection reward term ($\psi = 12$) exists.

### Rubinstein (1982) — *Perfect Equilibrium in a Bargaining Model*, Econometrica 50(1):97–109

In alternating-offers models with discount rates, impatience determines who concedes. First-mover advantage decays as patience asymmetry increases: the impatient party's share converges to $\frac{\delta_{2}}{1 + \delta_{2}}$ where $\delta_{2}$ is the opponent's discount factor. The Shark persona's deadline tactics are a direct implementation of this. Manufactured urgency is an attempt to artificially raise your apparent discount rate.

### Raiffa (1982) — *The Art and Science of Negotiation*, Harvard University Press

Integrative bargaining: when parties have different priority orderings across issues, both can gain without price movement. The `sweetener` card design came from this because adding a non-price concession creates joint surplus when the concession costs you less than it's worth to the opponent.

### Sutton & Barto (2018) — *Reinforcement Learning: An Introduction* (2nd ed), MIT Press

The formal MDP framing — state, action, reward, transition — and all mathematical notation in the reward section come from here. Every design decision in the environment maps back to the MDP formalism: hidden state is the partial observability, drift events are non-stationarity, the ZOPA collapse is a state-dependent terminal condition.

### Wei et al. (2025) — *TOMA: Theory of Mind Augmented LLM Agents for Strategic Negotiation*

The direct justification for the $\beta \cdot \text{ToM}_{t}$ reward term. TOMA shows that explicit mental state modeling before utterance generation produces agents that outperform non-ToM baselines by up to 18.9% on negotiation benchmarks. Without this paper, the ToM term is a design intuition. With it, it's a grounded hypothesis with prior empirical support.

### DeepSeek-AI (2025) — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*

The reason for using GRPO over PPO. DeepSeek-R1 demonstrated that group-relative policy optimization without a value model produces stable, efficient training for verifiable reward domains. Negotiation outcomes are verifiable which makes GRPO the natural fit.

### Shao et al. (2024) — *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*

First publication of GRPO as a formal algorithm. Group relative advantage: $A_{i} = \frac{r_{i} - \text{mean}(r_{1..G})}{\text{std}(r_{1..G})}$. The $G=4$ completions per prompt in Parlay's training config comes directly from this paper's ablations on group size.

### Camerer et al. (2004) — *A Cognitive Hierarchy Model of Games*, QJE 119(3):861–898

The k-level reasoning model maps directly to the Veteran persona's `tom_depth=0.92` parameter. Level-0 players act randomly, level-1 players best-respond to level-0, level-2 players best-respond to level-1. The Veteran operates at k=2 — it models your model of it, not just your stated position. This is also why the Veteran is the hardest opponent and the best training signal for developing genuine ToM.

### Ziegler et al. (2019) — *Fine-Tuning Language Models from Human Preferences*, arXiv:1909.08593

The human-as-teacher flywheel is inspired by RLHF's core insight: human preference data is a valuable signal even when sparse. High-efficiency human plays (≥0.60 deal efficiency) are flagged and written to the training JSONL, improving the distribution over time. Human expertise becomes training data.

---

*Code: [github.com/sh4shv4t/Parlay](https://github.com/sh4shv4t/Parlay) · Space: [huggingface.co/spaces/sh4shv4t/Parlay](https://huggingface.co/spaces/sh4shv4t/Parlay)*

*— Shashvat Singh*

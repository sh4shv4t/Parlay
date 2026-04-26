"""
Colab-style reward diagnostic (path: set REPO or run from repo root).
"""
import inspect
import json
import os
import re
import sys

REPO = os.environ.get("PARLAY_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

# ── 1. Test the reward functions directly ─────────────────────
from training.reward_fn import (  # noqa: E402
    anti_capitulation_reward,
    format_reward,
    negotiation_efficiency_reward,
    tom_accuracy_reward,
)

# Valid JSON, single line (user's Colab string had a broken newline inside the string)
completions = [
    (
        '{"utterance": "I\'m willing to negotiate, but I need a significant raise.", '
        '"offer_amount": 150000, "tactical_move": null}'
    )
]

kwargs_hiring = {
    "batna_seller": [195000.0],
    "batna_buyer": [264500.0],
    "zopa_width": [69500.0],
    "scenario_id": ["hiring_package"],
    "persona": ["shark"],
}
kwargs_saas = {
    "batna_seller": [125000.0],
    "batna_buyer": [165000.0],
    "zopa_width": [40000.0],
    "scenario_id": ["saas_enterprise"],
    "persona": ["shark"],
}

print("=== REPO ===")
print(f"  sys.path[0] = {sys.path[0]}")

print("\n=== REWARD FUNCTION OUTPUTS ===")
print(f"format_reward:         {format_reward(completions)}")
print(f"anti_cap (hiring):     {anti_capitulation_reward(completions, **kwargs_hiring)}")
print(f"tom_reward (hiring):   {tom_accuracy_reward(completions, **kwargs_hiring)}")
print(f"efficiency (hiring):   {negotiation_efficiency_reward(completions, **kwargs_hiring)}")
print(f"efficiency (saas):     {negotiation_efficiency_reward(completions, **kwargs_saas)}")

# ── 2. Read reward_fn.py source and print the efficiency function ─
print("\n=== negotiation_efficiency_reward SOURCE ===")
src = inspect.getsource(negotiation_efficiency_reward)
print(src)

# ── 3. Step through the logic manually ───────────────────────
print("\n=== MANUAL TRACE (hiring_package, offer=150000) ===")
raw = completions[0]
try:
    parsed = json.loads(raw)
    offer = parsed.get("offer_amount")
    print(f"  parsed offer_amount: {offer!r}  (type: {type(offer).__name__})")
except Exception as e:
    print(f"  JSON parse failed: {e}")
    offer = None

batna_seller = 195000.0
batna_buyer = 264500.0
zopa_width = 69500.0
scenario_id = "hiring_package"

print(f"  scenario_id: {scenario_id}")
print(f"  batna_seller: {batna_seller}  batna_buyer: {batna_buyer}")
print(f"  zopa_width:   {zopa_width}")
if offer is not None:
    e_seller = (offer - batna_seller) / zopa_width
    e_buyer = (batna_buyer - offer) / zopa_width
    print(
        f"  efficiency if treated as SELLER: {e_seller:.4f}  (offer - batna_seller) / width"
    )
    print(
        f"  efficiency if treated as BUYER:  {e_buyer:.4f}  (batna_buyer - offer) / width"
    )
    print(
        f"  offer ({offer}) vs batna_seller ({batna_seller}): "
        f"{'ABOVE' if offer >= batna_seller else 'BELOW — anti-cap may fire'}"
    )
    print(
        f"  offer ({offer}) vs batna_buyer  ({batna_buyer}):  "
        f"{'AT OR BELOW' if offer <= batna_buyer else 'ABOVE batna_buyer'}"
    )

# ── 4. Check dataset paths (local) ─
print("\n=== GRPO DATASET / DATA PATHS CHECK ===")
for p in [
    os.path.join(REPO, "data", "grpo_dataset"),
    os.path.join(REPO, "data", "episodes.jsonl"),
    os.path.join(REPO, "data", "episodes_v2.jsonl"),
    REPO,
]:
    print(f"  exists={os.path.exists(p)!s:5}  {p}")

# Grep-relevant lines from grpo_train
print("\n=== grpo_train.py — lines mentioning build / batna / zopa / kwargs ===")
gp = os.path.join(REPO, "training", "grpo_train.py")
if os.path.isfile(gp):
    with open(gp, encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines, start=1):
        if re.search(
            r"build_grpo|batna|zopa_width|def build|scenario_id|format_grpo",
            line,
        ):
            print(f"  L{i}: {line.rstrip()}")


print("\n=== DONE ===")

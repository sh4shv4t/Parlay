.\venv-train\Scripts\python -m training.evaluate `
    --base Qwen/Qwen2.5-7B-Instruct `
    --sft models/parlay-sft `
    --grpo models/parlay-grpo `
    --data data/episodes.jsonl `
    --output results/eval_results.json

.\venv-train\Scripts\python -m training.sft_train `
    --model Qwen/Qwen2.5-7B-Instruct `
    --data data/episodes.jsonl `
    --output models/parlay-sft `
    --threshold 0.60

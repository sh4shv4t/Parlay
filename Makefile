# Parlay — Windows-oriented Makefile (GNU Make from Git Bash or Chocolatey).
# Uses venv/Scripts/ paths. On Windows, clean uses cmd.exe.

.PHONY: setup setup-train run run-env run-mcp test clean train-data train-sft train-grpo evaluate

ifeq ($(OS),Windows_NT)
SHELL := cmd.exe
.SHELLFLAGS := /C
endif

setup:
	python -m venv venv
	venv\Scripts\python -m pip install --upgrade pip -q
	venv\Scripts\pip install -r requirements.txt -q
	if not exist .env copy .env.example .env
	venv\Scripts\python scripts\init_db.py

setup-train:
	python -m venv venv-train
	venv-train\Scripts\python -m pip install --upgrade pip -q
	venv-train\Scripts\pip install -r requirements.txt -q
	venv-train\Scripts\pip install -r requirements-train.txt

run:
	venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 8000 --reload

run-env:
	venv\Scripts\python -m parlay_env.server

run-mcp:
	venv\Scripts\python -m mcp_server.server stdio

test:
	venv\Scripts\pytest tests/ -v

train-data:
	# hackathon budget default; override with EPISODES=N
	venv-train\Scripts\python -m training.generate_data --episodes 80 --output data/episodes.jsonl

train-sft:
	venv-train\Scripts\python -m training.sft_train --model Qwen/Qwen2.5-7B-Instruct --data data/episodes.jsonl --output models/parlay-sft --threshold 0.30

train-grpo:
	venv-train\Scripts\python -m training.grpo_train --model models/parlay-sft --data data/episodes.jsonl --output models/parlay-grpo --steps 500

evaluate:
	venv-train\Scripts\python -m training.evaluate --base Qwen/Qwen2.5-7B-Instruct --sft models/parlay-sft --grpo models/parlay-grpo --data data/episodes.jsonl --output results/eval_results.json

test-keyless:
	venv\Scripts\pytest tests\test_keyless.py -v

docker-test:
	docker build -t parlay-test . && docker run -p 7860:7860 --env-file .env parlay-test

clean:
	if exist venv rd /s /q venv
	if exist venv-train rd /s /q venv-train
	for /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

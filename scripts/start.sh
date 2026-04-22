#!/bin/bash
# Start both the OpenEnv WebSocket server (internal port 8001)
# and the FastAPI dashboard on port 7860 (what HF Spaces exposes).
set -e

echo "Starting Parlay OpenEnv server on port 8001..."
python -m parlay_env.server &
ENV_PID=$!

echo "Starting Parlay dashboard on port 7860..."
uvicorn main:app --host 0.0.0.0 --port 7860

# If uvicorn exits, clean up the env process
kill $ENV_PID 2>/dev/null || true

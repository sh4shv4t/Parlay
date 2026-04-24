# scripts/ws_path.py
# Run this to get the correct WebSocket URL for the Parlay OpenEnv server
# (standalone: python -m parlay_env.server --port 8001)

import os

_DEFAULT_PORT = int(os.getenv("ENV_PORT", "8001"))
WS_PATH = "/env/ws"
WS_URL = f"ws://127.0.0.1:{_DEFAULT_PORT}{WS_PATH}"

if __name__ == "__main__":
    print(f"Parlay OpenEnv WebSocket URL: {WS_URL}")
    print("When using the combined main app (uvicorn main:app --port 8000):")
    print("  ws://127.0.0.1:8000/env/ws")

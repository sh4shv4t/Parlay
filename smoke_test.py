import asyncio
import json
import sys

import websockets

# Set from Step 1: router prefix "/env" + "/ws" => "/env/ws" on the env server process
WS_URL = "ws://localhost:8001/env/ws"


def _observation(data: dict) -> dict:
    """Reset/step responses wrap the observation; unwrap for assertions."""
    if "observation" in data and isinstance(data["observation"], dict):
        return data["observation"]
    if "error" in data:
        raise AssertionError(f"server error: {data.get('error')}")
    return data


async def smoke_test() -> None:
    print(f"Connecting to {WS_URL}...")
    try:
        async with websockets.connect(
            WS_URL,
            open_timeout=8,
            additional_headers={"Origin": "http://localhost:8000"},
        ) as ws:
            # --- Test 1: reset() ---
            print("Testing reset()...")
            await ws.send(
                json.dumps(
                    {
                        "method": "reset",
                        "params": {
                            "persona": "shark",
                            "scenario_id": "saas_enterprise",
                        },
                    }
                )
            )
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            data0 = json.loads(raw)
            session_id = data0.get("session_id")
            if not session_id and "error" in data0:
                raise AssertionError(f"reset failed: {data0!r}")
            assert session_id, f"reset() missing session_id. Got: {data0!r}"
            obs = _observation(data0)
            print(f"  reset() response keys: {sorted(data0.keys())}")
            # API uses step_count (not turn_count) on the observation
            assert "step_count" in obs, f"reset() missing step_count. Got: {list(obs.keys())}"
            print(
                f"  PASS  reset() | step_count={obs.get('step_count')} "
                f"tension_score={obs.get('tension_score')} "
                f"cumulative_reward={obs.get('cumulative_reward', 'N/A')}"
            )

            # --- Test 2: step() ---
            print("Testing step()...")
            # Use an offer above the scenario ceiling so the episode does not
            # immediately satisfy deal_reached (else step 2 would get "already done").
            await ws.send(
                json.dumps(
                    {
                        "method": "step",
                        "params": {
                            "session_id": session_id,
                            "utterance": "We are prepared to offer 500000 for the annual contract (probe).",
                            "offer_amount": 500_000.0,
                            "tactical_move": None,
                        },
                    }
                )
            )
            raw2 = await asyncio.wait_for(ws.recv(), timeout=20)
            data1 = json.loads(raw2)
            if "error" in data1 and "observation" not in data1:
                raise AssertionError(f"step failed: {data1!r}")
            obs2 = _observation(data1)
            print(f"  step() envelope keys: {sorted(data1.keys())}")
            print(f"  step() observation keys: {sorted(obs2.keys())}")
            assert "reward" in obs2, f"step() missing reward. Got: {list(obs2.keys())}"
            print(
                f"  PASS  step() | reward={obs2.get('reward')} "
                f"step_count={obs2.get('step_count')} "
                f"tension_score={obs2.get('tension_score')}"
            )

            # --- Test 3: state() ---
            print("Testing state()...")
            await ws.send(
                json.dumps({"method": "state", "params": {"session_id": session_id}})
            )
            raw3 = await asyncio.wait_for(ws.recv(), timeout=8)
            st = json.loads(raw3)
            state = st.get("state", st) if isinstance(st, dict) else st
            print(f"  state() response keys: {sorted(st.keys()) if isinstance(st, dict) else st}")
            assert isinstance(state, dict) and len(state) > 0, "state() returned empty"
            print(f"  PASS  state() | {len(state)} keys in state payload")

            # --- Test 4: ZOPA erosion fields in observation ---
            print("Testing ZOPA erosion fields...")
            has_erosion = "zopa_erosion_ticks" in obs2 or "zopa_width_pct_remaining" in obs2
            if has_erosion:
                print(
                    f"  PASS  ZOPA erosion fields present | "
                    f"ticks={obs2.get('zopa_erosion_ticks', 0)} "
                    f"width_pct={obs2.get('zopa_width_pct_remaining', 1.0):.2f}"
                )
            else:
                print(f"  WARN  ZOPA erosion fields NOT in obs2 keys: {sorted(obs2.keys())}")
                print(
                    "        This means the ZOPA collapse mechanic is not "
                    "surfaced in the observation model."
                )

            # --- Test 5: reward variance (not always 0.0) ---
            print("Testing reward is non-trivial...")
            reward_val = obs2.get("reward")
            assert isinstance(reward_val, (int, float)), f"reward is not numeric: {type(reward_val)}"
            print(f"  PASS  reward is numeric: {reward_val}")
            if reward_val == 0.0:
                print(
                    "  WARN  reward is exactly 0.0 — step rewards may not "
                    "be accumulating. Check grader.py compute_step_reward."
                )

            # --- Test 6: last utterance in observation (player line echoed in this env) ---
            print("Testing utterance in observation...")
            last_u = obs2.get("last_utterance", "")
            if last_u:
                preview = (last_u[:80] + "...") if len(last_u) > 80 else last_u
                print(f"  PASS  last_utterance set: '{preview}'")
            else:
                print("  WARN  last_utterance empty in obs2")

            # --- Test 7: second step to verify turn increments ---
            print("Testing turn increment on second step...")
            await ws.send(
                json.dumps(
                    {
                        "method": "step",
                        "params": {
                            "session_id": session_id,
                            "utterance": "I understand your position. We could go to 501000 (still probing).",
                            "offer_amount": 501_000.0,
                            "tactical_move": None,
                        },
                    }
                )
            )
            raw4 = await asyncio.wait_for(ws.recv(), timeout=20)
            data2 = json.loads(raw4)
            obs3 = _observation(data2)
            turn2 = obs3.get("step_count", -1)
            turn1 = obs2.get("step_count", -1)
            assert turn2 > turn1, f"Step count did not increment: was {turn1}, now {turn2}"
            print(f"  PASS  Step incremented: {turn1} -> {turn2}")

            print()
            print("=" * 55)
            print("  ALL SMOKE TESTS PASSED")
            print("  The WebSocket env is working correctly.")
            print("=" * 55)
            print()
            print("QUICK STATS:")
            print(f"  WebSocket URL:     {WS_URL}")
            print(f"  Reward at step 1:  {obs2.get('reward')}")
            print(f"  Reward at step 2:  {obs3.get('reward')}")
            print(f"  Tension (step 2): {obs3.get('tension_score')}")
            print(f"  Step count:        {turn2}")

    except ConnectionRefusedError:
        print()
        print("FAIL  ConnectionRefused — server is not running on this port.")
        print(f"      Tried: {WS_URL}")
        print("      Start the env server first, then rerun this test.")
        sys.exit(1)
    except websockets.exceptions.InvalidStatus as e:
        print()
        print(f"FAIL  Server rejected connection: {e}")
        code = getattr(e, "response", None)
        status = code.status_code if code is not None and hasattr(code, "status_code") else "unknown"
        print(f"      HTTP status: {status}")
        print("      This is the 403 bug — wrong path or host policy.")
        print("      Confirm WS_URL is ws://localhost:PORT/env/ws (not /ws).")
        sys.exit(1)
    except AssertionError as e:
        print()
        print(f"FAIL  Assertion failed: {e}")
        sys.exit(1)
    except asyncio.TimeoutError:
        print()
        print("FAIL  Timeout — server connected but did not respond in time.")
        print("      The Gemini API call in step() may be hanging.")
        print("      Check GOOGLE_API_KEY is set, or that mock mode is active.")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"FAIL  Unexpected error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(smoke_test())

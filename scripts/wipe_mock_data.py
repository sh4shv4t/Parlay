#!/usr/bin/env python3
"""
wipe_mock_data.py
Removes any training records that were generated in mock/keyless mode.
A record is "bad" (mock) if:
  - offer_amount is None in any turn, OR
  - all rewards across the episode are exactly 0.0 (no real grading happened), OR
  - the record contains the sentinel string "MOCK" in any utterance field.

Run before real data generation to avoid contaminating the training set.
Usage: python scripts/wipe_mock_data.py [--dry-run]
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from datetime import datetime

DATA_FILE = Path("data/episodes.jsonl")
BACKUP_DIR = Path("data/backups")

def is_mock_record(record: dict) -> bool:
    """Return True if this record should be considered mock/bad data."""
    # Sentinel string check
    conversation = record.get("conversation", [])
    for turn in conversation:
        utterance = turn.get("utterance", "") or ""
        if "MOCK" in utterance.upper():
            return True

    # All rewards zero — no real grading
    rewards = record.get("step_rewards", [])
    if rewards and all(r == 0.0 for r in rewards):
        return True

    # Terminal reward exactly 0 with no deal (indicates mock grader output)
    if record.get("reward", 0.0) == 0.0 and not record.get("deal_reached", False):
        # Only flag if there's also no efficiency score (truly ungraded)
        if record.get("deal_efficiency") is None:
            return True

    # offer_amount None in first turn is a strong mock signal
    if conversation:
        first_turn = conversation[0]
        if first_turn.get("offer_amount") is None:
            return True

    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be deleted without deleting")
    args = parser.parse_args()

    if not DATA_FILE.exists():
        print(f"[INFO] {DATA_FILE} does not exist — nothing to wipe.")
        return

    records = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[WARN] Skipping malformed line: {line[:80]}")

    total = len(records)
    good = [r for r in records if not is_mock_record(r)]
    bad  = [r for r in records if is_mock_record(r)]

    print(f"[INFO] Total records:  {total}")
    print(f"[INFO] Good records:   {len(good)}")
    print(f"[INFO] Mock/bad records to remove: {len(bad)}")

    if not bad:
        print("[INFO] No mock data found. Nothing to wipe.")
        return

    if args.dry_run:
        print("[DRY RUN] Would remove the following records (first 5 shown):")
        for r in bad[:5]:
            print(f"  episode_id={r.get('episode_id', '?')}  "
                  f"persona={r.get('persona', '?')}  "
                  f"reward={r.get('reward', '?')}")
        return

    # Back up existing file before modifying
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"episodes_backup_{ts}.jsonl"
    shutil.copy2(DATA_FILE, backup_path)
    print(f"[INFO] Backup saved to {backup_path}")

    # Write only good records back
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        for r in good:
            f.write(json.dumps(r) + "\n")

    print(f"[OK] Wiped {len(bad)} mock records. {len(good)} good records remain.")

if __name__ == "__main__":
    main()

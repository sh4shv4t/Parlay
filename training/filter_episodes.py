"""
Write a filtered JSONL for SFT/GRPO (drop bad offers + extreme rewards).

Does not modify the source file unless --in-place is passed (backup created).

Usage:
    python -m training.filter_episodes --input data/episodes.jsonl --output data/episodes_sft.jsonl
    python -m training.filter_episodes --input data/episodes.jsonl --in-place
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from .episode_filters import SFTFilterConfig, filter_records

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter episode JSONL for training quality")
    parser.add_argument("--input", default="data/episodes.jsonl")
    parser.add_argument("--output", default="", help="If empty and not --in-place, print stats only")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite --input after copying to data/backups/",
    )
    parser.add_argument("--reward-drop-min", type=float, default=-400.0)
    parser.add_argument("--reward-drop-max", type=float, default=400.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = SFTFilterConfig(
        reward_drop_min=args.reward_drop_min,
        reward_drop_max=args.reward_drop_max,
    )

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Missing input: {in_path}")

    records: list[dict] = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    kept, stats = filter_records(records, cfg)
    logger.info(
        f"Filtered {stats['total_in']} -> {stats['kept']} rows | dropped={stats['dropped']}"
    )

    out_path: Path
    if args.in_place:
        backup_dir = Path("data/backups")
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = backup_dir / f"episodes_pre_filter_{ts}.jsonl"
        shutil.copy2(in_path, backup)
        logger.info(f"Backup: {backup}")
        out_path = in_path
    elif args.output:
        out_path = Path(args.output)
    else:
        print(json.dumps(stats, indent=2))
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(kept)} records to {out_path.resolve()}")


if __name__ == "__main__":
    main()

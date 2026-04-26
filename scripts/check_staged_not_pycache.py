"""Fail if any staged file is under __pycache__ or is a .pyc / .pyo (pre-commit local hook)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    if not out:
        return 0
    bad: list[Path] = []
    for raw in out.split(b"\0"):
        if not raw:
            continue
        p = raw.decode("utf-8", errors="replace")
        pl = p.lower()
        if "__pycache__" in p:
            bad.append(Path(p))
        elif pl.endswith(".pyc") or pl.endswith(".pyo"):
            bad.append(Path(p))
    if not bad:
        return 0
    print("Refuse to commit bytecode or __pycache__ paths:", file=sys.stderr)
    for p in bad:
        print(f"  {p}", file=sys.stderr)
    print("Remove from the index: git reset HEAD -- <file>", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

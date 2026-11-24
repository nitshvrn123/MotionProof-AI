"""Verify that frozen D2 checkpoints and evaluation files are unchanged.

Usage:
    python -m src.experiments.verify_frozen
"""

import json
import sys

from src.experiments.common import ROOT, sha256

REGISTRY = ROOT / "configs" / "frozen_artifacts.json"


def check():
    files = json.loads(REGISTRY.read_text())["files"]
    problems = []
    for relative, expected in files.items():
        path = ROOT / relative
        if not path.exists():
            problems.append(f"missing: {relative}")
        elif sha256(path) != expected:
            problems.append(f"changed: {relative}")
    return len(files), problems


def main():
    count, problems = check()
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print(f"All {count} frozen artifacts match their recorded SHA-256.")


if __name__ == "__main__":
    main()

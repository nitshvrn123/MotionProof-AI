"""Score exploratory usability responses (SUS, task success, clarity/trust items).

Usage:
    python -m src.experiments.score_usability docs/usability/responses.csv
"""

import sys

import pandas as pd

SUS_ITEMS = [f"sus{i}" for i in range(1, 11)]
TASKS = ["t1", "t2", "t3", "t4"]
QUESTIONS = ["q1", "q2", "q3", "q4"]
TASK_VALUES = {"yes": 1.0, "partly": 0.5, "no": 0.0}


def sus_score(row):
    """Standard SUS: odd items contribute (x - 1), even items (5 - x); sum times 2.5 gives 0-100."""

    values = [int(row[c]) for c in SUS_ITEMS]
    if any(v < 1 or v > 5 for v in values):
        raise ValueError(f"SUS answers must be 1-5: {values}")
    return 2.5 * sum((v - 1) if i % 2 == 0 else (5 - v) for i, v in enumerate(values))


def score(frame):
    frame = frame.copy()
    frame["sus_score"] = frame.apply(sus_score, axis=1)
    for t in TASKS:
        frame[f"{t}_success"] = frame[t].str.strip().str.lower().map(TASK_VALUES)
        if frame[f"{t}_success"].isna().any():
            raise ValueError(f"{t} must be yes / partly / no")
    summary = {
        "participants": len(frame),
        "sus_mean": frame["sus_score"].mean(),
        "sus_min": frame["sus_score"].min(),
        "sus_max": frame["sus_score"].max(),
        **{f"{t}_success_rate": frame[f"{t}_success"].mean() for t in TASKS},
        **{f"{q}_mean": frame[q].astype(float).mean() for q in QUESTIONS},
    }
    return frame, summary


def main(path):
    frame, summary = score(pd.read_csv(path))
    print(frame[["participant", "sus_score"] + [f"{t}_success" for t in TASKS] + QUESTIONS].to_string(index=False))
    print()
    for k, v in summary.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
    if len(frame) < 5:
        print("\nNote: fewer than 5 participants; report individual scores and the range, not a population estimate.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/usability/responses.csv")

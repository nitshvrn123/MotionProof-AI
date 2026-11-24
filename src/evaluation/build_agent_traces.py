"""Cache learned-policy action traces for the D3 demo's recording discovery.

Replays the frozen fitted Q policy on the saved held-out test inputs, using
the same environment construction as evaluate_final_test_policies.py, and
records the action sequence for every window. Nothing is trained or tuned.

The output is verified against the frozen test_policy_decisions.csv: every
window must reproduce the saved decision and investigation count, otherwise
no file is written.

Usage:
    python -m src.evaluation.build_agent_traces
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.fitted_q_inference import load_fitted_q_policy
from src.evaluation import evaluate_validation_policies as evaluator

ROOT = Path(__file__).resolve().parents[2]
FINAL_TEST = ROOT / "results" / "metrics" / "final_test"
OUTPUT = ROOT / "results" / "demo" / "learned_agent_traces.csv"
SUBJECTS = (13, 14, 15)


def build_traces():
    inputs = pd.read_csv(FINAL_TEST / "test_policy_inputs_nonlinear.csv")
    policy = load_fitted_q_policy(
        ROOT / "checkpoints" / "learned_policy_fitted_q_full.pt",
        device="cpu",
    )

    records = []

    for subject in SUBJECTS:
        with np.load(ROOT / f"data/processed/test/S{subject}.npz") as data:
            ppg = data["ppg"]
            acc = data["acc"]

        rows = (
            inputs.loc[inputs["subject"] == subject]
            .sort_values("window_index")
            .reset_index(drop=True)
        )
        predictions = rows["cnn_predicted_hr"].to_numpy(dtype=np.float64)

        for index, row in rows.iterrows():
            history = predictions[
                max(0, index - evaluator.HISTORY_LENGTH):index
            ]
            environment = evaluator.make_environment(
                ppg[index], acc[index], row, history
            )
            result = policy.run(environment)

            records.append({
                "subject": subject,
                "window_index": int(row["window_index"]),
                "actions": "|".join(result["actions"]),
                "final_decision": result["final_decision"],
                "investigations_used": int(result["investigations_used"]),
            })

    return pd.DataFrame(records)


def verify_against_frozen(traces):
    decisions = pd.read_csv(FINAL_TEST / "test_policy_decisions.csv")
    learned = decisions.loc[decisions["policy"] == "Learned Fitted Q"]

    merged = traces.merge(
        learned[["subject", "window_index",
                 "predicted_unreliable", "investigations_used"]],
        on=["subject", "window_index"],
        suffixes=("", "_frozen"),
        validate="one_to_one",
    )

    if len(merged) != len(learned) or len(merged) != len(traces):
        raise RuntimeError("Trace windows do not match frozen decisions.")

    flagged = (merged["final_decision"] == "FLAG_UNRELIABLE").astype(int)
    decision_match = flagged == merged["predicted_unreliable"]
    budget_match = (
        merged["investigations_used"] == merged["investigations_used_frozen"]
    )

    if not (decision_match.all() and budget_match.all()):
        raise RuntimeError(
            f"Trace mismatch: {int((~decision_match).sum())} decisions, "
            f"{int((~budget_match).sum())} investigation counts."
        )

    return len(merged)


def main():
    traces = build_traces()
    verified = verify_against_frozen(traces)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    traces.to_csv(OUTPUT, index=False)
    print(f"Verified {verified} windows against frozen decisions.")
    print(f"Saved {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

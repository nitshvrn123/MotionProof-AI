"""Analyze learned-agent errors on nonlinear validation inputs."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

METRICS = ROOT / "results/metrics"
OUTPUT = METRICS / "nonlinear_validation"

DECISIONS_PATH = OUTPUT / "validation_policy_decisions.csv"
INPUT_PATH = METRICS / "validation_policy_inputs_nonlinear.csv"


def main():

    print("=" * 75)
    print("MOTIONPROOF AI — STEP 34.1")
    print("LEARNED AGENT ERROR ANALYSIS")
    print("=" * 75)

    decisions = pd.read_csv(DECISIONS_PATH)
    inputs = pd.read_csv(INPUT_PATH)

    learned = decisions[
        decisions["policy"] == "Learned Fitted Q"
    ].copy()

    assert len(learned) == 8475

    keys = ["subject", "window_index"]

    learned = learned.merge(
        inputs[
            keys
            + [
                "cnn_predicted_hr",
                "reference_hr",
                "absolute_error",
            ]
        ],
        on=keys,
        validate="one_to_one",
    )

    y_true = learned["unreliable_target"].to_numpy()
    y_pred = learned["predicted_unreliable"].to_numpy()

    learned["error_type"] = np.select(
        [
            (y_true == 1) & (y_pred == 1),
            (y_true == 0) & (y_pred == 0),
            (y_true == 0) & (y_pred == 1),
            (y_true == 1) & (y_pred == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="UNKNOWN",
    )

    print("\nERROR DISTRIBUTION")
    print("-" * 75)

    print(
        learned["error_type"]
        .value_counts()
        .reindex(["TP", "TN", "FP", "FN"], fill_value=0)
        .to_string()
    )

    print("\nERROR TYPE SUMMARY")
    print("-" * 75)

    summary = (
        learned.groupby("error_type")
        .agg(
            windows=("error_type", "size"),
            mean_probability=("unreliable_probability", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_investigations=("investigations_used", "mean"),
            mean_reward=("reward", "mean"),
        )
        .reset_index()
    )

    print(summary.round(4).to_string(index=False))

    print("\nFALSE POSITIVES BY SUBJECT")
    print("-" * 75)

    fp = learned[learned["error_type"] == "FP"]

    print(
        fp.groupby("subject")
        .agg(
            windows=("error_type", "size"),
            mean_probability=("unreliable_probability", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_investigations=("investigations_used", "mean"),
        )
        .round(4)
        .to_string()
    )

    print("\nFALSE POSITIVES BY PROBABILITY BIN")
    print("-" * 75)

    bins = np.linspace(0, 1, 11)

    learned["probability_bin"] = pd.cut(
        learned["unreliable_probability"],
        bins=bins,
        include_lowest=True,
    )

    probability_summary = (
        learned.groupby(
            ["probability_bin", "error_type"],
            observed=True,
        )
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["TP", "TN", "FP", "FN"], fill_value=0)
    )

    print(probability_summary.to_string())

    print("\nFALSE POSITIVES BY INVESTIGATION COUNT")
    print("-" * 75)

    print(
        fp.groupby("investigations_used")
        .size()
        .to_string()
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    detailed_path = OUTPUT / "learned_agent_error_details.csv"
    summary_path = OUTPUT / "learned_agent_error_summary.csv"

    if detailed_path.exists() or summary_path.exists():
        raise FileExistsError(
            "Error-analysis output already exists. "
            "Refusing to overwrite."
        )

    learned.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nSaved:", detailed_path)
    print("Saved:", summary_path)

    print("\nPASS: Step 34.1 completed.")


if __name__ == "__main__":
    main()

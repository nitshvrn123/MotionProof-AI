"""Validation-only threshold experiments for MotionProof AI."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "results/metrics/validation_policy_decisions.csv"
)

OUTPUT_DIR = ROOT / "results/metrics"

THRESHOLDS = np.round(
    np.arange(0.30, 0.951, 0.025),
    3,
)

TARGET_ACCURACY = 0.75
MINIMUM_RECALL = 0.80


def compute_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "specificity": (
            tn / (tn + fp)
            if (tn + fp) else np.nan
        ),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "correct": int(tp + tn),
    }


def main():

    print("=" * 72)
    print("MOTIONPROOF AI — VALIDATION DECISION OPTIMIZATION")
    print("=" * 72)

    data = pd.read_csv(INPUT_PATH)

    baseline = (
        data[data["policy"] == "No Investigation"]
        .sort_values(["subject", "window_index"])
        .reset_index(drop=True)
    )

    learned = (
        data[data["policy"] == "Learned Fitted Q"]
        .sort_values(["subject", "window_index"])
        .reset_index(drop=True)
    )

    assert len(baseline) == len(learned) == 8475

    assert np.array_equal(
        baseline[["subject", "window_index"]].to_numpy(),
        learned[["subject", "window_index"]].to_numpy(),
    )

    assert np.array_equal(
        baseline["unreliable_target"].to_numpy(),
        learned["unreliable_target"].to_numpy(),
    )

    assert np.allclose(
        baseline["unreliable_probability"].to_numpy(),
        learned["unreliable_probability"].to_numpy(),
    )

    target = baseline[
        "unreliable_target"
    ].to_numpy(dtype=int)

    probability = baseline[
        "unreliable_probability"
    ].to_numpy(dtype=float)

    learned_prediction = learned[
        "predicted_unreliable"
    ].to_numpy(dtype=int)

    records = []

    for threshold in THRESHOLDS:

        # Experiment A:
        # Probability-only classifier.
        probability_prediction = (
            probability >= threshold
        ).astype(int)

        # Experiment B:
        # The learned agent may flag a window only if
        # the reliability probability exceeds threshold.
        # Existing ACCEPT decisions remain ACCEPT.
        conservative_prediction = (
            (learned_prediction == 1)
            & (probability >= threshold)
        ).astype(int)

        experiments = {
            "Probability Only": probability_prediction,
            "Conservative Learned Agent": conservative_prediction,
        }

        for experiment_name, prediction in experiments.items():

            metrics = compute_metrics(
                target,
                prediction,
            )

            records.append({
                "experiment": experiment_name,
                "threshold": float(threshold),
                **metrics,
            })

    results = pd.DataFrame(records)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "validation_threshold_optimization.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 72)
    print("BEST ACCURACY FOR EACH EXPERIMENT")
    print("=" * 72)

    best_accuracy = (
        results.sort_values(
            ["experiment", "accuracy"],
            ascending=[True, False],
        )
        .groupby("experiment")
        .head(1)
    )

    print(
        best_accuracy.round(4).to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("BEST F1 FOR EACH EXPERIMENT")
    print("=" * 72)

    best_f1 = (
        results.sort_values(
            ["experiment", "f1"],
            ascending=[True, False],
        )
        .groupby("experiment")
        .head(1)
    )

    print(
        best_f1.round(4).to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("BEST ACCURACY WHILE PRESERVING >=80% RECALL")
    print("=" * 72)

    eligible = results[
        results["recall"] >= MINIMUM_RECALL
    ]

    if eligible.empty:

        print(
            "No threshold achieved at least 80% recall."
        )

    else:

        best_constrained = (
            eligible.sort_values(
                ["experiment", "accuracy"],
                ascending=[True, False],
            )
            .groupby("experiment")
            .head(1)
        )

        print(
            best_constrained.round(4).to_string(index=False)
        )

    print("\n" + "=" * 72)
    print("75% ACCURACY + 80% RECALL TARGET")
    print("=" * 72)

    successful = results[
        (results["accuracy"] >= TARGET_ACCURACY)
        & (results["recall"] >= MINIMUM_RECALL)
    ]

    if successful.empty:

        print(
            "No tested threshold met both targets."
        )

    else:

        print(
            successful.round(4).to_string(index=False)
        )

    print("\n" + "=" * 72)
    print("PER-SUBJECT RESULTS FOR TOP OVERALL ACCURACY")
    print("=" * 72)

    for _, row in best_accuracy.iterrows():

        experiment = row["experiment"]
        threshold = float(row["threshold"])

        if experiment == "Probability Only":

            prediction = (
                probability >= threshold
            ).astype(int)

        else:

            prediction = (
                (learned_prediction == 1)
                & (probability >= threshold)
            ).astype(int)

        print(
            f"\n{experiment} | threshold={threshold:.3f}"
        )

        for subject in (11, 12):

            mask = (
                baseline["subject"].to_numpy()
                == subject
            )

            metrics = compute_metrics(
                target[mask],
                prediction[mask],
            )

            print(
                f"S{subject}: "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"recall={metrics['recall']:.4f}, "
                f"f1={metrics['f1']:.4f}"
            )

    print("\nSaved:", output_path)

    print(
        "\nPASS: Validation threshold experiments completed."
    )


if __name__ == "__main__":
    main()

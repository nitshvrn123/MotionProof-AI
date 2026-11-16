"""Independent integration checks for frozen MotionProof AI results."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results/metrics/final_test"

INPUT_PATH = RESULT_DIR / "test_policy_inputs_nonlinear.csv"
DECISION_PATH = RESULT_DIR / "test_policy_decisions.csv"
COMPARISON_PATH = RESULT_DIR / "test_policy_comparison.csv"
ABLATION_PATH = (
    RESULT_DIR / "investigation_action_ablation_details.csv"
)

EXPECTED_SUBJECTS = {
    13: 4565,
    14: 4476,
    15: 3966,
}

POLICIES = {
    "Fixed Rule",
    "Learned Fitted Q",
    "No Investigation",
}


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def independently_calculate_metrics(group):
    y_true = group["unreliable_target"].to_numpy(dtype=int)
    y_pred = group["predicted_unreliable"].to_numpy(dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "windows": len(group),
        "accuracy": (tp + tn) / len(group),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_positive_rate": fp / (fp + tn),
        "false_negative_rate": fn / (fn + tp),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "average_investigations": float(
            group["investigations_used"].mean()
        ),
        "average_reward": float(group["reward"].mean()),
        "flag_rate": float(y_pred.mean()),
    }


def main():
    print("=" * 75)
    print("MOTIONPROOF AI — STEP 38.2")
    print("FINAL INTEGRATION VERIFICATION")
    print("=" * 75)

    inputs = pd.read_csv(INPUT_PATH)
    decisions = pd.read_csv(DECISION_PATH)
    comparison = pd.read_csv(COMPARISON_PATH)
    ablation = pd.read_csv(ABLATION_PATH)

    keys = ["subject", "window_index"]

    print("\n1. TEST INPUT INTEGRITY")

    check(len(inputs) == 13007, "13,007 test inputs")
    check(
        not inputs.duplicated(keys).any(),
        "Unique test input identifiers",
    )

    for subject, expected in EXPECTED_SUBJECTS.items():
        group = inputs[inputs["subject"] == subject]
        check(
            len(group) == expected,
            f"S{subject} contains {expected:,} windows",
        )

    calculated_error = np.abs(
        inputs["cnn_predicted_hr"].to_numpy()
        - inputs["reference_hr"].to_numpy()
    )

    check(
        np.allclose(
            calculated_error,
            inputs["absolute_error"].to_numpy(),
            atol=1e-5,
            rtol=0,
        ),
        "Absolute errors match CNN predictions",
    )

    check(
        np.array_equal(
            (calculated_error > 10).astype(int),
            inputs["unreliable_target"].to_numpy(dtype=int),
        ),
        "Unreliable labels use the 10 BPM threshold",
    )

    print("\n2. POLICY DECISION INTEGRITY")

    check(len(decisions) == 39021, "39,021 policy decisions")

    check(
        set(decisions["policy"]) == POLICIES,
        "All three policies present",
    )

    check(
        not decisions.duplicated(keys + ["policy"]).any(),
        "Unique policy decisions per test window",
    )

    for policy in POLICIES:
        group = decisions[decisions["policy"] == policy]
        check(
            len(group) == 13007,
            f"{policy}: 13,007 decisions",
        )

    merged = decisions.merge(
        inputs[
            keys
            + ["unreliable_target", "unreliable_probability"]
        ],
        on=keys,
        how="left",
        validate="many_to_one",
        suffixes=("_decision", "_input"),
        indicator=True,
    )

    check(
        (merged["_merge"] == "both").all(),
        "Every decision corresponds to a test input",
    )

    check(
        np.array_equal(
            merged["unreliable_target_decision"].to_numpy(),
            merged["unreliable_target_input"].to_numpy(),
        ),
        "Policy decisions preserve reference labels",
    )

    check(
        np.allclose(
            merged["unreliable_probability_decision"].to_numpy(),
            merged["unreliable_probability_input"].to_numpy(),
            atol=1e-10,
            rtol=0,
        ),
        "Policy decisions preserve reliability probabilities",
    )

    check(
        decisions["investigations_used"].between(0, 2).all(),
        "Investigation budget never exceeds two",
    )

    check(
        (
            decisions.loc[
                decisions["policy"] == "No Investigation",
                "investigations_used",
            ] == 0
        ).all(),
        "No-investigation baseline uses zero investigations",
    )

    check(
        (
            decisions["correct"].to_numpy(dtype=int)
            == (
                decisions["predicted_unreliable"].to_numpy(dtype=int)
                == decisions["unreliable_target"].to_numpy(dtype=int)
            ).astype(int)
        ).all(),
        "Correctness flags match predictions",
    )

    print("\n3. INDEPENDENT METRIC RECOMPUTATION")

    check(
        set(comparison["policy"]) == POLICIES,
        "Comparison includes all three policies",
    )

    for policy in sorted(POLICIES):
        group = decisions[decisions["policy"] == policy]
        calculated = independently_calculate_metrics(group)
        reported = comparison[
            comparison["policy"] == policy
        ].iloc[0]

        for metric, value in calculated.items():
            check(
                np.isclose(
                    float(reported[metric]),
                    float(value),
                    atol=1e-9,
                    rtol=0,
                ),
                f"{policy}: {metric} matches",
            )

    print("\n4. FULL-AGENT ABLATION CONSISTENCY")

    full_agent = ablation[
        ablation["configuration"] == "Full Agent"
    ].sort_values(keys).reset_index(drop=True)

    learned = decisions[
        decisions["policy"] == "Learned Fitted Q"
    ].sort_values(keys).reset_index(drop=True)

    check(
        len(full_agent) == len(learned) == 13007,
        "Full-agent ablation has 13,007 decisions",
    )

    check(
        full_agent[keys].equals(learned[keys]),
        "Full-agent and learned-policy window IDs match",
    )

    for column in (
        "predicted_unreliable",
        "investigations_used",
        "unreliable_target",
    ):
        check(
            np.array_equal(
                full_agent[column].to_numpy(),
                learned[column].to_numpy(),
            ),
            f"Full-agent {column} matches",
        )

    check(
        np.allclose(
            full_agent["reward"].to_numpy(),
            learned["reward"].to_numpy(),
            atol=1e-10,
            rtol=0,
        ),
        "Full-agent rewards match",
    )

    print("\n" + "=" * 75)
    print("PASS: Step 38.2 integration verification completed.")
    print("=" * 75)


if __name__ == "__main__":
    main()

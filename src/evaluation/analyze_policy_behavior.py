"""Analyze validation errors and investigation behavior for MotionProof AI."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "results/metrics/validation_policy_decisions.csv"
)

OUTPUT_DIR = ROOT / "results/metrics"

POLICIES = [
    "No Investigation",
    "Fixed Rule",
    "Learned Fitted Q",
]

PROBABILITY_BINS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]


def load_data():

    data = pd.read_csv(INPUT_PATH)

    required = {
        "subject",
        "window_index",
        "policy",
        "unreliable_target",
        "predicted_unreliable",
        "unreliable_probability",
        "investigations_used",
        "reward",
        "correct",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    assert len(data) == 8475 * 3

    assert set(data["subject"]) == {11, 12}

    assert set(data["policy"]) == set(POLICIES)

    assert not data.duplicated(
        ["subject", "window_index", "policy"]
    ).any()

    assert data[
        "unreliable_probability"
    ].between(0, 1).all()

    assert data[
        "investigations_used"
    ].between(0, 2).all()

    assert set(
        data["unreliable_target"].unique()
    ).issubset({0, 1})

    assert set(
        data["predicted_unreliable"].unique()
    ).issubset({0, 1})

    assert np.array_equal(
        data["correct"].to_numpy(dtype=int),
        (
            data["unreliable_target"].to_numpy(dtype=int)
            == data["predicted_unreliable"].to_numpy(dtype=int)
        ).astype(int),
    )

    return data


def analyze_confusion_matrix(data):

    records = []

    for policy in POLICIES:

        subset = data[
            data["policy"] == policy
        ]

        y_true = subset[
            "unreliable_target"
        ].to_numpy()

        y_pred = subset[
            "predicted_unreliable"
        ].to_numpy()

        tp = int(
            ((y_true == 1) & (y_pred == 1)).sum()
        )

        tn = int(
            ((y_true == 0) & (y_pred == 0)).sum()
        )

        fp = int(
            ((y_true == 0) & (y_pred == 1)).sum()
        )

        fn = int(
            ((y_true == 1) & (y_pred == 0)).sum()
        )

        records.append({
            "policy": policy,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "total_errors": fp + fn,
            "accuracy": (tp + tn) / len(subset),
            "false_positive_rate": (
                fp / (fp + tn)
                if fp + tn else np.nan
            ),
            "false_negative_rate": (
                fn / (fn + tp)
                if fn + tp else np.nan
            ),
        })

    return pd.DataFrame(records)


def analyze_probability_bins(data):

    frame = data.copy()

    frame["probability_bin"] = pd.cut(
        frame["unreliable_probability"],
        bins=PROBABILITY_BINS,
        include_lowest=True,
    )

    records = []

    for (
        policy,
        probability_bin,
    ), group in frame.groupby(
        ["policy", "probability_bin"],
        observed=True,
    ):

        target = group[
            "unreliable_target"
        ].to_numpy()

        prediction = group[
            "predicted_unreliable"
        ].to_numpy()

        records.append({
            "policy": policy,
            "probability_bin": str(probability_bin),
            "windows": len(group),
            "actual_unreliable_rate": target.mean(),
            "predicted_unreliable_rate": prediction.mean(),
            "accuracy": (target == prediction).mean(),
            "false_positives": int(
                ((target == 0) & (prediction == 1)).sum()
            ),
            "false_negatives": int(
                ((target == 1) & (prediction == 0)).sum()
            ),
            "average_investigations": group[
                "investigations_used"
            ].mean(),
            "average_reward": group["reward"].mean(),
        })

    return pd.DataFrame(records)


def analyze_investigations(data):

    records = []

    for policy in POLICIES:

        subset = data[
            data["policy"] == policy
        ]

        for investigations in (0, 1, 2):

            group = subset[
                subset["investigations_used"]
                == investigations
            ]

            if len(group) == 0:
                continue

            records.append({
                "policy": policy,
                "investigations_used": investigations,
                "windows": len(group),
                "percentage_of_policy_windows": (
                    100 * len(group) / len(subset)
                ),
                "accuracy": group["correct"].mean(),
                "actual_unreliable_rate": group[
                    "unreliable_target"
                ].mean(),
                "flag_rate": group[
                    "predicted_unreliable"
                ].mean(),
                "average_reward": group["reward"].mean(),
            })

    return pd.DataFrame(records)


def analyze_pairwise_decisions(data):

    wide = data.pivot(
        index=["subject", "window_index"],
        columns="policy",
        values="predicted_unreliable",
    )

    targets = (
        data[data["policy"] == "No Investigation"]
        .set_index(["subject", "window_index"])[
            "unreliable_target"
        ]
    )

    assert len(wide) == 8475

    records = []

    learned = wide["Learned Fitted Q"]

    for comparison_policy in (
        "No Investigation",
        "Fixed Rule",
    ):

        baseline = wide[comparison_policy]

        learned_correct = learned == targets
        baseline_correct = baseline == targets

        both_correct = int(
            (learned_correct & baseline_correct).sum()
        )

        learned_only = int(
            (learned_correct & ~baseline_correct).sum()
        )

        baseline_only = int(
            (~learned_correct & baseline_correct).sum()
        )

        both_wrong = int(
            (~learned_correct & ~baseline_correct).sum()
        )

        records.append({
            "comparison": comparison_policy,
            "both_correct": both_correct,
            "learned_only_correct": learned_only,
            "comparison_only_correct": baseline_only,
            "both_wrong": both_wrong,
            "different_decisions": int(
                (learned != baseline).sum()
            ),
            "net_additional_correct": (
                learned_only - baseline_only
            ),
        })

    return pd.DataFrame(records)


def analyze_subjects(data):

    records = []

    for (subject, policy), group in data.groupby(
        ["subject", "policy"]
    ):

        target = group[
            "unreliable_target"
        ].to_numpy()

        prediction = group[
            "predicted_unreliable"
        ].to_numpy()

        records.append({
            "subject": subject,
            "policy": policy,
            "windows": len(group),
            "accuracy": group["correct"].mean(),
            "false_positives": int(
                ((target == 0) & (prediction == 1)).sum()
            ),
            "false_negatives": int(
                ((target == 1) & (prediction == 0)).sum()
            ),
            "average_investigations": group[
                "investigations_used"
            ].mean(),
            "average_reward": group["reward"].mean(),
        })

    return pd.DataFrame(records)


def main():

    print("=" * 72)
    print("MOTIONPROOF AI — POLICY BEHAVIOR AND ERROR ANALYSIS")
    print("=" * 72)

    data = load_data()

    print("\nPASS: Validation decisions loaded.")
    print("Total policy decisions:", len(data))
    print("Unique validation windows:", 8475)

    confusion = analyze_confusion_matrix(data)
    probability = analyze_probability_bins(data)
    investigations = analyze_investigations(data)
    pairwise = analyze_pairwise_decisions(data)
    subjects = analyze_subjects(data)

    reports = {
        "policy_error_analysis.csv": confusion,
        "policy_probability_analysis.csv": probability,
        "policy_investigation_analysis.csv": investigations,
        "policy_pairwise_analysis.csv": pairwise,
        "policy_subject_error_analysis.csv": subjects,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename, frame in reports.items():

        frame.to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

    print("\n" + "=" * 72)
    print("1. CONFUSION MATRIX AND ERRORS")
    print("=" * 72)

    print(
        confusion.round(4).to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("2. INVESTIGATION FREQUENCY")
    print("=" * 72)

    print(
        investigations.round(4).to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("3. LEARNED AGENT PROBABILITY ANALYSIS")
    print("=" * 72)

    learned_probability = probability[
        probability["policy"] == "Learned Fitted Q"
    ]

    print(
        learned_probability.round(4).to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("4. PAIRWISE POLICY COMPARISON")
    print("=" * 72)

    print(
        pairwise.to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("5. SUBJECT-LEVEL ERRORS")
    print("=" * 72)

    print(
        subjects.round(4).to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("SAVED REPORTS")
    print("=" * 72)

    for filename in reports:
        print(OUTPUT_DIR / filename)

    print(
        "\nPASS: Policy behavior and error analysis completed."
    )


if __name__ == "__main__":
    main()

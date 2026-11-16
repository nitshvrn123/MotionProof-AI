"""Audit frozen MotionProof AI final-test decisions."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = ROOT / "results/metrics/final_test"

DECISIONS_PATH = TEST_DIR / "test_policy_decisions.csv"

INPUT_PATH = TEST_DIR / "test_policy_inputs_nonlinear.csv"

OUTPUT_PATH = TEST_DIR / "test_policy_error_audit.csv"

POLICIES = (
    "No Investigation",
    "Fixed Rule",
    "Learned Fitted Q",
)

SUBJECTS = (13, 14, 15)


def analyze_group(group, subject, policy):

    y_true = group[
        "unreliable_target"
    ].to_numpy(dtype=int)

    y_pred = group[
        "predicted_unreliable"
    ].to_numpy(dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    total = len(group)

    return {
        "subject": subject,
        "policy": policy,
        "windows": total,
        "unreliable_prevalence": float(
            np.mean(y_true)
        ),
        "flag_rate": float(
            np.mean(y_pred)
        ),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "false_positive_rate": (
            fp / (fp + tn)
            if fp + tn else np.nan
        ),
        "false_negative_rate": (
            fn / (fn + tp)
            if fn + tp else np.nan
        ),
        "precision": (
            tp / (tp + fp)
            if tp + fp else 0.0
        ),
        "recall": (
            tp / (tp + fn)
            if tp + fn else 0.0
        ),
        "average_investigations": float(
            group["investigations_used"].mean()
        ),
        "average_reward": float(
            group["reward"].mean()
        ),
    }


def main():

    print("=" * 75)
    print("MOTIONPROOF AI — STEP 37.1")
    print("FINAL TEST ERROR AND DISTRIBUTION AUDIT")
    print("=" * 75)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing audit: {OUTPUT_PATH}"
        )

    decisions = pd.read_csv(
        DECISIONS_PATH
    )

    test_inputs = pd.read_csv(
        INPUT_PATH
    )

    assert len(test_inputs) == 13007
    assert len(decisions) == 13007 * 3

    assert set(decisions["subject"]) == set(SUBJECTS)
    assert set(decisions["policy"]) == set(POLICIES)

    assert not decisions.duplicated(
        ["subject", "window_index", "policy"]
    ).any()

    records = []

    for policy in POLICIES:

        policy_rows = decisions[
            decisions["policy"] == policy
        ]

        records.append(
            analyze_group(
                policy_rows,
                "Overall",
                policy,
            )
        )

        for subject in SUBJECTS:

            subject_rows = policy_rows[
                policy_rows["subject"] == subject
            ]

            records.append(
                analyze_group(
                    subject_rows,
                    f"S{subject}",
                    policy,
                )
            )

    audit = pd.DataFrame(records)

    print("\nOVERALL POLICY ERROR AUDIT")
    print("-" * 75)

    overall = audit[
        audit["subject"] == "Overall"
    ]

    print(
        overall.round(4).to_string(
            index=False
        )
    )

    print("\nPER-SUBJECT ERROR AUDIT")
    print("-" * 75)

    per_subject = audit[
        audit["subject"] != "Overall"
    ]

    print(
        per_subject.round(4).to_string(
            index=False
        )
    )

    audit.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSaved:", OUTPUT_PATH)

    print("\nPASS: Step 37.1 completed.")


if __name__ == "__main__":
    main()

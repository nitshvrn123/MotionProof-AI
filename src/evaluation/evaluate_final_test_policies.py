"""Evaluate frozen MotionProof AI policies on untouched test subjects."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import evaluate_validation_policies as evaluator
from src.agents.fitted_q_inference import load_fitted_q_policy


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "results/metrics/final_test/test_policy_inputs_nonlinear.csv"
)

OUTPUT_DIRECTORY = (
    ROOT / "results/metrics/final_test"
)

CHECKPOINT_PATH = (
    ROOT / "checkpoints/learned_policy_fitted_q_full.pt"
)

SUBJECTS = (13, 14, 15)

EXPECTED_WINDOWS = {
    13: 4565,
    14: 4476,
    15: 3966,
}

EXPECTED_POLICIES = {
    "No Investigation",
    "Fixed Rule",
    "Learned Fitted Q",
}


def evaluate_test_subject(subject, test_inputs, learned_policy):
    """Run existing frozen policies on one test subject."""

    subject_path = (
        ROOT / f"data/processed/test/S{subject}.npz"
    )

    with np.load(subject_path) as data:
        ppg = data["ppg"]
        acc = data["acc"]

    rows = (
        test_inputs.loc[
            test_inputs["subject"] == subject
        ]
        .sort_values("window_index")
        .reset_index(drop=True)
    )

    expected = EXPECTED_WINDOWS[subject]

    assert len(rows) == expected
    assert ppg.shape == (expected, 512)
    assert acc.shape == (expected, 256, 3)

    assert np.array_equal(
        rows["window_index"].to_numpy(),
        np.arange(expected),
    )

    predictions = rows[
        "cnn_predicted_hr"
    ].to_numpy(dtype=np.float64)

    records = []

    for index, row in rows.iterrows():

        history = predictions[
            max(
                0,
                index - evaluator.HISTORY_LENGTH,
            ):index
        ]

        baseline_prediction = int(
            row["unreliable_probability"] >= 0.5
        )

        # Existing fixed-rule implementation.
        fixed_environment = evaluator.make_environment(
            ppg[index],
            acc[index],
            row,
            history,
        )

        fixed_prediction, fixed_investigations = (
            evaluator.run_fixed(fixed_environment)
        )

        # Existing learned fitted Q implementation.
        learned_environment = evaluator.make_environment(
            ppg[index],
            acc[index],
            row,
            history,
        )

        learned_prediction, learned_investigations = (
            evaluator.run_learned(
                learned_environment,
                learned_policy,
            )
        )

        target = int(row["unreliable_target"])

        policy_outputs = {
            "No Investigation": (
                baseline_prediction,
                0,
            ),
            "Fixed Rule": (
                fixed_prediction,
                fixed_investigations,
            ),
            "Learned Fitted Q": (
                learned_prediction,
                learned_investigations,
            ),
        }

        for policy_name, (
            prediction,
            investigations,
        ) in policy_outputs.items():

            assert prediction in (0, 1)

            assert (
                0 <= investigations <= evaluator.BUDGET
            )

            decision = (
                "FLAG_UNRELIABLE"
                if prediction == 1
                else "ACCEPT"
            )

            reward_result = (
                evaluator.calculate_investigation_reward(
                    final_decision=decision,
                    unreliable_target=target,
                    investigations_used=investigations,
                )
            )

            records.append({
                "subject": subject,
                "window_index": index,
                "policy": policy_name,
                "unreliable_target": target,
                "predicted_unreliable": prediction,
                "unreliable_probability": float(
                    row["unreliable_probability"]
                ),
                "investigations_used": investigations,
                "reward": float(
                    reward_result["total_reward"]
                ),
                "correct": int(
                    prediction == target
                ),
            })

        if (index + 1) % 1000 == 0:
            print(
                f"S{subject}: "
                f"{index + 1:,}/{expected:,} "
                "windows evaluated.",
                flush=True,
            )

    result = pd.DataFrame(records)

    assert len(result) == expected * 3

    return result


def main():

    print("=" * 75)
    print("MOTIONPROOF AI — STEP 36.5B")
    print("FINAL FROZEN POLICY TEST EVALUATION")
    print("=" * 75)

    output_paths = {
        "decisions": (
            OUTPUT_DIRECTORY
            / "test_policy_decisions.csv"
        ),
        "comparison": (
            OUTPUT_DIRECTORY
            / "test_policy_comparison.csv"
        ),
        "per_subject": (
            OUTPUT_DIRECTORY
            / "test_policy_per_subject.csv"
        ),
    }

    for path in output_paths.values():
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite final test results: {path}"
            )

    test_inputs = pd.read_csv(INPUT_PATH)

    assert len(test_inputs) == 13007

    assert set(test_inputs["subject"]) == {
        13, 14, 15
    }

    assert not test_inputs.duplicated(
        ["subject", "window_index"]
    ).any()

    assert test_inputs[
        "unreliable_probability"
    ].between(0, 1).all()

    print("\nPASS: Frozen test inputs verified.")

    learned_policy = load_fitted_q_policy(
        CHECKPOINT_PATH,
        device="cpu",
    )

    print("PASS: Frozen fitted Q policy loaded.")

    results = []

    for subject in SUBJECTS:

        print(
            f"\nEvaluating S{subject}...",
            flush=True,
        )

        subject_results = evaluate_test_subject(
            subject,
            test_inputs,
            learned_policy,
        )

        results.append(subject_results)

        print(
            f"PASS: S{subject} completed.",
            flush=True,
        )

    decisions = pd.concat(
        results,
        ignore_index=True,
    )

    assert len(decisions) == 13007 * 3

    assert set(decisions["policy"]) == (
        EXPECTED_POLICIES
    )

    assert not decisions.duplicated(
        ["subject", "window_index", "policy"]
    ).any()

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    overall_records = []

    for policy_name, group in decisions.groupby(
        "policy"
    ):

        metrics = evaluator.calculate_metrics(
            group
        )

        metrics["policy"] = policy_name

        overall_records.append(metrics)

    overall = pd.DataFrame(
        overall_records
    )

    # --------------------------------------------------------
    # Per-subject metrics
    # --------------------------------------------------------

    subject_records = []

    for (
        subject,
        policy_name,
    ), group in decisions.groupby(
        ["subject", "policy"]
    ):

        metrics = evaluator.calculate_metrics(
            group
        )

        metrics["subject"] = subject
        metrics["policy"] = policy_name

        subject_records.append(metrics)

    per_subject = pd.DataFrame(
        subject_records
    )

    # --------------------------------------------------------
    # Save final test results
    # --------------------------------------------------------

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    decisions.to_csv(
        output_paths["decisions"],
        index=False,
    )

    overall.to_csv(
        output_paths["comparison"],
        index=False,
    )

    per_subject.to_csv(
        output_paths["per_subject"],
        index=False,
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    display_columns = [
        "policy",
        "windows",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "average_investigations",
        "average_reward",
    ]

    print("\n" + "=" * 75)
    print("OVERALL FINAL TEST COMPARISON")
    print("=" * 75)

    print(
        overall[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("PER-SUBJECT FINAL TEST COMPARISON")
    print("=" * 75)

    print(
        per_subject[
            ["subject"] + display_columns
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("SAVED FINAL TEST FILES")
    print("=" * 75)

    for path in output_paths.values():
        print(path)

    print("\nPASS: Step 36.5B completed.")


if __name__ == "__main__":
    main()

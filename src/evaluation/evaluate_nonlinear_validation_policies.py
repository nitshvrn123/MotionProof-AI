"""Evaluate existing policies using nonlinear reliability probabilities."""

from pathlib import Path

import pandas as pd

import src.evaluation.evaluate_validation_policies as evaluator


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "results/metrics/validation_policy_inputs_nonlinear.csv"
)

OUTPUT_DIRECTORY = (
    ROOT / "results/metrics/nonlinear_validation"
)


def main():
    print("=" * 72)
    print("MOTIONPROOF AI — STEP 33.8")
    print("NONLINEAR RELIABILITY POLICY EVALUATION")
    print("=" * 72)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    validation = pd.read_csv(INPUT_PATH)

    assert len(validation) == 8475
    assert set(validation["subject"]) == {11, 12}
    assert not validation.duplicated(
        ["subject", "window_index"]
    ).any()

    assert validation[
        "unreliable_probability"
    ].between(0, 1).all()

    print("\nPASS: Nonlinear validation inputs verified.")

    if OUTPUT_DIRECTORY.exists() and any(OUTPUT_DIRECTORY.iterdir()):
        raise FileExistsError(
            "Nonlinear evaluation output directory is not empty. "
            "Refusing to overwrite existing results."
        )

    evaluator.INPUT_PATH = INPUT_PATH
    evaluator.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY

    print("\nRunning original evaluator with nonlinear probabilities...")
    evaluator.main()

    expected_files = [
        "validation_policy_decisions.csv",
        "validation_policy_comparison.csv",
        "validation_policy_per_subject.csv",
    ]

    for filename in expected_files:
        path = OUTPUT_DIRECTORY / filename
        assert path.exists(), f"Missing output: {path}"

    overall = pd.read_csv(
        OUTPUT_DIRECTORY / "validation_policy_comparison.csv"
    )

    assert set(overall["policy"]) == {
        "No Investigation",
        "Fixed Rule",
        "Learned Fitted Q",
    }

    print("\n" + "=" * 72)
    print("NONLINEAR EVALUATION SUMMARY")
    print("=" * 72)

    columns = [
        "policy",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "average_investigations",
        "average_reward",
    ]

    print(
        overall[columns].round(4).to_string(index=False)
    )

    print("\nSaved results:", OUTPUT_DIRECTORY)
    print("\nPASS: Step 33.8 completed.")


if __name__ == "__main__":
    main()

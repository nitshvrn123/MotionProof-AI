from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.evaluation.reliability_baseline import FEATURE_COLUMNS
from src.evaluation.reliability_features import extract_reliability_features

ROOT = Path(__file__).resolve().parents[2]

ORIGINAL = ROOT / "results/metrics/validation_policy_inputs.csv"
NONLINEAR = ROOT / "results/metrics/validation_nonlinear_probabilities.csv"
OUTPUT = ROOT / "results/metrics/validation_policy_inputs_nonlinear.csv"

LOGISTIC_MODEL = ROOT / "checkpoints/reliability_classifier_full.joblib"


def main():
    print("=" * 70)
    print("MOTIONPROOF AI — STEP 33.6B")
    print("NONLINEAR POLICY INPUT PREPARATION")
    print("=" * 70)

    original = pd.read_csv(ORIGINAL)
    nonlinear = pd.read_csv(NONLINEAR)

    assert len(original) == len(nonlinear) == 8475

    keys = ["subject", "window_index"]

    assert not original.duplicated(keys).any()
    assert not nonlinear.duplicated(keys).any()

    merged = original.merge(
        nonlinear[
            keys + ["nonlinear_probability"]
        ],
        on=keys,
        validate="one_to_one",
    )

    assert len(merged) == 8475

    # Reconstruct features independently to check the
    # feature-generation path used by the nonlinear CSV.
    feature_parts = []

    for subject in [11, 12]:
        subject_rows = (
            merged[merged["subject"] == subject]
            .sort_values("window_index")
        )

        indices = subject_rows["window_index"].to_numpy(dtype=int)

        with np.load(
            ROOT / f"data/processed/val/S{subject}.npz"
        ) as data:
            features = extract_reliability_features(
                data["ppg"][indices],
                data["acc"][indices],
            )

        feature_df = pd.DataFrame(
            features,
            columns=FEATURE_COLUMNS[:-1],
        )

        feature_df["cnn_predicted_hr"] = (
            subject_rows["cnn_predicted_hr"].to_numpy()
        )

        feature_df["subject"] = subject
        feature_df["window_index"] = indices

        feature_parts.append(feature_df)

    feature_frame = pd.concat(
        feature_parts,
        ignore_index=True,
    )

    # Check that reconstructed features reproduce the
    # saved logistic-regression validation probabilities.
    logistic = joblib.load(LOGISTIC_MODEL)

    reproduced = logistic.predict_proba(
        feature_frame[FEATURE_COLUMNS]
    )[:, 1]

    check = feature_frame[keys].copy()
    check["reproduced_logistic"] = reproduced

    check = original.merge(
        check,
        on=keys,
        validate="one_to_one",
    )

    differences = np.abs(
        check["unreliable_probability"].to_numpy()
        - check["reproduced_logistic"].to_numpy()
    )

    print("\nLOGISTIC REPRODUCTION CHECK")
    print("Maximum difference:", differences.max())
    print("Mean difference:", differences.mean())

    if not np.allclose(
        check["unreliable_probability"],
        check["reproduced_logistic"],
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError(
            "Feature consistency check failed. "
            "Do not use nonlinear policy inputs yet."
        )

    print("PASS: Feature pipeline reproduces logistic probabilities.")

    prepared = original.copy()

    probability_lookup = nonlinear.set_index(keys)[
        "nonlinear_probability"
    ]

    prepared["unreliable_probability"] = [
        probability_lookup.loc[(row.subject, row.window_index)]
        for row in original.itertuples(index=False)
    ]

    assert prepared["unreliable_probability"].between(0, 1).all()

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing file: {OUTPUT}"
        )

    prepared.to_csv(OUTPUT, index=False)

    print("\nSaved:", OUTPUT)
    print("Rows:", len(prepared))
    print("\nPASS: Step 33.6B completed.")


if __name__ == "__main__":
    main()

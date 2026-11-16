from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.evaluation.reliability_baseline import FEATURE_COLUMNS
from src.evaluation.reliability_nonlinear import build_nonlinear_classifier
from src.evaluation.reliability_features import extract_reliability_features


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "results/metrics/reliability_training_dataset.csv"
VALIDATION_PATH = ROOT / "results/metrics/validation_policy_inputs.csv"
OUTPUT_PATH = ROOT / "results/metrics/validation_nonlinear_probabilities.csv"
MODEL_PATH = ROOT / "checkpoints/reliability_classifier_nonlinear_full.joblib"


def main():
    print("=" * 70)
    print("MOTIONPROOF AI — STEP 33.4")
    print("=" * 70)

    training = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    assert set(training["subject"].unique()) == set(range(1, 11))
    assert set(validation["subject"].unique()) == {11, 12}

    assert not OUTPUT_PATH.exists(), (
        f"Output already exists; refusing to overwrite: {OUTPUT_PATH}"
    )
    assert not MODEL_PATH.exists(), (
        f"Model already exists; refusing to overwrite: {MODEL_PATH}"
    )

    print("\nTraining histogram gradient boosting...")

    model = build_nonlinear_classifier()
    model.fit(
        training[FEATURE_COLUMNS],
        training["unreliable_target"],
    )

    print("Training completed.")

    rows = []

    for subject in [11, 12]:
        path = ROOT / f"data/processed/val/S{subject}.npz"

        print(f"\nProcessing S{subject}...")

        with np.load(path) as data:
            print("Available keys:", list(data.files))

            ppg = data["ppg"]
            acc = data["acc"]

            subject_validation = (
                validation[validation["subject"] == subject]
                .sort_values("window_index")
                .copy()
            )

            assert len(ppg) == len(acc) == len(subject_validation)
            assert subject_validation["window_index"].is_unique
            assert subject_validation["window_index"].min() >= 0
            assert subject_validation["window_index"].max() < len(ppg)

            indices = subject_validation["window_index"].to_numpy(dtype=int)

            features = extract_reliability_features(
                ppg[indices],
                acc[indices],
            )

            if isinstance(features, pd.DataFrame):
                feature_df = features.copy()

                if list(feature_df.columns) != list(FEATURE_COLUMNS[:-1]):
                    if feature_df.shape[1] == len(FEATURE_COLUMNS) - 1:
                        feature_df.columns = FEATURE_COLUMNS[:-1]
                    else:
                        raise ValueError(
                            f"Unexpected feature columns: {feature_df.columns.tolist()}"
                        )

            else:
                features = np.asarray(features)

                expected_shape = (
                    len(subject_validation),
                    len(FEATURE_COLUMNS) - 1,
                )

                if features.shape != expected_shape:
                    raise ValueError(
                        f"Expected feature shape {expected_shape}, "
                        f"got {features.shape}"
                    )

                feature_df = pd.DataFrame(
                    features,
                    columns=FEATURE_COLUMNS[:-1],
                )

            assert len(feature_df) == len(subject_validation)

            feature_df["subject"] = subject
            feature_df["window_index"] = indices
            feature_df["cnn_predicted_hr"] = (
                subject_validation["cnn_predicted_hr"].to_numpy()
            )

            rows.extend(feature_df.to_dict(orient="records"))

    feature_frame = pd.DataFrame(rows)

    assert len(feature_frame) == 8475

    probabilities = model.predict_proba(
        feature_frame[FEATURE_COLUMNS]
    )[:, 1]

    output = feature_frame[
        ["subject", "window_index"]
    ].copy()

    output["nonlinear_probability"] = probabilities

    output = output.merge(
        validation,
        on=["subject", "window_index"],
        validate="one_to_one",
    )

    assert len(output) == len(validation)
    assert output["nonlinear_probability"].between(0, 1).all()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(OUTPUT_PATH, index=False)
    joblib.dump(model, MODEL_PATH)

    print("\nRESULTS")
    print("-" * 70)
    print("Validation windows:", len(output))
    print("Mean nonlinear probability:", probabilities.mean())
    print("Minimum probability:", probabilities.min())
    print("Maximum probability:", probabilities.max())

    print("\nSaved predictions:", OUTPUT_PATH)
    print("Saved classifier:", MODEL_PATH)

    print("\nPASS: Step 33.4 completed.")


if __name__ == "__main__":
    main()

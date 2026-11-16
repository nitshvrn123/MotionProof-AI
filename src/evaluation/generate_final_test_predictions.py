"""Generate frozen-model predictions for untouched S13-S15 test subjects."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.evaluation.generate_validation_predictions import (
    load_cnn,
    predict_cnn,
)
from src.evaluation.reliability_features import (
    extract_reliability_features,
    FEATURE_NAMES,
)
from src.evaluation.reliability_baseline import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[2]

TEST_DIRECTORY = ROOT / "data/processed/test"

CLASSIFIER_PATH = (
    ROOT / "checkpoints/reliability_classifier_nonlinear_full.joblib"
)

OUTPUT_DIRECTORY = ROOT / "results/metrics/final_test"

OUTPUT_PATH = OUTPUT_DIRECTORY / "test_policy_inputs_nonlinear.csv"

SUBJECTS = (13, 14, 15)

EXPECTED_WINDOWS = {
    13: 4565,
    14: 4476,
    15: 3966,
}


def generate_subject(subject, cnn, mean, std, classifier, device):

    subject_path = TEST_DIRECTORY / f"S{subject}.npz"

    with np.load(subject_path) as data:
        ppg = data["ppg"].copy()
        acc = data["acc"].copy()
        reference_hr = data["hr"].copy()

    expected = EXPECTED_WINDOWS[subject]

    assert ppg.shape == (expected, 512)
    assert acc.shape == (expected, 256, 3)
    assert reference_hr.shape == (expected,)

    assert np.isfinite(ppg).all()
    assert np.isfinite(acc).all()
    assert np.isfinite(reference_hr).all()

    # CNN inference uses only PPG.
    predicted_hr = predict_cnn(
        model=cnn,
        ppg=ppg,
        mean=mean,
        std=std,
        device=device,
    )

    predicted_hr = np.asarray(
        predicted_hr,
        dtype=np.float64,
    ).reshape(-1)

    assert predicted_hr.shape == (expected,)

    # Reliability inference uses signal features and CNN predictions.
    signal_features = extract_reliability_features(
        ppg,
        acc,
    )

    assert signal_features.shape == (
        expected,
        len(FEATURE_NAMES),
    )

    features = pd.DataFrame(
        signal_features,
        columns=FEATURE_NAMES,
    )

    features["cnn_predicted_hr"] = predicted_hr

    X_test = features.loc[:, FEATURE_COLUMNS].copy()

    expected_columns = list(classifier.feature_names_in_)

    assert list(X_test.columns) == expected_columns, (
        "Reliability classifier feature ordering mismatch."
    )

    assert np.isfinite(
        X_test.to_numpy(dtype=np.float64)
    ).all()

    probabilities = classifier.predict_proba(
        X_test
    )[:, 1]

    assert probabilities.shape == (expected,)
    assert np.isfinite(probabilities).all()
    assert np.all(
        (probabilities >= 0.0)
        & (probabilities <= 1.0)
    )

    # Reference HR is used ONLY for evaluation.
    absolute_error = np.abs(
        predicted_hr - reference_hr
    )

    unreliable_target = (
        absolute_error > 10.0
    ).astype(np.int8)

    result = pd.DataFrame({
        "subject": subject,
        "window_index": np.arange(expected),
        "cnn_predicted_hr": predicted_hr,
        "unreliable_probability": probabilities,
        "reference_hr": reference_hr,
        "absolute_error": absolute_error,
        "unreliable_target": unreliable_target,
    })

    mae = float(np.mean(absolute_error))
    rmse = float(np.sqrt(np.mean(absolute_error ** 2)))

    print(
        f"S{subject}: {expected:,} windows | "
        f"MAE: {mae:.4f} BPM | "
        f"RMSE: {rmse:.4f} BPM | "
        f"Unreliable rate: {unreliable_target.mean():.4f}"
    )

    return result


def main():

    print("=" * 75)
    print("MOTIONPROOF AI — STEP 36.4")
    print("FINAL UNTOUCHED TEST PREDICTION GENERATION")
    print("=" * 75)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing test results: {OUTPUT_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    cnn, mean, std = load_cnn(device)

    print("PASS: Frozen CNN loaded.")
    print("Normalization mean:", mean)
    print("Normalization std:", std)

    classifier = joblib.load(CLASSIFIER_PATH)

    assert type(classifier).__name__ == (
        "HistGradientBoostingClassifier"
    )

    print("PASS: Frozen nonlinear classifier loaded.")

    results = []

    for subject in SUBJECTS:

        print(f"\nEvaluating S{subject}...")

        result = generate_subject(
            subject=subject,
            cnn=cnn,
            mean=mean,
            std=std,
            classifier=classifier,
            device=device,
        )

        results.append(result)

    combined = pd.concat(
        results,
        ignore_index=True,
    )

    assert len(combined) == 13007

    assert set(combined["subject"]) == {
        13, 14, 15
    }

    assert not combined.duplicated(
        ["subject", "window_index"]
    ).any()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 75)
    print("OVERALL TEST SUMMARY")
    print("=" * 75)

    errors = combined["absolute_error"].to_numpy()

    print("Total windows:", len(combined))
    print("CNN MAE:", round(float(np.mean(errors)), 4), "BPM")
    print(
        "CNN RMSE:",
        round(float(np.sqrt(np.mean(errors ** 2))), 4),
        "BPM",
    )
    print(
        "Unreliable prediction rate:",
        round(float(combined["unreliable_target"].mean()), 4),
    )
    print(
        "Mean unreliable probability:",
        round(float(combined["unreliable_probability"].mean()), 4),
    )

    print("\nSaved:", OUTPUT_PATH)
    print("\nPASS: Step 36.4 completed.")


if __name__ == "__main__":
    main()

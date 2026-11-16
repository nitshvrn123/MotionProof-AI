"""Generate leakage-free CNN and reliability predictions for S11 and S12."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.models.ppg_cnn import PPGHeartRateCNN
from src.evaluation.reliability_features import (
    FEATURE_NAMES,
    extract_reliability_features,
)
from src.evaluation.reliability_baseline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_classifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_DATASET = (
    PROJECT_ROOT
    / "results/metrics/reliability_training_dataset.csv"
)

CNN_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints/ppg_cnn_best.pt"
)

CLASSIFIER_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints/reliability_classifier_full.joblib"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results/metrics/validation_policy_inputs.csv"
)

BATCH_SIZE = 512


def load_cnn(device):
    """Load the original CNN with its saved training normalization."""

    checkpoint = torch.load(
        CNN_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model = PPGHeartRateCNN()

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    mean = float(
        checkpoint["normalization_mean"]
    )

    std = float(
        checkpoint["normalization_std"]
    )

    if not np.isfinite(mean):
        raise ValueError("Invalid CNN normalization mean.")

    if not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid CNN normalization std.")

    return model, mean, std


def predict_cnn(model, ppg, mean, std, device):
    """Predict HR without exposing reference labels."""

    predictions = []

    with torch.no_grad():

        for start in range(0, len(ppg), BATCH_SIZE):

            batch = ppg[
                start:start + BATCH_SIZE
            ]

            normalized = (
                batch.astype(np.float32) - mean
            ) / std

            tensor = torch.from_numpy(
                normalized
            ).to(device)

            output = model(tensor)

            predictions.append(
                output.cpu().numpy()
            )

    predictions = np.concatenate(
        predictions
    ).astype(np.float32)

    if not np.isfinite(predictions).all():
        raise ValueError("CNN produced nonfinite predictions.")

    return predictions


def fit_reliability_classifier():
    """Fit the original logistic classifier using training subjects only."""

    training = pd.read_csv(
        TRAINING_DATASET
    )

    required = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN, "subject", "window_index"]
    )

    missing = set(required) - set(training.columns)

    if missing:
        raise ValueError(
            f"Missing training columns: {sorted(missing)}"
        )

    if set(training["subject"].unique()) != set(range(1, 11)):
        raise ValueError(
            "Reliability training must use only S1-S10."
        )

    if training.duplicated(
        ["subject", "window_index"]
    ).any():
        raise ValueError(
            "Duplicate reliability training windows."
        )

    X_train = training[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float64)

    y_train = training[
        TARGET_COLUMN
    ].to_numpy(dtype=np.int8)

    if not np.isfinite(X_train).all():
        raise ValueError(
            "Nonfinite reliability training features."
        )

    classifier = build_classifier()

    classifier.fit(
        X_train,
        y_train,
    )

    CLASSIFIER_CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        classifier,
        CLASSIFIER_CHECKPOINT,
    )

    print(
        "PASS: Reliability classifier fitted on S1-S10."
    )

    return classifier


def generate_subject_predictions(
    subject,
    cnn,
    cnn_mean,
    cnn_std,
    classifier,
    device,
):
    """Generate predictions for one unseen validation subject."""

    subject_path = (
        PROJECT_ROOT
        / f"data/processed/val/S{subject}.npz"
    )

    with np.load(subject_path) as data:

        ppg = data["ppg"].copy()
        acc = data["acc"].copy()
        reference_hr = data["hr"].copy()

    if ppg.ndim != 2 or ppg.shape[1] != 512:
        raise ValueError("Invalid PPG shape.")

    if acc.shape != (len(ppg), 256, 3):
        raise ValueError("Invalid ACC shape.")

    if reference_hr.shape != (len(ppg),):
        raise ValueError("Invalid reference HR shape.")

    # --------------------------------------------------------
    # CNN prediction
    # --------------------------------------------------------

    predicted_hr = predict_cnn(
        model=cnn,
        ppg=ppg,
        mean=cnn_mean,
        std=cnn_std,
        device=device,
    )

    # --------------------------------------------------------
    # Reliability features
    # --------------------------------------------------------

    signal_features = extract_reliability_features(
        ppg,
        acc,
    )

    features = pd.DataFrame(
        signal_features,
        columns=FEATURE_NAMES,
    )

    features["cnn_predicted_hr"] = predicted_hr

    X_validation = features[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float64)

    probabilities = classifier.predict_proba(
        X_validation
    )[:, 1]

    # --------------------------------------------------------
    # Evaluation labels: never passed to the classifier
    # --------------------------------------------------------

    absolute_error = np.abs(
        predicted_hr - reference_hr
    )

    unreliable_target = (
        absolute_error > 10.0
    ).astype(np.int8)

    result = pd.DataFrame({
        "subject": subject,
        "window_index": np.arange(len(ppg)),
        "cnn_predicted_hr": predicted_hr,
        "unreliable_probability": probabilities,
        "reference_hr": reference_hr,
        "absolute_error": absolute_error,
        "unreliable_target": unreliable_target,
    })

    assert len(result) == len(ppg)

    assert result[
        "unreliable_probability"
    ].between(0, 1).all()

    mae = float(
        np.mean(absolute_error)
    )

    print(
        f"S{subject}: {len(result):,} windows | "
        f"CNN MAE: {mae:.4f} BPM | "
        f"Unreliable rate: {unreliable_target.mean():.4f}"
    )

    return result


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("MOTIONPROOF AI — VALIDATION PREDICTIONS")
    print("=" * 65)

    print("Device:", device)

    cnn, mean, std = load_cnn(
        device
    )

    print("PASS: CNN loaded.")

    classifier = fit_reliability_classifier()

    results = []

    for subject in (11, 12):

        frame = generate_subject_predictions(
            subject=subject,
            cnn=cnn,
            cnn_mean=mean,
            cnn_std=std,
            classifier=classifier,
            device=device,
        )

        results.append(frame)

    validation = pd.concat(
        results,
        ignore_index=True,
    )

    assert len(validation) == 8475

    assert set(validation["subject"]) == {11, 12}

    assert not validation.duplicated(
        ["subject", "window_index"]
    ).any()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 65)
    print("VALIDATION DATASET SUMMARY")
    print("=" * 65)

    print("Total windows:", len(validation))
    print("Subjects:", sorted(validation["subject"].unique()))
    print("Saved:", OUTPUT_PATH)
    print("Classifier:", CLASSIFIER_CHECKPOINT)

    print(
        "\nPASS: Validation predictions generated."
    )


if __name__ == "__main__":
    main()

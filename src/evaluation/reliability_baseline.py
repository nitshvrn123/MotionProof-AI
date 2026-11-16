"""Subject-grouped logistic regression reliability baseline."""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)

from src.evaluation.reliability_features import FEATURE_NAMES


FEATURE_COLUMNS = list(FEATURE_NAMES) + [
    "cnn_predicted_hr"
]

TARGET_COLUMN = "unreliable_target"

FOLD_SUBJECTS = {
    1: (1, 2),
    2: (3, 4),
    3: (5, 6),
    4: (7, 8),
    5: (9, 10),
}


def build_classifier():
    """Create an unfitted reliability classifier."""

    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                C=1.0,
            ),
        ),
    ])


def evaluate_reliability_baseline(data):
    """Evaluate reliability prediction using grouped folds."""

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN, "subject", "fold", "window_index"]
    )

    missing = set(required_columns) - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if data.duplicated(["subject", "window_index"]).any():
        raise ValueError("Duplicate subject-window rows detected.")

    if len(data) != 43215:
        raise ValueError("Unexpected training dataset size.")

    if set(data["subject"].unique()) != set(range(1, 11)):
        raise ValueError("Unexpected subject coverage.")

    if not np.isfinite(
        data[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Nonfinite classifier features detected.")

    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Reliability targets must be binary.")

    fold_results = []
    prediction_frames = []

    for fold_id, held_out_subjects in FOLD_SUBJECTS.items():

        train_mask = ~data["subject"].isin(
            held_out_subjects
        )

        test_mask = data["subject"].isin(
            held_out_subjects
        )

        train_data = data.loc[train_mask]
        test_data = data.loc[test_mask]

        assert set(train_data["subject"]).isdisjoint(
            set(test_data["subject"])
        )

        assert set(test_data["subject"]) == set(
            held_out_subjects
        )

        assert set(test_data["fold"]) == {fold_id}

        X_train = train_data[FEATURE_COLUMNS].to_numpy()
        y_train = train_data[TARGET_COLUMN].to_numpy()

        X_test = test_data[FEATURE_COLUMNS].to_numpy()
        y_test = test_data[TARGET_COLUMN].to_numpy()

        model = build_classifier()

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(np.int8)

        results = {
            "fold": fold_id,
            "subjects": ",".join(
                f"S{s}" for s in held_out_subjects
            ),
            "windows": len(test_data),
            "unreliable_rate": float(y_test.mean()),
            "auroc": roc_auc_score(
                y_test, probabilities
            ),
            "auprc": average_precision_score(
                y_test, probabilities
            ),
            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "brier": brier_score_loss(
                y_test,
                probabilities,
            ),
        }

        fold_results.append(results)

        prediction_frame = test_data[
            ["subject", "window_index", "fold", TARGET_COLUMN]
        ].copy()

        prediction_frame["unreliable_probability"] = (
            probabilities
        )

        prediction_frame["predicted_unreliable"] = (
            predictions
        )

        prediction_frames.append(
            prediction_frame
        )

    metrics = pd.DataFrame(fold_results)

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    assert len(predictions) == len(data)

    assert not predictions.duplicated(
        ["subject", "window_index"]
    ).any()

    assert predictions["unreliable_probability"].between(
        0, 1
    ).all()

    return metrics, predictions

"""Nonlinear reliability classifier with subject-grouped evaluation."""

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)

from src.evaluation.reliability_baseline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FOLD_SUBJECTS,
)


def build_nonlinear_classifier():
    """Create a deterministic nonlinear classifier."""

    return HistGradientBoostingClassifier(
        max_iter=150,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=42,
    )


def evaluate_nonlinear_classifier(data):
    """Evaluate nonlinear reliability prediction across subject folds."""

    required = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN, "subject", "fold", "window_index"]
    )

    missing = set(required) - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if len(data) != 43215:
        raise ValueError("Unexpected dataset size.")

    if data.duplicated(
        ["subject", "window_index"]
    ).any():
        raise ValueError("Duplicate windows detected.")

    if not np.isfinite(
        data[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Nonfinite features detected.")

    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Invalid binary targets.")

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

        X_train = train_data[
            FEATURE_COLUMNS
        ].to_numpy(dtype=np.float64)

        y_train = train_data[
            TARGET_COLUMN
        ].to_numpy()

        X_test = test_data[
            FEATURE_COLUMNS
        ].to_numpy(dtype=np.float64)

        y_test = test_data[
            TARGET_COLUMN
        ].to_numpy()

        model = build_nonlinear_classifier()

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(np.int8)

        fold_results.append({
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
                y_test, predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test, predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_test, predictions,
                zero_division=0,
            ),
            "brier": brier_score_loss(
                y_test, probabilities
            ),
        })

        prediction_frame = test_data[
            [
                "subject",
                "window_index",
                "fold",
                TARGET_COLUMN,
            ]
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

    assert predictions[
        "unreliable_probability"
    ].between(0, 1).all()

    return metrics, predictions

"""Build aligned training data for MotionProof AI investigation policies."""

from pathlib import Path

import numpy as np
import pandas as pd


OBSERVATION_COLUMNS = (
    "cnn_predicted_hr",
    "unreliable_probability",
)

IDENTIFIER_COLUMNS = (
    "subject",
    "window_index",
    "fold",
)

TRAINING_LABEL_COLUMNS = (
    "reference_hr",
    "absolute_error",
    "unreliable_target",
)


def build_policy_training_data(
    reliability_path,
    probability_path,
):
    """Merge CNN predictions with OOF reliability probabilities."""

    reliability = pd.read_csv(reliability_path)
    probabilities = pd.read_csv(probability_path)

    keys = [
        "subject",
        "window_index",
        "fold",
        "unreliable_target",
    ]

    required_reliability = set(keys) | {
        "predicted_hr",
        "reference_hr",
        "absolute_error",
    }

    required_probabilities = set(keys) | {
        "unreliable_probability",
    }

    if not required_reliability.issubset(reliability.columns):
        raise ValueError("Reliability dataset is missing columns.")

    if not required_probabilities.issubset(probabilities.columns):
        raise ValueError("Probability dataset is missing columns.")

    if reliability.duplicated(
        ["subject", "window_index"]
    ).any():
        raise ValueError("Duplicate reliability windows.")

    if probabilities.duplicated(
        ["subject", "window_index"]
    ).any():
        raise ValueError("Duplicate probability windows.")

    merged = reliability.merge(
        probabilities[
            keys + ["unreliable_probability"]
        ],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not (merged["_merge"] == "both").all():
        raise ValueError(
            "Reliability and probability rows do not align."
        )

    merged = merged.drop(columns=["_merge"])

    merged["cnn_predicted_hr"] = merged["predicted_hr"]

    merged = merged.sort_values(
        ["subject", "window_index"]
    ).reset_index(drop=True)

    if not np.isfinite(
        merged[
            [
                "cnn_predicted_hr",
                "unreliable_probability",
                "reference_hr",
                "absolute_error",
            ]
        ].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Nonfinite numerical values.")

    if not merged["unreliable_probability"].between(0, 1).all():
        raise ValueError("Invalid probability values.")

    for subject, group in merged.groupby("subject"):
        indexes = group["window_index"].to_numpy()

        if not np.array_equal(
            indexes,
            np.arange(len(indexes)),
        ):
            raise ValueError(
                f"Noncontiguous windows for subject {subject}."
            )

    return merged


def get_policy_observations(data):
    """Return only features allowed before investigation."""

    return data[list(OBSERVATION_COLUMNS)].copy()


def get_training_labels(data):
    """Return offline supervision, not policy observations."""

    return data[list(TRAINING_LABEL_COLUMNS)].copy()


def get_subject_sequences(data):
    """Return chronological data grouped by participant."""

    return {
        int(subject): group.copy().reset_index(drop=True)
        for subject, group in data.groupby("subject", sort=True)
    }


def save_policy_training_data(data, output_path):
    """Save the aligned policy training dataset."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(output_path, index=False)

    return output_path

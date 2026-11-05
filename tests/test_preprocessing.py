"""
MotionProof AI
Automated preprocessing regression tests.
"""

import numpy as np
import pytest

from src.preprocessing.windows import generate_windows
from src.preprocessing.features import (
    estimate_hr_from_ppg,
    calculate_motion_intensity,
    estimate_motion_cadence,
    extract_ppg_quality_features,
    extract_reliability_features,
)


def make_synthetic_participant():
    """Create a small participant without loading real data."""

    duration = 12
    fs_ppg = 64
    fs_acc = 32

    t_ppg = np.arange(duration * fs_ppg) / fs_ppg
    t_acc = np.arange(duration * fs_acc) / fs_acc

    ppg = np.sin(2 * np.pi * 1.2 * t_ppg)

    acc = np.column_stack([
        np.sin(2 * np.pi * 1.5 * t_acc),
        np.cos(2 * np.pi * 1.5 * t_acc),
        np.zeros_like(t_acc),
    ])

    return {
        "subject": "TEST",
        "ppg": ppg,
        "acc": acc,
        "labels": np.array([72.0, 72.0, 72.0]),
        "sampling_rates": {
            "ppg": 64,
            "acc": 32,
        },
    }


def test_window_shapes():
    participant = make_synthetic_participant()

    result = generate_windows(participant)

    assert result["ppg_windows"].shape == (3, 512)
    assert result["acc_windows"].shape == (3, 256, 3)
    assert result["hr_labels"].shape == (3,)


def test_window_alignment():
    participant = make_synthetic_participant()

    result = generate_windows(participant)

    assert np.array_equal(
        result["ppg_windows"][0],
        participant["ppg"][:512],
    )

    assert np.array_equal(
        result["ppg_windows"][1],
        participant["ppg"][128:640],
    )

    assert np.array_equal(
        result["acc_windows"][1],
        participant["acc"][64:320],
    )


def test_heart_rate_estimator():
    fs = 64
    t = np.arange(512) / fs

    signal = np.sin(2 * np.pi * 1.2 * t)

    predicted_hr = estimate_hr_from_ppg(signal)

    assert np.isfinite(predicted_hr)
    assert abs(predicted_hr - 72.0) < 3.0


def test_motion_intensity():
    acc = np.zeros((256, 3))

    intensity = calculate_motion_intensity(acc)

    assert intensity == pytest.approx(0.0)


def test_motion_cadence():
    fs = 32
    t = np.arange(256) / fs

    acc = np.column_stack([
        np.sin(2 * np.pi * 1.5 * t),
        np.zeros_like(t),
        np.zeros_like(t),
    ])

    cadence = estimate_motion_cadence(acc)

    assert np.isfinite(cadence)
    assert abs(cadence - 90.0) < 5.0


def test_ppg_quality_features():
    fs = 64
    t = np.arange(512) / fs

    signal = np.sin(2 * np.pi * 1.2 * t)

    concentration, entropy = (
        extract_ppg_quality_features(signal)
    )

    assert 0 <= concentration <= 1
    assert 0 <= entropy <= 1


def test_reliability_feature_extraction():
    participant = make_synthetic_participant()

    windows = generate_windows(participant)
    features = extract_reliability_features(windows)

    assert len(features) == 3

    assert features["subject"].eq("TEST").all()

    assert features["error_over_10_label"].isin(
        [0, 1]
    ).all()

    assert "reference_hr" in features.columns
    assert "absolute_error" in features.columns


def test_invalid_ppg_returns_nan():
    signal = np.zeros(512)

    assert np.isnan(estimate_hr_from_ppg(signal))


def test_invalid_acc_returns_nan():
    acc = np.full((256, 3), np.nan)

    assert np.isnan(calculate_motion_intensity(acc))
    assert np.isnan(estimate_motion_cadence(acc))
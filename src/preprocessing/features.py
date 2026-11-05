"""
MotionProof AI
Reusable reliability feature extraction.

ECG-derived reference HR is used only to construct
offline training and evaluation labels.
"""

import numpy as np
import pandas as pd

from scipy.signal import butter, sosfiltfilt, welch


FS_PPG = 64
FS_ACC = 32

LOW_HZ = 40 / 60
HIGH_HZ = 180 / 60

CADENCE_MIN_HZ = 0.7
CADENCE_MAX_HZ = 3.0

SOS = butter(
    3,
    [LOW_HZ, HIGH_HZ],
    btype="bandpass",
    fs=FS_PPG,
    output="sos",
)


def estimate_hr_from_ppg(ppg_window):
    signal = np.asarray(
        ppg_window,
        dtype=np.float64
    ).reshape(-1)

    if (
        len(signal) != 512
        or not np.all(np.isfinite(signal))
        or np.std(signal) < 1e-8
    ):
        return np.nan

    filtered = sosfiltfilt(
        SOS,
        signal - np.mean(signal)
    )

    frequencies, power = welch(
        filtered,
        fs=FS_PPG,
        nperseg=len(filtered),
        nfft=4096
    )

    valid = (
        (frequencies >= LOW_HZ)
        & (frequencies <= HIGH_HZ)
    )

    if not np.any(valid):
        return np.nan

    peak_frequency = frequencies[valid][
        np.argmax(power[valid])
    ]

    return float(peak_frequency * 60)


def calculate_motion_intensity(acc_window):
    values = np.asarray(
        acc_window,
        dtype=np.float64
    )

    if (
        values.shape != (256, 3)
        or not np.all(np.isfinite(values))
    ):
        return np.nan

    centered = values - np.mean(
        values,
        axis=0,
        keepdims=True
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(centered ** 2, axis=1)
            )
        )
    )


def estimate_motion_cadence(acc_window):
    values = np.asarray(
        acc_window,
        dtype=np.float64
    )

    if (
        values.shape != (256, 3)
        or not np.all(np.isfinite(values))
    ):
        return np.nan

    centered = values - np.mean(
        values,
        axis=0,
        keepdims=True
    )

    total_power = None

    for axis in range(3):
        frequencies, power = welch(
            centered[:, axis],
            fs=FS_ACC,
            nperseg=len(centered),
            nfft=2048
        )

        if total_power is None:
            total_power = np.zeros_like(power)

        total_power += power

    valid = (
        (frequencies >= CADENCE_MIN_HZ)
        & (frequencies <= CADENCE_MAX_HZ)
    )

    if (
        not np.any(valid)
        or np.max(total_power[valid]) <= 1e-12
    ):
        return np.nan

    peak_frequency = frequencies[valid][
        np.argmax(total_power[valid])
    ]

    return float(peak_frequency * 60)


def extract_ppg_quality_features(ppg_window):
    signal = np.asarray(
        ppg_window,
        dtype=np.float64
    ).reshape(-1)

    if (
        len(signal) != 512
        or not np.all(np.isfinite(signal))
        or np.std(signal) < 1e-8
    ):
        return np.nan, np.nan

    filtered = sosfiltfilt(
        SOS,
        signal - np.mean(signal)
    )

    frequencies, power = welch(
        filtered,
        fs=FS_PPG,
        nperseg=len(filtered),
        nfft=4096
    )

    valid = (
        (frequencies >= LOW_HZ)
        & (frequencies <= HIGH_HZ)
    )

    selected_power = power[valid]
    total_power = selected_power.sum()

    if total_power <= 1e-12:
        return np.nan, np.nan

    normalized_power = selected_power / total_power

    peak_index = int(np.argmax(selected_power))

    left = max(0, peak_index - 3)
    right = min(len(selected_power), peak_index + 4)

    concentration = (
        selected_power[left:right].sum()
        / total_power
    )

    entropy = -np.sum(
        normalized_power
        * np.log(normalized_power + 1e-12)
    )

    entropy = entropy / np.log(
        len(normalized_power)
    )

    return float(concentration), float(entropy)


def extract_reliability_features(window_data):
    """
    Generate inference-time features and offline labels.

    The returned DataFrame contains reference information
    for offline analysis. Model training must explicitly
    select only the permitted feature columns.
    """

    ppg_windows = window_data["ppg_windows"]
    acc_windows = window_data["acc_windows"]
    labels = window_data["hr_labels"]

    if not (
        len(ppg_windows)
        == len(acc_windows)
        == len(labels)
    ):
        raise ValueError(
            "PPG, ACC and label counts must match."
        )

    rows = []
    previous_hr = None

    for i in range(len(labels)):
        ppg_window = ppg_windows[i]
        acc_window = acc_windows[i]

        predicted_hr = estimate_hr_from_ppg(
            ppg_window
        )

        motion_intensity = calculate_motion_intensity(
            acc_window
        )

        motion_cadence = estimate_motion_cadence(
            acc_window
        )

        concentration, entropy = (
            extract_ppg_quality_features(ppg_window)
        )

        cadence_gap = (
            abs(predicted_hr - motion_cadence)
            if np.isfinite(predicted_hr)
            and np.isfinite(motion_cadence)
            else np.nan
        )

        hr_change = (
            abs(predicted_hr - previous_hr)
            if previous_hr is not None
            and np.isfinite(previous_hr)
            and np.isfinite(predicted_hr)
            else 0.0
        )

        reference_hr = float(labels[i])

        absolute_error = (
            abs(predicted_hr - reference_hr)
            if np.isfinite(predicted_hr)
            and np.isfinite(reference_hr)
            else np.nan
        )

        error_label = (
            int(absolute_error > 10)
            if np.isfinite(absolute_error)
            else np.nan
        )

        rows.append({
            "subject": window_data["subject"],
            "window_index": i,
            "predicted_hr": predicted_hr,
            "motion_intensity": motion_intensity,
            "motion_cadence_cpm": motion_cadence,
            "prediction_cadence_gap": cadence_gap,
            "ppg_spectral_concentration": concentration,
            "ppg_spectral_entropy": entropy,
            "hr_change_from_previous": hr_change,
            "reference_hr": reference_hr,
            "absolute_error": absolute_error,
            "error_over_10_label": error_label,
        })

        previous_hr = predicted_hr

    return pd.DataFrame(rows)
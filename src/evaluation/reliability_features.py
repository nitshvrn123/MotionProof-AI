"""Extract inference-available PPG and accelerometer reliability features."""

import numpy as np


PPG_FS = 64.0
EPS = 1e-12

FEATURE_NAMES = (
    "ppg_std",
    "ppg_peak_to_peak",
    "ppg_rms",
    "ppg_mean_abs_diff",
    "ppg_dominant_frequency",
    "ppg_spectral_concentration",
    "ppg_spectral_entropy",
    "acc_motion_intensity",
    "acc_magnitude_mean",
    "acc_magnitude_range",
    "acc_x_std",
    "acc_y_std",
    "acc_z_std",
)


def extract_reliability_features(ppg, acc):
    """Extract 13 features from aligned PPG and ACC windows.

    Parameters
    ----------
    ppg : array-like
        Shape (N, 512), raw PPG windows.
    acc : array-like
        Shape (N, 256, 3), raw accelerometer windows.

    Returns
    -------
    numpy.ndarray
        Feature matrix with shape (N, 13).
    """

    ppg = np.asarray(ppg, dtype=np.float64)
    acc = np.asarray(acc, dtype=np.float64)

    if ppg.ndim != 2 or ppg.shape[1] != 512:
        raise ValueError(
            f"Expected PPG shape (N, 512), got {ppg.shape}"
        )

    if acc.ndim != 3 or acc.shape[1:] != (256, 3):
        raise ValueError(
            f"Expected ACC shape (N, 256, 3), got {acc.shape}"
        )

    if len(ppg) != len(acc):
        raise ValueError("PPG and ACC window counts differ.")

    if len(ppg) == 0:
        raise ValueError("At least one window is required.")

    if not np.isfinite(ppg).all():
        raise ValueError("PPG contains nonfinite values.")

    if not np.isfinite(acc).all():
        raise ValueError("ACC contains nonfinite values.")

    # PPG time-domain features
    ppg_std = np.std(ppg, axis=1)

    ppg_peak_to_peak = np.ptp(ppg, axis=1)

    ppg_rms = np.sqrt(
        np.mean(ppg ** 2, axis=1)
    )

    ppg_mean_abs_diff = np.mean(
        np.abs(np.diff(ppg, axis=1)),
        axis=1,
    )

    # PPG frequency-domain features
    centered_ppg = ppg - np.mean(
        ppg,
        axis=1,
        keepdims=True,
    )

    spectrum = np.abs(
        np.fft.rfft(centered_ppg, axis=1)
    ) ** 2

    frequencies = np.fft.rfftfreq(
        ppg.shape[1],
        d=1.0 / PPG_FS,
    )

    # Plausible heart-rate frequency range: 0.7–3.5 Hz.
    frequency_mask = (
        (frequencies >= 0.7)
        & (frequencies <= 3.5)
    )

    selected_spectrum = spectrum[
        :, frequency_mask
    ]

    selected_frequencies = frequencies[
        frequency_mask
    ]

    dominant_indices = np.argmax(
        selected_spectrum,
        axis=1,
    )

    dominant_frequency = selected_frequencies[
        dominant_indices
    ]

    total_power = np.sum(
        selected_spectrum,
        axis=1,
    )

    dominant_power = np.max(
        selected_spectrum,
        axis=1,
    )

    spectral_concentration = (
        dominant_power / (total_power + EPS)
    )

    probabilities = (
        selected_spectrum
        / (total_power[:, None] + EPS)
    )

    spectral_entropy = -np.sum(
        probabilities
        * np.log(probabilities + EPS),
        axis=1,
    )

    # Accelerometer motion features
    acc_magnitude = np.linalg.norm(
        acc,
        axis=2,
    )

    motion_intensity = np.std(
        acc_magnitude,
        axis=1,
    )

    magnitude_mean = np.mean(
        acc_magnitude,
        axis=1,
    )

    magnitude_range = np.ptp(
        acc_magnitude,
        axis=1,
    )

    axis_std = np.std(
        acc,
        axis=1,
    )

    features = np.column_stack([
        ppg_std,
        ppg_peak_to_peak,
        ppg_rms,
        ppg_mean_abs_diff,
        dominant_frequency,
        spectral_concentration,
        spectral_entropy,
        motion_intensity,
        magnitude_mean,
        magnitude_range,
        axis_std[:, 0],
        axis_std[:, 1],
        axis_std[:, 2],
    ])

    assert features.shape == (
        len(ppg),
        len(FEATURE_NAMES),
    )

    if not np.isfinite(features).all():
        raise ValueError(
            "Extracted features contain nonfinite values."
        )

    return features


if __name__ == "__main__":

    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]

    sample_path = (
        project_root
        / "data/processed/train/S1.npz"
    )

    with np.load(sample_path) as data:

        ppg = data["ppg"][:32]
        acc = data["acc"][:32]

    features = extract_reliability_features(
        ppg,
        acc,
    )

    print("Reliability Feature Extraction Verification\n")

    print("Input PPG shape:", ppg.shape)
    print("Input ACC shape:", acc.shape)
    print("Feature matrix shape:", features.shape)
    print("Number of features:", len(FEATURE_NAMES))

    assert features.shape == (32, 13)

    for name, value in zip(
        FEATURE_NAMES,
        features[0],
    ):
        print(f"{name}: {value:.6f}")

    print("\nFeature extraction: PASSED")
    print("Numerical integrity: PASSED")
    print("Step 50.9B: COMPLETED")

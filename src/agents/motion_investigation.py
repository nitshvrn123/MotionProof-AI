"""Accelerometer motion investigation for MotionProof AI.

Uses only the current 8-second accelerometer window.
No ECG reference HR or ground-truth error is accessed.
"""

from dataclasses import dataclass, asdict

import numpy as np


ACC_SAMPLE_RATE = 32.0
ACC_WINDOW_SAMPLES = 256

MIN_MOTION_FREQUENCY = 0.25
MAX_MOTION_FREQUENCY = 8.0

EPSILON = 1e-12


@dataclass(frozen=True)
class MotionInvestigationResult:
    motion_intensity: float
    magnitude_mean: float
    magnitude_range: float
    acc_x_std: float
    acc_y_std: float
    acc_z_std: float
    dominant_motion_frequency_hz: float
    motion_spectral_concentration: float

    def to_dict(self):
        return asdict(self)


def investigate_motion(
    acc_window,
    sample_rate=ACC_SAMPLE_RATE,
):
    """Analyze one 8-second accelerometer window.

    Parameters
    ----------
    acc_window:
        Raw accelerometer data with shape (256, 3).

    sample_rate:
        Accelerometer sampling rate in Hz.

    Returns
    -------
    MotionInvestigationResult
        Motion-related diagnostic measurements.
    """

    acc = np.asarray(
        acc_window,
        dtype=np.float64,
    )

    if acc.shape != (ACC_WINDOW_SAMPLES, 3):
        raise ValueError(
            "Expected accelerometer shape (256, 3), "
            f"received {acc.shape}"
        )

    if not np.isfinite(acc).all():
        raise ValueError(
            "Accelerometer contains nonfinite values."
        )

    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(
            "Sample rate must be positive and finite."
        )

    # ---------------------------------------------
    # 1. Acceleration magnitude
    # ---------------------------------------------

    magnitude = np.linalg.norm(
        acc,
        axis=1,
    )

    motion_intensity = float(
        np.std(magnitude)
    )

    magnitude_mean = float(
        np.mean(magnitude)
    )

    magnitude_range = float(
        np.ptp(magnitude)
    )

    axis_std = np.std(
        acc,
        axis=0,
    )

    # ---------------------------------------------
    # 2. Frequency-domain motion analysis
    # ---------------------------------------------

    centered = magnitude - magnitude.mean()

    windowed = centered * np.hanning(
        len(centered)
    )

    spectrum = np.fft.rfft(
        windowed
    )

    power = np.abs(spectrum) ** 2

    frequencies = np.fft.rfftfreq(
        len(centered),
        d=1.0 / sample_rate,
    )

    motion_mask = (
        (frequencies >= MIN_MOTION_FREQUENCY)
        & (frequencies <= MAX_MOTION_FREQUENCY)
    )

    motion_power = power[motion_mask]
    motion_frequencies = frequencies[motion_mask]

    total_power = float(
        motion_power.sum()
    )

    if total_power <= EPSILON:
        dominant_frequency = 0.0
        spectral_concentration = 0.0
    else:
        peak_index = int(
            np.argmax(motion_power)
        )

        dominant_frequency = float(
            motion_frequencies[peak_index]
        )

        spectral_concentration = float(
            motion_power[peak_index] / total_power
        )

    return MotionInvestigationResult(
        motion_intensity=motion_intensity,
        magnitude_mean=magnitude_mean,
        magnitude_range=magnitude_range,
        acc_x_std=float(axis_std[0]),
        acc_y_std=float(axis_std[1]),
        acc_z_std=float(axis_std[2]),
        dominant_motion_frequency_hz=dominant_frequency,
        motion_spectral_concentration=spectral_concentration,
    )

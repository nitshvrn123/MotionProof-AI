"""PPG investigation tool for MotionProof AI.

Analyzes one 8-second PPG window using only information
available at inference time.

No ECG reference heart rate or ground-truth error is used.
"""

from dataclasses import dataclass, asdict

import numpy as np


PPG_SAMPLE_RATE = 64.0
PPG_WINDOW_SAMPLES = 512

MIN_PULSE_FREQUENCY = 0.7
MAX_PULSE_FREQUENCY = 3.5

EPSILON = 1e-12


@dataclass(frozen=True)
class PPGInvestigationResult:
    dominant_frequency_hz: float
    spectral_hr_bpm: float
    spectral_concentration: float
    spectral_entropy: float
    autocorrelation_strength: float
    cnn_predicted_hr_bpm: float
    cnn_spectral_disagreement_bpm: float

    def to_dict(self):
        return asdict(self)


def investigate_ppg(
    ppg_window,
    cnn_predicted_hr,
    sample_rate=PPG_SAMPLE_RATE,
):
    """Investigate one PPG window.

    Parameters
    ----------
    ppg_window:
        Raw, unnormalized PPG samples with shape (512,).

    cnn_predicted_hr:
        CNN-predicted heart rate in BPM.

    sample_rate:
        PPG sampling frequency in Hz.

    Returns
    -------
    PPGInvestigationResult
        Signal-quality and frequency-based diagnostic evidence.
    """

    signal = np.asarray(
        ppg_window,
        dtype=np.float64,
    )

    if signal.shape != (PPG_WINDOW_SAMPLES,):
        raise ValueError(
            "Expected PPG window shape (512,), "
            f"received {signal.shape}"
        )

    if not np.isfinite(signal).all():
        raise ValueError(
            "PPG window contains nonfinite values."
        )

    if not np.isfinite(cnn_predicted_hr):
        raise ValueError(
            "CNN prediction must be finite."
        )

    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(
            "Sample rate must be positive and finite."
        )

    # ---------------------------------------------
    # 1. Remove DC offset
    # ---------------------------------------------

    centered = signal - signal.mean()

    # ---------------------------------------------
    # 2. Frequency-domain analysis
    # ---------------------------------------------

    windowed = centered * np.hanning(
        len(centered)
    )

    spectrum = np.fft.rfft(windowed)

    power = np.abs(spectrum) ** 2

    frequencies = np.fft.rfftfreq(
        len(centered),
        d=1.0 / sample_rate,
    )

    pulse_mask = (
        (frequencies >= MIN_PULSE_FREQUENCY)
        & (frequencies <= MAX_PULSE_FREQUENCY)
    )

    pulse_power = power[pulse_mask]
    pulse_frequencies = frequencies[pulse_mask]

    total_power = float(
        pulse_power.sum()
    )

    if total_power <= EPSILON:
        raise ValueError(
            "PPG window has insufficient pulse-band energy."
        )

    peak_index = int(
        np.argmax(pulse_power)
    )

    dominant_frequency = float(
        pulse_frequencies[peak_index]
    )

    spectral_hr = (
        dominant_frequency * 60.0
    )

    spectral_concentration = float(
        pulse_power[peak_index]
        / total_power
    )

    normalized_power = (
        pulse_power / total_power
    )

    positive_power = normalized_power[
        normalized_power > 0
    ]

    spectral_entropy = float(
        -np.sum(
            positive_power
            * np.log(positive_power)
        )
    )

    # ---------------------------------------------
    # 3. Autocorrelation analysis
    # ---------------------------------------------

    autocorrelation = np.correlate(
        centered,
        centered,
        mode="full",
    )

    autocorrelation = autocorrelation[
        len(centered) - 1:
    ]

    zero_lag = float(
        autocorrelation[0]
    )

    if zero_lag <= EPSILON:
        raise ValueError(
            "PPG window has insufficient signal variance."
        )

    autocorrelation /= zero_lag

    minimum_lag = max(
        1,
        int(np.ceil(
            sample_rate / MAX_PULSE_FREQUENCY
        )),
    )

    maximum_lag = min(
        len(autocorrelation) - 1,
        int(np.floor(
            sample_rate / MIN_PULSE_FREQUENCY
        )),
    )

    if minimum_lag > maximum_lag:
        raise ValueError(
            "Sampling rate does not support the pulse lag range."
        )

    autocorrelation_strength = float(
        np.max(
            autocorrelation[
                minimum_lag:maximum_lag + 1
            ]
        )
    )

    autocorrelation_strength = float(
        np.clip(
            autocorrelation_strength,
            -1.0,
            1.0,
        )
    )

    # ---------------------------------------------
    # 4. CNN versus spectral estimate
    # ---------------------------------------------

    disagreement = abs(
        float(cnn_predicted_hr)
        - spectral_hr
    )

    return PPGInvestigationResult(
        dominant_frequency_hz=dominant_frequency,
        spectral_hr_bpm=float(spectral_hr),
        spectral_concentration=spectral_concentration,
        spectral_entropy=spectral_entropy,
        autocorrelation_strength=autocorrelation_strength,
        cnn_predicted_hr_bpm=float(cnn_predicted_hr),
        cnn_spectral_disagreement_bpm=float(disagreement),
    )

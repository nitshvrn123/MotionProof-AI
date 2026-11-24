"""Human-readable descriptions of MotionProof AI investigation evidence."""

import math
from numbers import Real


def _number(value, decimals=1):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None

    value = float(value)

    if not math.isfinite(value):
        return None

    return f"{value:.{decimals}f}"


def explain_investigation(action, findings):
    """Describe recorded evidence without attributing causality to the policy."""

    if not isinstance(findings, dict):
        return "Investigation measurements are unavailable."

    if action == "CHECK_PPG":
        spectral_hr = _number(findings.get("spectral_hr_bpm"))
        cnn_hr = _number(findings.get("cnn_predicted_hr_bpm"))
        disagreement = _number(
            findings.get("cnn_spectral_disagreement_bpm")
        )
        autocorrelation = _number(
            findings.get("autocorrelation_strength"),
            decimals=2,
        )

        descriptions = []

        if spectral_hr is not None and cnn_hr is not None:
            descriptions.append(
                f"The PPG frequency analysis estimated {spectral_hr} BPM, "
                f"while the CNN predicted {cnn_hr} BPM."
            )

        if disagreement is not None:
            descriptions.append(
                f"The difference between these estimates was "
                f"{disagreement} BPM."
            )

        if autocorrelation is not None:
            descriptions.append(
                f"The measured PPG autocorrelation strength was "
                f"{autocorrelation}."
            )

        return " ".join(descriptions) or "PPG measurements are unavailable."

    if action == "CHECK_MOTION":
        intensity = _number(findings.get("motion_intensity"), 3)
        frequency = _number(
            findings.get("dominant_motion_frequency_hz"),
            2,
        )
        magnitude_range = _number(
            findings.get("magnitude_range"),
            3,
        )

        descriptions = []

        if intensity is not None:
            descriptions.append(
                f"The acceleration-magnitude variation was {intensity}."
            )

        if magnitude_range is not None:
            descriptions.append(
                f"The acceleration-magnitude range was {magnitude_range}."
            )

        if frequency is not None:
            descriptions.append(
                f"The dominant motion frequency was {frequency} Hz."
            )

        return " ".join(descriptions) or "Motion measurements are unavailable."

    if action == "CHECK_CONSISTENCY":
        count = findings.get("history_count")

        if isinstance(count, int) and count == 0:
            return (
                "No earlier heart-rate predictions were available "
                "for this consistency check."
            )

        previous = _number(findings.get("previous_hr_bpm"))
        historical_median = _number(
            findings.get("historical_median_hr_bpm")
        )
        jump = _number(findings.get("hr_jump_bpm"))

        descriptions = []

        if previous is not None:
            descriptions.append(
                f"The previous predicted heart rate was {previous} BPM."
            )

        if historical_median is not None:
            descriptions.append(
                f"The median of recent predictions was "
                f"{historical_median} BPM."
            )

        if jump is not None:
            descriptions.append(
                f"The absolute change from the previous prediction "
                f"was {jump} BPM."
            )

        return " ".join(descriptions) or (
            "Historical heart-rate measurements are unavailable."
        )

    return "No measurement explanation is defined for this action."

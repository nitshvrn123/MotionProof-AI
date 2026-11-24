import math

from src.agents.investigation_explanations import explain_investigation


def test_ppg_explanation():
    findings = {
        "spectral_hr_bpm": 82.5,
        "cnn_predicted_hr_bpm": 95.0,
        "cnn_spectral_disagreement_bpm": 12.5,
        "autocorrelation_strength": 0.72,
    }

    explanation = explain_investigation("CHECK_PPG", findings)

    assert "82.5 BPM" in explanation
    assert "95.0 BPM" in explanation
    assert "12.5 BPM" in explanation
    assert "0.72" in explanation


def test_motion_explanation():
    findings = {
        "motion_intensity": 0.425,
        "magnitude_range": 1.234,
        "dominant_motion_frequency_hz": 2.5,
    }

    explanation = explain_investigation("CHECK_MOTION", findings)

    assert "0.425" in explanation
    assert "1.234" in explanation
    assert "2.50 Hz" in explanation


def test_consistency_explanation():
    findings = {
        "history_count": 5,
        "previous_hr_bpm": 80.0,
        "historical_median_hr_bpm": 78.0,
        "hr_jump_bpm": 15.0,
    }

    explanation = explain_investigation("CHECK_CONSISTENCY", findings)

    assert "80.0 BPM" in explanation
    assert "78.0 BPM" in explanation
    assert "15.0 BPM" in explanation


def test_missing_history():
    findings = {
        "history_count": 0,
        "previous_hr_bpm": math.nan,
        "historical_median_hr_bpm": math.nan,
        "hr_jump_bpm": math.nan,
    }

    explanation = explain_investigation("CHECK_CONSISTENCY", findings)

    assert "No earlier heart-rate predictions" in explanation
    assert "nan" not in explanation.lower()


def test_missing_measurements():
    explanation = explain_investigation("CHECK_PPG", {})

    assert explanation == "PPG measurements are unavailable."


def test_nonfinite_measurements():
    findings = {
        "spectral_hr_bpm": math.inf,
        "cnn_predicted_hr_bpm": math.nan,
        "cnn_spectral_disagreement_bpm": math.inf,
        "autocorrelation_strength": math.nan,
    }

    explanation = explain_investigation("CHECK_PPG", findings)

    assert explanation == "PPG measurements are unavailable."


def test_unknown_action():
    explanation = explain_investigation("UNKNOWN", {})

    assert "No measurement explanation" in explanation


def test_invalid_findings():
    explanation = explain_investigation("CHECK_MOTION", None)

    assert explanation == "Investigation measurements are unavailable."

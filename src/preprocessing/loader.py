"""
MotionProof AI
PPG-DaLiA participant data loader.

Loads trusted participant pickle files and validates
the expected physiological signals.
"""

from pathlib import Path
import pickle

import numpy as np


SAMPLING_RATES = {
    "ppg": 64,
    "acc": 32,
    "ecg": 700,
    "activity": 4,
}


def load_participant(file_path):
    """
    Load and validate a PPG-DaLiA participant.

    Parameters
    ----------
    file_path : str or Path
        Path to a trusted participant pickle file.

    Returns
    -------
    dict
        Participant identifier, signals, labels,
        activities and sampling rates.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Participant file not found: {path}"
        )

    # Pickle loading is permitted only for trusted dataset files.
    with path.open("rb") as file:
        data = pickle.load(file, encoding="latin1")

    if not isinstance(data, dict):
        raise ValueError("Expected a dictionary.")

    required_keys = {"subject", "signal", "label", "activity"}

    missing = required_keys - set(data)

    if missing:
        raise ValueError(
            f"Missing required dataset keys: {sorted(missing)}"
        )

    signals = data["signal"]

    if "wrist" not in signals or "chest" not in signals:
        raise ValueError(
            "Expected wrist and chest sensor recordings."
        )

    wrist = signals["wrist"]
    chest = signals["chest"]

    for key in ("BVP", "ACC"):
        if key not in wrist:
            raise ValueError(
                f"Missing wrist sensor: {key}"
            )

    if "ECG" not in chest:
        raise ValueError("Missing chest ECG reference.")

    ppg = np.asarray(wrist["BVP"]).reshape(-1)
    acc = np.asarray(wrist["ACC"])
    ecg = np.asarray(chest["ECG"]).reshape(-1)

    labels = np.asarray(data["label"]).reshape(-1)
    activities = np.asarray(data["activity"]).reshape(-1)

    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(
            "Expected accelerometer shape (N, 3)."
        )

    if len(ppg) == 0 or len(acc) == 0:
        raise ValueError("Empty wrist sensor recording.")

    if len(labels) == 0:
        raise ValueError("Empty heart-rate labels.")

    ppg_duration = len(ppg) / SAMPLING_RATES["ppg"]
    acc_duration = len(acc) / SAMPLING_RATES["acc"]
    ecg_duration = len(ecg) / SAMPLING_RATES["ecg"]

    if max(
        ppg_duration,
        acc_duration,
        ecg_duration,
    ) - min(
        ppg_duration,
        acc_duration,
        ecg_duration,
    ) > 1.0:
        raise ValueError(
            "Sensor durations differ by more than one second."
        )

    return {
        "subject": str(data["subject"]),
        "ppg": ppg,
        "acc": acc,
        "ecg": ecg,
        "labels": labels,
        "activity": activities,
        "sampling_rates": SAMPLING_RATES.copy(),
        "duration_seconds": ppg_duration,
    }
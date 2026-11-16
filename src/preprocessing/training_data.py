"""
MotionProof AI
Participant-independent training data preparation.

Processes PPG-DaLiA participants individually and saves
aligned PPG, accelerometer, and reference HR arrays.

Reference HR labels are used for supervised training
and offline evaluation only.
"""

from pathlib import Path

import numpy as np

from src.preprocessing.loader import load_participant
from src.preprocessing.windows import generate_windows


TRAIN_SUBJECTS = tuple(f"S{i}" for i in range(1, 11))
VAL_SUBJECTS = ("S11", "S12")
TEST_SUBJECTS = ("S13", "S14", "S15")


def get_subject_split(subject):
    """Return the predetermined participant-independent split."""

    if subject in TRAIN_SUBJECTS:
        return "train"

    if subject in VAL_SUBJECTS:
        return "val"

    if subject in TEST_SUBJECTS:
        return "test"

    raise ValueError(f"Unknown participant: {subject}")


def prepare_participant(
    subject,
    raw_dir,
    output_dir,
    overwrite=False,
):
    """
    Convert one participant's raw recording into training arrays.

    Parameters
    ----------
    subject : str
        Participant identifier, e.g. S1.
    raw_dir : str or Path
        Directory containing PPG_FieldStudy.
    output_dir : str or Path
        Directory for processed participant files.
    overwrite : bool
        Whether to replace existing processed files.

    Returns
    -------
    dict
        Participant processing summary.
    """

    split = get_subject_split(subject)

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    participant_path = (
        raw_dir
        / "PPG_FieldStudy"
        / subject
        / f"{subject}.pkl"
    )

    destination = output_dir / split / f"{subject}.npz"

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        return {
            "subject": subject,
            "split": split,
            "status": "SKIPPED",
            "path": str(destination),
        }

    participant = load_participant(participant_path)

    if participant["subject"] != subject:
        raise ValueError(
            f"Subject mismatch: expected {subject}, "
            f"received {participant['subject']}"
        )

    windows = generate_windows(participant)

    ppg = np.asarray(
        windows["ppg_windows"],
        dtype=np.float32,
    )

    acc = np.asarray(
        windows["acc_windows"],
        dtype=np.float32,
    )

    hr = np.asarray(
        windows["hr_labels"],
        dtype=np.float32,
    )

    if ppg.ndim != 2 or ppg.shape[1] != 512:
        raise ValueError("Invalid PPG window dimensions.")

    if acc.ndim != 3 or acc.shape[1:] != (256, 3):
        raise ValueError("Invalid accelerometer dimensions.")

    if len(ppg) != len(acc) or len(ppg) != len(hr):
        raise ValueError("Window and label counts differ.")

    # Remove windows with invalid input or reference values.
    valid = (
        np.isfinite(ppg).all(axis=1)
        & np.isfinite(acc).all(axis=(1, 2))
        & np.isfinite(hr)
        & (hr > 0)
    )

    ppg = ppg[valid]
    acc = acc[valid]
    hr = hr[valid]

    if len(hr) == 0:
        raise ValueError(
            f"No valid training windows for {subject}."
        )

    np.savez(
        destination,
        ppg=ppg,
        acc=acc,
        hr=hr,
        subject=np.array(subject),
        split=np.array(split),
    )

    return {
        "subject": subject,
        "split": split,
        "status": "COMPLETE",
        "windows": len(hr),
        "path": str(destination),
    }


def prepare_all_participants(
    project_root,
    overwrite=False,
):
    """Process all 15 participants sequentially."""

    project_root = Path(project_root)

    raw_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed"

    subjects = (
        TRAIN_SUBJECTS
        + VAL_SUBJECTS
        + TEST_SUBJECTS
    )

    results = []

    for subject in subjects:
        print(f"Processing {subject}...", flush=True)

        result = prepare_participant(
            subject=subject,
            raw_dir=raw_dir,
            output_dir=output_dir,
            overwrite=overwrite,
        )

        results.append(result)

        print(
            f"{subject}: {result['status']} "
            f"({result.get('windows', 'existing')} windows)",
            flush=True,
        )

    return results
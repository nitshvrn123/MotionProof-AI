"""Compute PPG normalization using fold training participants only."""

from pathlib import Path

import numpy as np

try:
    from src.evaluation.subject_folds import (
        create_subject_folds,
        validate_subject_folds,
    )
except ModuleNotFoundError:
    from subject_folds import create_subject_folds, validate_subject_folds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "train"


def compute_fold_normalization(train_subjects):
    """Compute population mean and std without loading all subjects together."""

    total_count = 0
    total_sum = 0.0
    total_sum_squares = 0.0

    for subject_id in train_subjects:
        path = PROCESSED_DIR / f"S{subject_id}.npz"

        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path) as data:
            ppg = np.asarray(data["ppg"], dtype=np.float64)

        if ppg.ndim != 2 or ppg.shape[1] != 512:
            raise ValueError(f"Invalid PPG shape for S{subject_id}: {ppg.shape}")

        if not np.isfinite(ppg).all():
            raise ValueError(f"Nonfinite PPG values for S{subject_id}")

        total_count += ppg.size
        total_sum += float(np.sum(ppg))
        total_sum_squares += float(np.sum(ppg * ppg))

    if total_count == 0:
        raise ValueError("No PPG samples available.")

    mean = total_sum / total_count
    variance = max(total_sum_squares / total_count - mean**2, 0.0)
    std = float(np.sqrt(variance))

    if not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid PPG standard deviation.")

    return float(mean), std


def main():
    folds = create_subject_folds()
    validate_subject_folds(folds)

    print("Fold-Specific PPG Normalization\n")

    for fold in folds:
        fold_id = fold["fold"]
        train_subjects = fold["train_subjects"]
        held_out_subjects = fold["held_out_subjects"]

        assert set(train_subjects).isdisjoint(held_out_subjects)

        mean, std = compute_fold_normalization(train_subjects)

        print(
            f"Fold {fold_id}: "
            f"mean={mean:.6f}, std={std:.6f} | "
            f"train={list(train_subjects)} | "
            f"held out={list(held_out_subjects)}"
        )

    print("\nFold normalization: PASSED")
    print("Held-out subject isolation: PASSED")
    print("Step 50.4: COMPLETED")


if __name__ == "__main__":
    main()

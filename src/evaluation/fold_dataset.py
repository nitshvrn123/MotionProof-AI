"""Participant-specific PPG dataset for subject-held-out evaluation."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "train"

ALLOWED_SUBJECTS = set(range(1, 11))


class FoldPPGDataset(Dataset):
    """Load only explicitly selected training participants."""

    def __init__(self, subject_ids, mean, std):

        self.subject_ids_requested = tuple(
            int(subject) for subject in subject_ids
        )

        if not self.subject_ids_requested:
            raise ValueError("At least one subject is required.")

        if len(set(self.subject_ids_requested)) != len(
            self.subject_ids_requested
        ):
            raise ValueError("Duplicate subject IDs are not allowed.")

        if not set(self.subject_ids_requested).issubset(
            ALLOWED_SUBJECTS
        ):
            raise ValueError(
                "Only S1–S10 are permitted in fold datasets."
            )

        self.mean = float(mean)
        self.std = float(std)

        if not np.isfinite(self.mean):
            raise ValueError("Normalization mean must be finite.")

        if not np.isfinite(self.std) or self.std <= 0:
            raise ValueError(
                "Normalization std must be finite and positive."
            )

        ppg_arrays = []
        hr_arrays = []
        subject_arrays = []

        for subject_id in self.subject_ids_requested:

            path = TRAIN_DATA_DIR / f"S{subject_id}.npz"

            if not path.exists():
                raise FileNotFoundError(path)

            with np.load(path) as data:

                ppg = np.asarray(
                    data["ppg"],
                    dtype=np.float32,
                )

                hr = np.asarray(
                    data["hr"],
                    dtype=np.float32,
                ).reshape(-1)

            if ppg.ndim != 2 or ppg.shape[1] != 512:
                raise ValueError(
                    f"S{subject_id}: invalid PPG shape {ppg.shape}"
                )

            if len(ppg) != len(hr):
                raise ValueError(
                    f"S{subject_id}: PPG/HR length mismatch."
                )

            if not np.isfinite(ppg).all():
                raise ValueError(
                    f"S{subject_id}: nonfinite PPG values."
                )

            if not np.isfinite(hr).all():
                raise ValueError(
                    f"S{subject_id}: nonfinite HR labels."
                )

            if not np.all(hr > 0):
                raise ValueError(
                    f"S{subject_id}: nonpositive HR labels."
                )

            ppg_arrays.append(ppg)
            hr_arrays.append(hr)

            subject_arrays.append(
                np.full(
                    len(hr),
                    subject_id,
                    dtype=np.int16,
                )
            )

        self.ppg = np.concatenate(
            ppg_arrays,
            axis=0,
        )

        self.hr = np.concatenate(
            hr_arrays,
            axis=0,
        )

        self.subject_ids = np.concatenate(
            subject_arrays,
            axis=0,
        )

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, index):

        ppg = (
            self.ppg[index] - self.mean
        ) / self.std

        hr = self.hr[index]

        return (
            torch.from_numpy(
                np.asarray(ppg, dtype=np.float32)
            ),
            torch.tensor(
                hr,
                dtype=torch.float32,
            ),
        )


if __name__ == "__main__":

    try:
        from src.evaluation.subject_folds import (
            create_subject_folds,
            validate_subject_folds,
        )
        from src.evaluation.fold_normalization import (
            compute_fold_normalization,
        )
    except ModuleNotFoundError:
        from subject_folds import (
            create_subject_folds,
            validate_subject_folds,
        )
        from fold_normalization import (
            compute_fold_normalization,
        )

    folds = create_subject_folds()
    validate_subject_folds(folds)

    print("Fold-Specific Dataset Verification\n")

    for fold in folds:

        mean, std = compute_fold_normalization(
            fold["train_subjects"]
        )

        train_dataset = FoldPPGDataset(
            fold["train_subjects"],
            mean,
            std,
        )

        held_out_dataset = FoldPPGDataset(
            fold["held_out_subjects"],
            mean,
            std,
        )

        train_subjects = set(
            train_dataset.subject_ids.tolist()
        )

        held_out_subjects = set(
            held_out_dataset.subject_ids.tolist()
        )

        assert train_subjects.isdisjoint(
            held_out_subjects
        )

        assert train_subjects == set(
            fold["train_subjects"]
        )

        assert held_out_subjects == set(
            fold["held_out_subjects"]
        )

        sample_ppg, sample_hr = train_dataset[0]

        assert sample_ppg.shape == (512,)
        assert sample_hr.ndim == 0
        assert torch.isfinite(sample_ppg).all()

        print(
            f"Fold {fold['fold']}: "
            f"Train={len(train_dataset):,} windows | "
            f"Held out={len(held_out_dataset):,} windows | "
            f"Isolation=PASSED"
        )

    print("\nAll fold datasets: PASSED")
    print("Step 50.6: COMPLETED")

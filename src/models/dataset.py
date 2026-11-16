"""
MotionProof AI
PyTorch dataset for participant-independent
heart rate estimation.

Loads preprocessed PPG windows and ECG-derived
reference heart rate labels.

ECG signals are never provided as model inputs.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PPGHeartRateDataset(Dataset):
    """
    Dataset for PPG-based heart rate regression.

    Parameters
    ----------
    processed_dir : str or Path
        Root directory containing train, val, and test.

    split : str
        One of train, val, or test.

    mean : float, optional
        Training-derived PPG normalization mean.

    std : float, optional
        Training-derived PPG normalization standard deviation.
    """

    SUBJECT_SPLITS = {
        "train": tuple(f"S{i}" for i in range(1, 11)),
        "val": ("S11", "S12"),
        "test": ("S13", "S14", "S15"),
    }

    def __init__(
        self,
        processed_dir,
        split,
        mean=None,
        std=None,
    ):

        if split not in self.SUBJECT_SPLITS:
            raise ValueError(
                "split must be train, val, or test."
            )

        if (mean is None) != (std is None):
            raise ValueError(
                "mean and std must be supplied together."
            )

        if std is not None and (
            not np.isfinite(std) or std <= 0
        ):
            raise ValueError(
                "Normalization std must be positive and finite."
            )

        if mean is not None and not np.isfinite(mean):
            raise ValueError(
                "Normalization mean must be finite."
            )

        self.split = split
        self.mean = mean
        self.std = std

        processed_dir = Path(processed_dir)

        ppg_arrays = []
        hr_arrays = []
        subject_arrays = []

        for subject in self.SUBJECT_SPLITS[split]:

            path = (
                processed_dir
                / split
                / f"{subject}.npz"
            )

            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing processed participant: {path}"
                )

            with np.load(path, allow_pickle=False) as data:

                ppg = data["ppg"].astype(
                    np.float32,
                    copy=False,
                )

                hr = data["hr"].astype(
                    np.float32,
                    copy=False,
                )

                saved_subject = data["subject"].item()
                saved_split = data["split"].item()

            if saved_subject != subject:
                raise ValueError(
                    f"Participant mismatch: {subject}"
                )

            if saved_split != split:
                raise ValueError(
                    f"Split mismatch: {subject}"
                )

            if ppg.ndim != 2 or ppg.shape[1] != 512:
                raise ValueError(
                    f"Invalid PPG dimensions: {subject}"
                )

            if hr.ndim != 1 or len(hr) != len(ppg):
                raise ValueError(
                    f"Invalid HR labels: {subject}"
                )

            if not np.isfinite(ppg).all():
                raise ValueError(
                    f"Non-finite PPG values: {subject}"
                )

            if not np.isfinite(hr).all() or (hr <= 0).any():
                raise ValueError(
                    f"Invalid HR values: {subject}"
                )

            ppg_arrays.append(ppg)
            hr_arrays.append(hr)

            subject_arrays.append(
                np.full(
                    len(hr),
                    int(subject[1:]),
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

        ppg = self.ppg[index]

        if self.mean is not None:
            ppg = (
                (ppg - self.mean)
                / self.std
            ).astype(np.float32, copy=False)

        hr = self.hr[index]

        return (
            torch.from_numpy(
                np.array(ppg, dtype=np.float32, copy=True)
            ),
            torch.tensor(
                hr,
                dtype=torch.float32,
            ),
        )
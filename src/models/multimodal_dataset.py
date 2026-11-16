"""
MotionProof AI
Multimodal dataset for PPG and accelerometer heart rate estimation.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MultimodalHeartRateDataset(Dataset):

    SUBJECT_SPLITS = {
        "train": tuple(f"S{i}" for i in range(1, 11)),
        "val": ("S11", "S12"),
        "test": ("S13", "S14", "S15"),
    }

    def __init__(
        self,
        processed_dir,
        split,
        ppg_mean,
        ppg_std,
    ):
        if split not in self.SUBJECT_SPLITS:
            raise ValueError("Invalid dataset split.")

        if not np.isfinite(ppg_mean):
            raise ValueError("Invalid PPG mean.")

        if not np.isfinite(ppg_std) or ppg_std <= 0:
            raise ValueError("Invalid PPG standard deviation.")

        self.split = split
        self.ppg_mean = float(ppg_mean)
        self.ppg_std = float(ppg_std)

        processed_dir = Path(processed_dir)

        ppg_arrays = []
        acc_arrays = []
        hr_arrays = []
        subject_arrays = []

        # ── Load participant recordings ────────────

        for subject in self.SUBJECT_SPLITS[split]:

            path = processed_dir / split / f"{subject}.npz"

            if not path.is_file():
                raise FileNotFoundError(path)

            with np.load(path, allow_pickle=False) as data:
                ppg = data["ppg"].astype(np.float32, copy=False)
                acc = data["acc"].astype(np.float32, copy=False)
                hr = data["hr"].astype(np.float32, copy=False)

                saved_subject = data["subject"].item()
                saved_split = data["split"].item()

            if saved_subject != subject or saved_split != split:
                raise ValueError(f"Participant metadata mismatch: {subject}")

            if ppg.ndim != 2 or ppg.shape[1] != 512:
                raise ValueError(f"Invalid PPG shape: {subject}")

            if acc.ndim != 3 or acc.shape[1:] != (256, 3):
                raise ValueError(f"Invalid ACC shape: {subject}")

            if hr.ndim != 1 or len(ppg) != len(acc) or len(hr) != len(ppg):
                raise ValueError(f"Window alignment mismatch: {subject}")

            if not np.isfinite(ppg).all():
                raise ValueError(f"Invalid PPG values: {subject}")

            if not np.isfinite(acc).all():
                raise ValueError(f"Invalid ACC values: {subject}")

            if not np.isfinite(hr).all() or (hr <= 0).any():
                raise ValueError(f"Invalid HR values: {subject}")

            ppg_arrays.append(ppg)
            acc_arrays.append(acc)
            hr_arrays.append(hr)

            subject_arrays.append(
                np.full(
                    len(hr),
                    int(subject[1:]),
                    dtype=np.int16,
                )
            )

        # ── Combine participants within each split ─

        self.ppg = np.concatenate(ppg_arrays, axis=0)
        self.acc = np.concatenate(acc_arrays, axis=0)
        self.hr = np.concatenate(hr_arrays, axis=0)
        self.subject_ids = np.concatenate(subject_arrays, axis=0)

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, index):

        # PPG normalization uses training-only statistics.
        ppg = (
            (self.ppg[index] - self.ppg_mean)
            / self.ppg_std
        ).astype(np.float32, copy=False)

        # Keep accelerometer values in their original units.
        acc = self.acc[index]

        hr = self.hr[index]

        return (
            torch.from_numpy(np.array(ppg, dtype=np.float32, copy=True)),
            torch.from_numpy(np.array(acc, dtype=np.float32, copy=True)),
            torch.tensor(hr, dtype=torch.float32),
        )
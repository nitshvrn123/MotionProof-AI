"""
MotionProof AI: PPG-only 1D CNN heart rate estimator.

Input:
    PPG windows with shape (batch_size, 512)
    or (batch_size, 1, 512).

Output:
    Predicted heart rate in BPM with shape (batch_size,).

This module defines the model architecture only.
Training, validation, and evaluation are implemented separately.
"""

import torch
from torch import nn


class PPGHeartRateCNN(nn.Module):
    """A compact 1D CNN for PPG-based heart rate regression."""

    def __init__(self, dropout: float = 0.2):
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be between 0 and 1.")

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, ppg: torch.Tensor) -> torch.Tensor:
        if ppg.ndim == 2:
            ppg = ppg.unsqueeze(1)

        if ppg.ndim != 3:
            raise ValueError(
                "Expected PPG input with shape (batch, 512) "
                "or (batch, 1, 512)."
            )

        if ppg.shape[1] != 1 or ppg.shape[2] != 512:
            raise ValueError(
                f"Expected shape (batch, 1, 512), got {tuple(ppg.shape)}."
            )

        if not torch.isfinite(ppg).all():
            raise ValueError("PPG input contains NaN or infinite values.")

        features = self.features(ppg)
        predictions = self.regressor(features)

        return predictions.squeeze(-1)

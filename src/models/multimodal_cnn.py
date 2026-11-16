import torch
from torch import nn


class MultimodalHeartRateCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # ── PPG encoder ────────────────────────────

        self.ppg_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # ── Accelerometer encoder ──────────────────

        self.acc_encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # ── Feature fusion ─────────────────────────

        self.regressor = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

    def forward(self, ppg, acc):

        if ppg.ndim == 2:
            ppg = ppg.unsqueeze(1)

        if ppg.ndim != 3 or ppg.shape[1:] != (1, 512):
            raise ValueError(
                "PPG input must have shape (batch, 512) "
                "or (batch, 1, 512)."
            )

        if acc.ndim != 3 or acc.shape[1:] != (256, 3):
            raise ValueError(
                "ACC input must have shape (batch, 256, 3)."
            )

        if ppg.shape[0] != acc.shape[0]:
            raise ValueError(
                "PPG and ACC batch sizes must match."
            )

        acc = acc.transpose(1, 2)

        ppg_features = self.ppg_encoder(ppg)
        acc_features = self.acc_encoder(acc)

        fused_features = torch.cat(
            [ppg_features, acc_features],
            dim=1,
        )

        predictions = self.regressor(fused_features)

        return predictions.squeeze(-1)
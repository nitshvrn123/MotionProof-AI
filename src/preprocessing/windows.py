"""
MotionProof AI
Synchronized PPG and accelerometer window generation.
"""

import numpy as np


def generate_windows(
    participant,
    window_seconds=8,
    stride_seconds=2,
):
    """
    Generate synchronized PPG and accelerometer windows.

    Reference HR labels are used for offline training
    and evaluation only.
    """

    ppg = participant["ppg"]
    acc = participant["acc"]
    labels = participant["labels"]

    fs_ppg = participant["sampling_rates"]["ppg"]
    fs_acc = participant["sampling_rates"]["acc"]

    ppg_window_size = window_seconds * fs_ppg
    acc_window_size = window_seconds * fs_acc

    ppg_stride = stride_seconds * fs_ppg
    acc_stride = stride_seconds * fs_acc

    ppg_windows = []
    acc_windows = []
    aligned_labels = []

    for i in range(len(labels)):
        ppg_start = i * ppg_stride
        acc_start = i * acc_stride

        ppg_window = ppg[
            ppg_start:ppg_start + ppg_window_size
        ]

        acc_window = acc[
            acc_start:acc_start + acc_window_size
        ]

        if len(ppg_window) != ppg_window_size:
            break

        if len(acc_window) != acc_window_size:
            break

        ppg_windows.append(ppg_window)
        acc_windows.append(acc_window)
        aligned_labels.append(labels[i])

    if not ppg_windows:
        raise ValueError("No complete synchronized windows found.")

    return {
        "subject": participant["subject"],
        "ppg_windows": np.asarray(ppg_windows),
        "acc_windows": np.asarray(acc_windows),
        "hr_labels": np.asarray(aligned_labels),
    }
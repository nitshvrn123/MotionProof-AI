"""Train PPG-only CNNs for subject-held-out reliability estimation."""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Prioritize the compatible project-local dill installation.
sys.path.insert(0, str(PROJECT_ROOT / ".local_python"))
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.ppg_cnn import PPGHeartRateCNN
from src.evaluation.subject_folds import (
    create_subject_folds,
    validate_subject_folds,
)
from src.evaluation.fold_normalization import (
    compute_fold_normalization,
)
from src.evaluation.fold_dataset import FoldPPGDataset


# ── 1. Training configuration ──────────────────────

EPOCHS = 7
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LOSS_BETA = 10.0

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "reliability_folds"


# ── 2. Reproducibility ─────────────────────────────

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── 3. Train one fold ──────────────────────────────

def train_fold(fold_id):

    folds = create_subject_folds()
    validate_subject_folds(folds)

    if not 1 <= fold_id <= len(folds):
        raise ValueError("Fold ID must be between 1 and 5.")

    fold = folds[fold_id - 1]

    train_subjects = tuple(fold["train_subjects"])
    held_out_subjects = tuple(fold["held_out_subjects"])

    assert set(train_subjects).isdisjoint(held_out_subjects)

    seed = 42 + fold_id
    set_seed(seed)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "GPU training requested, but CUDA is unavailable."
        )

    print(f"\nTraining Fold {fold_id}")
    print(f"Training subjects: {list(train_subjects)}")
    print(f"Held-out subjects: {list(held_out_subjects)}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")

    # ── 4. Training-only normalization ────────────

    mean, std = compute_fold_normalization(
        train_subjects
    )

    print(f"PPG mean: {mean:.6f}")
    print(f"PPG std: {std:.6f}")

    dataset = FoldPPGDataset(
        subject_ids=train_subjects,
        mean=mean,
        std=std,
    )

    observed_subjects = set(
        dataset.subject_ids.tolist()
    )

    assert observed_subjects == set(train_subjects)
    assert observed_subjects.isdisjoint(held_out_subjects)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )

    print(f"Training windows: {len(dataset):,}")
    print(f"Training batches: {len(loader)}")

    # ── 5. Initialize model ────────────────────────

    model = PPGHeartRateCNN(
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.SmoothL1Loss(
        beta=LOSS_BETA
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert parameter_count == 171649

    print(f"Model parameters: {parameter_count:,}")

    # ── 6. Training loop ───────────────────────────

    history = []
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0.0
        total_samples = 0

        for ppg, hr in loader:

            ppg = ppg.to(
                device,
                non_blocking=True,
            )

            hr = hr.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions = model(ppg)

            loss = criterion(
                predictions,
                hr,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Nonfinite loss in fold {fold_id}, "
                    f"epoch {epoch}."
                )

            loss.backward()

            optimizer.step()

            batch_size = ppg.shape[0]

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        epoch_loss = (
            total_loss / total_samples
        )

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Training loss: {epoch_loss:.4f}"
        )

    elapsed = time.time() - start_time

    # ── 7. Save final-epoch checkpoint ─────────────

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        CHECKPOINT_DIR / f"ppg_cnn_fold_{fold_id}.pt"
    )

    if checkpoint_path.exists():
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint_path}. "
            "Existing fold checkpoints will not be overwritten."
        )

    checkpoint = {
        "fold": fold_id,
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "normalization_mean": mean,
        "normalization_std": std,
        "train_subjects": list(train_subjects),
        "held_out_subjects": list(held_out_subjects),
        "seed": seed,
        "training_history": history,
        "training_config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss_beta": LOSS_BETA,
            "dropout": 0.2,
            "checkpoint_selection": "final_epoch",
        },
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print("\nTraining Summary")
    print(f"Fold: {fold_id}")
    print(f"Epochs completed: {EPOCHS}")
    print(f"Final training loss: {history[-1]['train_loss']:.4f}")
    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Checkpoint: {checkpoint_path}")
    print("Held-out participants used for training: NO")
    print(f"Step 50.7A Fold {fold_id}: COMPLETED")


# ── 8. Command-line entry point ────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=range(1, 6),
    )

    args = parser.parse_args()

    train_fold(args.fold)

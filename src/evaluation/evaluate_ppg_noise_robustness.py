"""Evaluate frozen CNN robustness to deterministic synthetic PPG noise."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.generate_validation_predictions import (
    load_cnn,
    predict_cnn,
)


ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = ROOT / "data/processed/test"
RESULT_DIR = ROOT / "results/metrics/final_test"

ORIGINAL_PATH = RESULT_DIR / "test_policy_inputs_nonlinear.csv"

SUMMARY_PATH = RESULT_DIR / "ppg_noise_robustness.csv"
DETAIL_PATH = RESULT_DIR / "ppg_noise_robustness_details.csv"

SUBJECTS = (13, 14, 15)

EXPECTED_WINDOWS = {
    13: 4565,
    14: 4476,
    15: 3966,
}

NOISE_LEVELS = {
    "Clean": 0.00,
    "Mild": 0.05,
    "Moderate": 0.10,
    "Severe": 0.20,
}

SEED = 42


def perturb_ppg(ppg, noise_fraction, subject):
    """Add reproducible Gaussian noise scaled per PPG window."""

    ppg = np.asarray(ppg, dtype=np.float32)

    if noise_fraction == 0:
        return ppg.copy()

    rng = np.random.default_rng(SEED + subject)

    window_std = np.std(
        ppg.astype(np.float64),
        axis=1,
        keepdims=True,
    )

    noise = rng.standard_normal(ppg.shape)

    noise *= window_std * noise_fraction

    perturbed = (
        ppg.astype(np.float64) + noise
    ).astype(np.float32)

    if not np.isfinite(perturbed).all():
        raise RuntimeError("Noise produced nonfinite PPG samples.")

    return perturbed


def calculate_summary(group, condition):

    errors = group["absolute_error"].to_numpy()

    return {
        "condition": condition,
        "windows": len(group),
        "mae_bpm": float(np.mean(errors)),
        "rmse_bpm": float(np.sqrt(np.mean(errors ** 2))),
        "unreliable_rate": float(np.mean(errors > 10.0)),
    }


def main():

    print("=" * 75)
    print("MOTIONPROOF AI — STEP 37.4C")
    print("FROZEN CNN PPG NOISE ROBUSTNESS")
    print("=" * 75)

    for path in (SUMMARY_PATH, DETAIL_PATH):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}"
            )

    original = pd.read_csv(ORIGINAL_PATH)

    assert len(original) == 13007

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    cnn, mean, std = load_cnn(device)

    print("\nDevice:", device)
    print("PASS: Frozen CNN loaded.")
    print("Normalization mean:", mean)
    print("Normalization std:", std)

    all_details = []

    for subject in SUBJECTS:

        path = TEST_DIR / f"S{subject}.npz"

        with np.load(path) as data:
            ppg = data["ppg"].copy()
            reference_hr = data["hr"].copy()

        expected = EXPECTED_WINDOWS[subject]

        assert ppg.shape == (expected, 512)
        assert reference_hr.shape == (expected,)

        subject_original = (
            original.loc[original["subject"] == subject]
            .sort_values("window_index")
            .reset_index(drop=True)
        )

        assert len(subject_original) == expected

        assert np.array_equal(
            subject_original["window_index"].to_numpy(),
            np.arange(expected),
        )

        assert np.allclose(
            subject_original["reference_hr"].to_numpy(),
            reference_hr,
            rtol=0,
            atol=1e-5,
        )

        for condition, noise_fraction in NOISE_LEVELS.items():

            print(
                f"\nS{subject} | {condition} "
                f"| noise={noise_fraction:.0%}",
                flush=True,
            )

            perturbed = perturb_ppg(
                ppg,
                noise_fraction,
                subject,
            )

            predicted_hr = predict_cnn(
                model=cnn,
                ppg=perturbed,
                mean=mean,
                std=std,
                device=device,
            )

            predicted_hr = np.asarray(
                predicted_hr,
                dtype=np.float64,
            ).reshape(-1)

            assert predicted_hr.shape == (expected,)
            assert np.isfinite(predicted_hr).all()

            if condition == "Clean":

                original_predictions = (
                    subject_original["cnn_predicted_hr"]
                    .to_numpy(dtype=np.float64)
                )

                if not np.allclose(
                    predicted_hr,
                    original_predictions,
                    rtol=0,
                    atol=1e-4,
                ):
                    raise RuntimeError(
                        f"S{subject} clean predictions "
                        "do not reproduce frozen test results."
                    )

                print(
                    "PASS: Clean predictions reproduce "
                    "original frozen CNN predictions."
                )

            errors = np.abs(
                predicted_hr - reference_hr
            )

            details = pd.DataFrame({
                "subject": subject,
                "window_index": np.arange(expected),
                "condition": condition,
                "noise_fraction": noise_fraction,
                "reference_hr": reference_hr,
                "cnn_predicted_hr": predicted_hr,
                "absolute_error": errors,
                "unreliable_target": (
                    errors > 10.0
                ).astype(int),
            })

            all_details.append(details)

            print(
                f"MAE={errors.mean():.4f} BPM | "
                f"RMSE={np.sqrt(np.mean(errors ** 2)):.4f} BPM | "
                f"Unreliable rate={np.mean(errors > 10):.4f}"
            )

    combined = pd.concat(
        all_details,
        ignore_index=True,
    )

    assert len(combined) == 13007 * len(NOISE_LEVELS)

    assert not combined.duplicated(
        ["subject", "window_index", "condition"]
    ).any()

    summaries = []

    for condition in NOISE_LEVELS:

        group = combined.loc[
            combined["condition"] == condition
        ]

        summaries.append(
            calculate_summary(group, condition)
        )

    summary = pd.DataFrame(summaries)

    clean_mae = float(
        summary.loc[
            summary["condition"] == "Clean",
            "mae_bpm",
        ].iloc[0]
    )

    summary["mae_change_bpm"] = (
        summary["mae_bpm"] - clean_mae
    )

    print("\n" + "=" * 75)
    print("PPG NOISE ROBUSTNESS RESULTS")
    print("=" * 75)

    print(
        summary.round(4).to_string(index=False)
    )

    combined.to_csv(
        DETAIL_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\nSaved:", DETAIL_PATH)
    print("Saved:", SUMMARY_PATH)

    print("\nPASS: Step 37.4C completed.")


if __name__ == "__main__":
    main()

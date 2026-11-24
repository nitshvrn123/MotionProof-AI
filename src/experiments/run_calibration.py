"""Phase 1: reliability-classifier calibration.

1. Score the original probabilities and each calibrator on validation (S11,
   S12) with leave-one-subject-out fitting.
2. Select the method with the lowest pooled out-of-subject Brier score.
3. Refit every candidate on all validation windows and save the artifacts.
4. Evaluate original vs calibrated probabilities once on held-out test.

Usage:
    python -m src.experiments.run_calibration
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.calibration import (
    calibration_metrics,
    leave_one_subject_out,
    make_calibrator,
    reliability_table,
)
from src.experiments.common import (
    EXPERIMENTAL_CHECKPOINTS,
    TEST_INPUTS,
    VALIDATION_INPUTS,
    assert_no_test_subjects,
    classification_metrics,
    load_config,
    output_dir,
    set_seed,
    write_manifest,
)

PROB = "unreliable_probability"
TARGET = "unreliable_target"
COLORS = {"original": "#948277", "platt": "#C99A65", "isotonic": "#B96547"}


def select_method(validation, config):
    bins = config["ece_bins"]
    pooled, folds = [], []

    for method in config["candidates"]:
        preds, method_folds = leave_one_subject_out(validation, method, PROB, TARGET, n_bins=bins)
        folds.extend(method_folds)
        pooled.append({"method": method, **calibration_metrics(validation[TARGET], preds, bins)})

    pooled = pd.DataFrame(pooled)
    ranked = pooled.sort_values([config["selection_metric"], config["tie_breaker"]])
    return ranked.iloc[0]["method"], pooled, pd.DataFrame(folds)


def plot_reliability(test, probs, bins, path):
    fig, (ax, hist) = plt.subplots(
        2, 1, figsize=(6.4, 7.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], ls=":", color="#B8A898", label="Perfect calibration")

    for method, p in probs.items():
        table = pd.DataFrame(reliability_table(test[TARGET], p, bins))
        table = table[table["count"] >= 20]
        ax.plot(table["mean_predicted"], table["observed_rate"], marker="o",
                color=COLORS[method], label=method.capitalize())
        hist.hist(p, bins=30, range=(0, 1), histtype="step", lw=1.6, color=COLORS[method])

    ax.set_ylabel("Observed failure rate (|error| > 10 BPM)")
    ax.set_title("Reliability diagram, held-out test S13–S15 (bins with ≥20 windows)")
    ax.legend(frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    hist.set_xlabel("Predicted failure probability")
    hist.set_ylabel("Windows")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(calibrators, path):
    grid = np.linspace(0.001, 0.999, 400)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot(grid, grid, ls=":", color="#B8A898", label="Identity")

    for method, cal in calibrators.items():
        if method != "original":
            ax.plot(grid, cal.predict(grid), color=COLORS[method], lw=2, label=method.capitalize())

    ax.set_xlabel("Original classifier probability")
    ax.set_ylabel("Calibrated probability")
    ax.set_title("Calibration maps fitted on validation S11–S12")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    config = load_config("phase1_calibration")
    set_seed(config["seed"])
    bins = config["ece_bins"]
    out = output_dir("phase1_calibration")
    artifacts = EXPERIMENTAL_CHECKPOINTS / "calibration"
    artifacts.mkdir(parents=True, exist_ok=True)

    validation = pd.read_csv(VALIDATION_INPUTS)
    test = pd.read_csv(TEST_INPUTS)
    assert_no_test_subjects(validation)
    assert set(validation["subject"]) == set(config["fit_subjects"])
    assert set(test["subject"]) == set(config["evaluation_subjects"])

    selected, pooled, folds = select_method(validation, config)
    pooled.to_csv(out / "validation_selection_pooled.csv", index=False)
    folds.to_csv(out / "validation_selection_folds.csv", index=False)

    calibrators, test_probs, outputs = {}, {}, []
    for method in config["candidates"]:
        cal = make_calibrator(method).fit(validation[PROB].to_numpy(), validation[TARGET].to_numpy())
        calibrators[method] = cal
        test_probs[method] = cal.predict(test[PROB].to_numpy())
        if method != "original":
            path = artifacts / f"{method}_calibrator.joblib"
            joblib.dump(cal, path)
            outputs.append(path)

    test_rows, subject_rows, bin_rows = [], [], []
    for method, p in test_probs.items():
        flagged = (p >= config["decision_threshold"]).astype(int)
        cls = classification_metrics(test[TARGET], flagged)
        test_rows.append({"method": method, "selected": method == selected,
                          **calibration_metrics(test[TARGET], p, bins),
                          **{f"threshold_{k}": cls[k] for k in
                             ("precision", "recall", "f1", "false_positive_rate", "flag_rate")}})
        for subject, idx in test.groupby("subject").groups.items():
            subject_rows.append({"method": method, "subject": int(subject),
                                 **calibration_metrics(test.loc[idx, TARGET], p[test.index.get_indexer(idx)], bins)})
        for row in reliability_table(test[TARGET], p, bins):
            bin_rows.append({"method": method, **row})

    files = {
        "test_calibration_metrics.csv": pd.DataFrame(test_rows),
        "test_calibration_per_subject.csv": pd.DataFrame(subject_rows),
        "test_reliability_bins.csv": pd.DataFrame(bin_rows),
        "test_calibrated_probabilities.csv": test[["subject", "window_index", TARGET]].assign(
            **{f"p_{m}": p for m, p in test_probs.items()}),
    }
    for name, frame in files.items():
        frame.to_csv(out / name, index=False)
        outputs.append(out / name)

    plot_reliability(test, test_probs, bins, out / "reliability_diagram_test.png")
    plot_curves(calibrators, out / "calibration_maps.png")
    selection = {"selected_method": selected,
                 "calibrators": {m: c.describe() for m, c in calibrators.items()}}
    with open(out / "selection.json", "w") as handle:
        json.dump(selection, handle, indent=2)

    outputs += [out / "validation_selection_pooled.csv", out / "validation_selection_folds.csv",
                out / "reliability_diagram_test.png", out / "calibration_maps.png", out / "selection.json"]
    write_manifest(out, config, [VALIDATION_INPUTS, TEST_INPUTS], outputs,
                   notes=["Calibrators fitted on S11-S12 only; test used once for reporting."])

    print("Validation (leave-one-subject-out, pooled):")
    print(pooled[["method", "ece", "brier", "log_loss", "roc_auc"]].round(4).to_string(index=False))
    print(f"Selected on validation: {selected}")
    print("Held-out test:")
    print(pd.DataFrame(test_rows)[["method", "ece", "brier", "log_loss", "roc_auc", "mean_predicted",
                                   "positive_rate"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

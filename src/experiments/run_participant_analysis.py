"""D4 Phase 2: per-participant results on the held-out test participants (no training, no selection).

Usage:
    python -m src.experiments.run_participant_analysis
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.experiments.common import TEST_INPUTS, load_config, output_dir, write_manifest
from src.experiments.d4_common import D4_DECISIONS, POLICIES, counts, load_test_windows, rates
from src.experiments.run_fp_reduction import TEST_DECISIONS

NAME = "d4_participant_analysis"


def summarize(part, label):
    err = part["cnn_predicted_hr"] - part["reference_hr"]
    y = part["unreliable_target"].to_numpy()
    base = {"participant": label, "windows": len(part), "cnn_mae_bpm": np.abs(err).mean(),
            "cnn_rmse_bpm": float(np.sqrt(np.mean(err ** 2))), "unreliable_rate": y.mean(),
            "classifier_auc": roc_auc_score(y, part["unreliable_probability"]),
            "median_motion_g": part["motion_intensity_g"].median()}
    rows = []
    for policy in POLICIES:
        flags = part[f"flag::{policy}"].to_numpy()
        tp, fp, tn, fn = counts(y, flags)
        r = {k: float(v) for k, v in rates(tp, fp, tn, fn).items()}
        rows.append({**base, "policy": policy, **r, "true_positives": int(tp), "false_positives": int(fp),
                     "true_negatives": int(tn), "false_negatives": int(fn),
                     "average_investigations": part[f"checks::{policy}"].mean(), "flag_rate": flags.mean()})
    return rows


def plot(table, path):
    parts = ["S13", "S14", "S15"]
    show = ["No Investigation", "Fixed Rule", "Learned Fitted Q", "Global calibrated veto"]
    colors = ["#8a8984", "#2a78d6", "#eb6834", "#1baf7a"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, metric in zip(axes, ("recall", "false_positive_rate")):
        x = np.arange(3)
        for k, (pol, c) in enumerate(zip(show, colors)):
            vals = table[(table.policy == pol) & table.participant.isin(parts)].set_index("participant").loc[parts, metric]
            ax.bar(x + (k - 1.5) * 0.2, vals * 100, 0.19, color=c, label=pol)
        ax.set_xticks(x, parts)
        ax.set_ylabel(metric.replace("_", " ") + " (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    config = load_config("d4_evaluation")
    out = output_dir(NAME)
    frame = load_test_windows()
    rows = []
    for subject, part in frame.groupby("subject"):
        rows += summarize(part, f"S{subject}")
    rows += summarize(frame, "All (pooled)")
    table = pd.DataFrame(rows)

    # how well the classifier probability separates failures within each motion level (explains Phase 1)
    level_auc = []
    for (label, level), part in frame.groupby([frame["subject"].map(lambda s: f"S{s}"), "motion_level"]):
        y = part["unreliable_target"]
        level_auc.append({"participant": label, "motion_level": level, "windows": len(part),
                          "unreliable_rate": y.mean(),
                          "classifier_auc": roc_auc_score(y, part["unreliable_probability"]) if 0 < y.mean() < 1 else np.nan})
    for level, part in frame.groupby("motion_level"):
        y = part["unreliable_target"]
        level_auc.append({"participant": "All (pooled)", "motion_level": level, "windows": len(part),
                          "unreliable_rate": y.mean(), "classifier_auc": roc_auc_score(y, part["unreliable_probability"])})
    level_auc = pd.DataFrame(level_auc)

    paths = [out / "participant_metrics.csv", out / "classifier_auc_by_motion.csv", out / "participant_comparison.png"]
    table.to_csv(paths[0], index=False)
    level_auc.to_csv(paths[1], index=False)
    plot(table, paths[2])
    write_manifest(out, config["participant_analysis"], [TEST_INPUTS, TEST_DECISIONS, D4_DECISIONS], paths)

    cols = ["participant", "policy", "cnn_mae_bpm", "unreliable_rate", "recall", "f1", "false_positive_rate",
            "average_investigations"]
    print(table[cols].round(3).to_string(index=False))
    print("\nClassifier AUC by motion level\n", level_auc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

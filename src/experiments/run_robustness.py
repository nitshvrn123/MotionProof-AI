"""Phase 3: robustness of the full MotionProof pipeline.

Synthetic part: each configured perturbation is applied to the held-out
test recordings (copies only), then the CNN, reliability features,
classifier and all three policies are re-run. The unreliable label is
recomputed from the perturbed CNN error, and agent history uses the
perturbed CNN predictions of earlier windows.

Natural part: clean test windows grouped by measured motion intensity
(accelerometer magnitude std), scored from the frozen D2 decisions.

Usage:
    python -m src.experiments.run_robustness
"""

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.agents.fitted_q_inference import load_fitted_q_policy
from src.agents.investigation_environment import InvestigationEnvironment
from src.evaluation.evaluate_validation_policies import run_fixed
from src.evaluation.generate_validation_predictions import load_cnn, predict_cnn
from src.evaluation.reliability_baseline import FEATURE_COLUMNS
from src.evaluation.reliability_features import extract_reliability_features
from src.experiments.calibration import calibration_metrics
from src.experiments.common import (
    FINAL_TEST,
    ROOT,
    TEST_INPUTS,
    classification_metrics,
    load_config,
    output_dir,
    write_manifest,
)
from src.experiments.policy_evaluation import decision_rewards
from src.experiments.robustness import apply, motion_intensity

HISTORY = 5
POLICIES = ("No Investigation", "Fixed Rule", "Learned Fitted Q")
CHECKS = ("CHECK_PPG", "CHECK_MOTION", "CHECK_CONSISTENCY")


def load_models():
    device = "cpu"
    cnn, mean, std = load_cnn(device)
    classifier = joblib.load(ROOT / "checkpoints/reliability_classifier_nonlinear_full.joblib")
    agent = load_fitted_q_policy(ROOT / "checkpoints/learned_policy_fitted_q_full.pt", device="cpu")
    return cnn, mean, std, classifier, agent, device


def run_condition(models, signals, perturbation, level, seed_base):
    cnn, mean, std, classifier, agent, device = models
    records = []

    for subject, sig in signals.items():
        ppg, acc = apply(perturbation, sig["ppg"], sig["acc"], level, seed_base + subject)
        with torch.no_grad():
            hr = np.asarray(predict_cnn(model=cnn, ppg=ppg, mean=mean, std=std, device=device),
                            dtype=np.float64).reshape(-1)
        features = extract_reliability_features(ppg, acc)
        prob = classifier.predict_proba(
            pd.DataFrame(np.column_stack([features, hr]), columns=FEATURE_COLUMNS))[:, 1]
        error = np.abs(hr - sig["hr"])

        for i in range(len(ppg)):
            history = hr[max(0, i - HISTORY):i]
            env_args = dict(ppg_window=ppg[i], acc_window=acc[i], cnn_predicted_hr=float(hr[i]),
                            unreliable_probability=float(prob[i]), previous_hr_predictions=history, budget=2)
            fixed_flag, fixed_inv = run_fixed(InvestigationEnvironment(**env_args))
            result = agent.run(InvestigationEnvironment(**env_args))
            records.append({
                "subject": subject, "window_index": i, "cnn_predicted_hr": hr[i],
                "absolute_error": error[i], "unreliable_target": int(error[i] > 10.0),
                "unreliable_probability": prob[i],
                "no_investigation_flag": int(prob[i] >= 0.5),
                "fixed_flag": fixed_flag, "fixed_investigations": fixed_inv,
                "learned_flag": int(result["final_decision"] == "FLAG_UNRELIABLE"),
                "learned_investigations": int(result["investigations_used"]),
                "learned_actions": "|".join(result["actions"]),
            })

    return pd.DataFrame(records)


def summarize(detail, labels):
    y = detail["unreliable_target"].to_numpy()
    base = {**labels, "windows": len(detail), "cnn_mae_bpm": detail["absolute_error"].mean(),
            "cnn_rmse_bpm": float(np.sqrt((detail["absolute_error"] ** 2).mean())),
            "unreliable_rate": y.mean()}
    cal = calibration_metrics(y, detail["unreliable_probability"])
    base.update({"classifier_roc_auc": cal["roc_auc"], "classifier_brier": cal["brier"],
                 "classifier_ece": cal["ece"], "classifier_mean_probability": cal["mean_predicted"]})
    actions = detail["learned_actions"]
    for check in CHECKS:
        base[f"learned_uses_{check.lower()}"] = actions.str.contains(check).mean()

    rows = []
    for policy, flag, inv in (("No Investigation", "no_investigation_flag", None),
                              ("Fixed Rule", "fixed_flag", "fixed_investigations"),
                              ("Learned Fitted Q", "learned_flag", "learned_investigations")):
        investigations = detail[inv].to_numpy() if inv else np.zeros(len(detail))
        rewards = decision_rewards(detail[flag], y, investigations)
        rows.append({**base, "policy": policy,
                     **classification_metrics(y, detail[flag], investigations, rewards)})
    return rows


def natural_subgroups(signals):
    inputs = pd.read_csv(TEST_INPUTS)
    decisions = pd.read_csv(FINAL_TEST / "test_policy_decisions.csv")
    intensity = pd.concat([
        pd.DataFrame({"subject": s, "window_index": np.arange(len(sig["acc"])),
                      "motion_intensity_g": motion_intensity(sig["acc"])})
        for s, sig in signals.items()])
    edges = np.quantile(intensity["motion_intensity_g"], [0, 0.25, 0.5, 0.75, 1.0])
    names = ["Q1 lowest motion", "Q2", "Q3", "Q4 highest motion"]
    intensity["motion_quartile"] = pd.cut(intensity["motion_intensity_g"], edges, labels=names,
                                          include_lowest=True)
    merged = decisions.merge(inputs[["subject", "window_index", "absolute_error"]],
                             on=["subject", "window_index"]).merge(intensity, on=["subject", "window_index"])
    rows = []
    for (quartile, policy), group in merged.groupby(["motion_quartile", "policy"], observed=True):
        lo, hi = group["motion_intensity_g"].min(), group["motion_intensity_g"].max()
        rows.append({"motion_quartile": quartile, "policy": policy,
                     "motion_range_g": f"{lo:.3f}-{hi:.3f}",
                     "cnn_mae_bpm": group["absolute_error"].mean(),
                     "unreliable_rate": group["unreliable_target"].mean(),
                     "mean_probability": group["unreliable_probability"].mean(),
                     **classification_metrics(group["unreliable_target"], group["predicted_unreliable"],
                                              group["investigations_used"], group["reward"])})
    return pd.DataFrame(rows), edges


def plot_conditions(summary, path):
    learned = summary[summary["policy"] == "Learned Fitted Q"].reset_index(drop=True)
    labels = [f"{p.replace('_', ' ')}\n{l:g}" if p != "clean" else "clean"
              for p, l in zip(learned["perturbation"], learned["level"])]
    x = np.arange(len(learned))
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].bar(x, learned["cnn_mae_bpm"], color="#C99A65")
    axes[0].set_ylabel("CNN MAE (BPM)")
    for policy, color in (("Learned Fitted Q", "#B96547"), ("Fixed Rule", "#C99A65"),
                          ("No Investigation", "#948277")):
        part = summary[summary["policy"] == policy].reset_index(drop=True)
        axes[1].plot(x, part["recall"], marker="o", color=color, label=policy)
        axes[2].plot(x, part["false_positive_rate"], marker="o", color=color, label=policy)
    axes[1].set_ylabel("Recall")
    axes[2].set_ylabel("False-positive rate")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].set_xticks(x, labels, rotation=60, ha="right", fontsize=7)
    axes[0].set_title("Synthetic perturbations of held-out test recordings (S13–S15)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    config = load_config("phase3_robustness")
    out = output_dir("phase3_robustness")
    models = load_models()
    signals = {}
    for subject in config["subjects"]:
        with np.load(ROOT / f"data/processed/test/S{subject}.npz") as data:
            signals[subject] = {"ppg": data["ppg"].copy(), "acc": data["acc"].copy(), "hr": data["hr"].copy()}

    summary_rows, subject_rows, details = [], [], []
    for index, cond in enumerate(config["synthetic_conditions"]):
        labels = {"condition_type": "synthetic" if cond["perturbation"] != "clean" else "clean reference",
                  "group": cond["group"], "perturbation": cond["perturbation"], "level": cond["level"]}
        detail = run_condition(models, signals, cond["perturbation"], cond["level"],
                               config["seed"] + 1000 * index)
        summary_rows += summarize(detail, labels)
        for subject, group in detail.groupby("subject"):
            subject_rows += [{**r, "subject": subject} for r in summarize(group, labels)]
        details.append(detail.assign(perturbation=cond["perturbation"], level=cond["level"]))
        learned = summary_rows[-1]
        print(f"{cond['perturbation']:<32} {cond['level']:<5} MAE {learned['cnn_mae_bpm']:.2f}  "
              f"recall {learned['recall']:.3f}  FPR {learned['false_positive_rate']:.3f}  "
              f"inv {learned['average_investigations']:.2f}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "synthetic_summary.csv", index=False)
    pd.DataFrame(subject_rows).to_csv(out / "synthetic_per_subject.csv", index=False)
    pd.concat(details).to_csv(out / "synthetic_window_details.csv.gz", index=False, compression="gzip")
    plot_conditions(summary, out / "synthetic_conditions.png")

    # Integrity: the clean live re-run must match the frozen D2 learned decisions.
    clean = details[0].merge(pd.read_csv(ROOT / "results/demo/learned_agent_traces.csv"),
                             on=["subject", "window_index"])
    agreement = float((clean["learned_flag"] == (clean["final_decision"] == "FLAG_UNRELIABLE")).mean())

    natural, edges = natural_subgroups(signals)
    natural.to_csv(out / "natural_motion_subgroups.csv", index=False)
    participants = pd.read_csv(FINAL_TEST / "test_policy_per_subject.csv")
    participants.to_csv(out / "participant_level_frozen.csv", index=False)

    notes = [f"Clean live re-run vs frozen learned decisions: {agreement:.4%} agreement.",
             "Synthetic perturbations alter copies of real held-out recordings; no participants or "
             "recordings were created.",
             "Natural subgroups use measured accelerometer motion, not activity labels; activity labels "
             "are not present in the processed files.",
             f"Motion quartile edges (g): {np.round(edges, 4).tolist()}"]
    write_manifest(out, config, [TEST_INPUTS, FINAL_TEST / "test_policy_decisions.csv"],
                   list(out.glob("*.csv")) + list(out.glob("*.gz")) + list(out.glob("*.png")), notes=notes)
    print("\n".join(notes))
    print(natural[natural["policy"] == "Learned Fitted Q"][
        ["motion_quartile", "motion_range_g", "cnn_mae_bpm", "unreliable_rate", "recall",
         "false_positive_rate", "average_investigations"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

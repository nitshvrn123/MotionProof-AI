"""D4 Phase 4: failure-case analysis of the frozen D2 agent on held-out test participants.

Investigation measurements are recomputed for every test window with the same
deterministic tools the agent uses (PPG spectrum, motion, temporal
consistency). Representative windows are chosen by a fixed rule (closest to the
category median of standardized p, motion intensity and absolute error), not
by hand. Needs the processed test recordings.

Usage:
    python -m src.experiments.run_failure_cases
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agents.consistency_investigation import investigate_consistency
from src.agents.motion_investigation import investigate_motion
from src.agents.ppg_investigation import investigate_ppg
from src.experiments.common import ROOT, TEST_INPUTS, load_config, output_dir, write_manifest
from src.experiments.d4_common import D4_DECISIONS, load_test_windows
from src.experiments.policy_evaluation import load_split
from src.experiments.run_fp_reduction import TEST_DECISIONS

NAME = "d4_failure_cases"
TRACES = ROOT / "results" / "demo" / "learned_agent_traces.csv"
HISTORY = 5
FINDINGS = ["spectral_hr_bpm", "cnn_spectral_disagreement_bpm", "spectral_concentration", "autocorrelation_strength",
            "motion_intensity", "dominant_motion_frequency_hz", "hr_jump_bpm", "median_deviation_bpm"]


def findings_table(frame, signals):
    rows = []
    for subject, part in frame.groupby("subject"):
        part = part.sort_values("window_index")
        preds = part["cnn_predicted_hr"].to_numpy()
        sig = signals[subject]
        for i, (w, hr) in enumerate(zip(part["window_index"], preds)):
            ppg = investigate_ppg(sig["ppg"][w], hr).to_dict()
            mot = investigate_motion(sig["acc"][w]).to_dict()
            con = investigate_consistency(hr, preds[max(0, i - HISTORY):i]).to_dict()
            rows.append({"subject": subject, "window_index": int(w), **{k: ppg[k] for k in FINDINGS[:4]},
                         "motion_intensity": mot["motion_intensity"],
                         "dominant_motion_frequency_hz": mot["dominant_motion_frequency_hz"],
                         "hr_jump_bpm": con["hr_jump_bpm"], "median_deviation_bpm": con["median_deviation_bpm"]})
    return pd.DataFrame(rows)


def categorize(frame):
    y = frame["unreliable_target"] == 1
    agent = frame["flag::Learned Fitted Q"] == 1
    prob_only = frame["unreliable_probability"] >= 0.5
    checked = frame["checks::Learned Fitted Q"] > 0
    changed = checked & (agent != prob_only)
    frame["category"] = np.select([agent & y, ~agent & y, agent & ~y], ["caught_failure", "missed_failure", "false_alarm"],
                                  "correct_accept")
    frame["changed_by_investigation"] = changed
    frame["change_type"] = np.select(
        [changed & agent & y, changed & ~agent & ~y, changed & agent & ~y, changed & ~agent & y],
        ["helpful: caught a failure p<0.5 would accept", "helpful: avoided a false alarm p>=0.5 would raise",
         "harmful: raised a false alarm p<0.5 would avoid", "harmful: missed a failure p>=0.5 would catch"], "")
    return frame


def representative(part):
    cols = ["unreliable_probability", "motion_intensity_g", "absolute_error"]
    z = (part[cols] - part[cols].median()) / part[cols].std(ddof=0).replace(0, 1)
    return part.loc[(z ** 2).sum(axis=1).idxmin()]


def plot_examples(examples, signals, path):
    fig, axes = plt.subplots(2, len(examples), figsize=(4 * len(examples), 4.6), sharex="row")
    for col, (title, r) in enumerate(examples):
        sig = signals[int(r.subject)]
        w = int(r.window_index)
        axes[0, col].plot(np.arange(512) / 64, sig["ppg"][w], color="#2a78d6", lw=0.9)
        axes[0, col].set_title(f"{title}\nS{int(r.subject)} window {w}", fontsize=9)
        mag = np.linalg.norm(sig["acc"][w], axis=1)
        axes[1, col].plot(np.arange(256) / 32, mag, color="#eb6834", lw=0.9)
        axes[1, col].set_xlabel("Time (s)")
        axes[1, col].text(0.02, 0.97,
                          f"CNN {r.cnn_predicted_hr:.0f} / ref {r.reference_hr:.0f} BPM\np = {r.unreliable_probability:.2f}, "
                          f"motion {r.motion_intensity_g:.2f} g\n{r.actions.replace('|', ' → ')}",
                          transform=axes[1, col].transAxes, va="top", fontsize=7,
                          bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    axes[0, 0].set_ylabel("PPG (sensor units)")
    axes[1, 0].set_ylabel("|ACC| (g)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    config = load_config("d4_evaluation")["failure_cases"]
    out = output_dir(NAME)
    frame = load_test_windows()
    _, signals = load_split("test")
    traces = pd.read_csv(TRACES)[["subject", "window_index", "actions"]]
    frame = categorize(frame.merge(traces, on=["subject", "window_index"]))
    frame = frame.merge(findings_table(frame, signals), on=["subject", "window_index"])

    summary = frame.groupby("category").agg(
        windows=("category", "size"), median_p=("unreliable_probability", "median"),
        median_abs_error_bpm=("absolute_error", "median"), median_motion_g=("motion_intensity_g", "median"),
        share_high_motion=("motion_level", lambda s: (s == "high").mean()),
        median_spectral_disagreement_bpm=("cnn_spectral_disagreement_bpm", "median"),
        median_autocorrelation=("autocorrelation_strength", "median"),
        median_hr_jump_bpm=("hr_jump_bpm", "median"),
        mean_checks=("checks::Learned Fitted Q", "mean"),
        share_decided_without_checks=("checks::Learned Fitted Q", lambda s: (s == 0).mean())).reset_index()
    changes = frame[frame.changed_by_investigation].groupby("change_type").agg(
        windows=("change_type", "size"), median_p=("unreliable_probability", "median"),
        median_abs_error_bpm=("absolute_error", "median"), median_motion_g=("motion_intensity_g", "median"),
        median_spectral_disagreement_bpm=("cnn_spectral_disagreement_bpm", "median")).reset_index()
    by_motion = pd.crosstab(frame["category"], frame["motion_level"]).reindex(columns=["low", "moderate", "high"])
    near_threshold = frame[(frame.category == "false_alarm")]["absolute_error"].between(8, 10).mean()

    examples, rows = [], []
    for cat, title in (("caught_failure", "Caught failure"), ("missed_failure", "Missed failure"),
                       ("false_alarm", "False alarm")):
        r = representative(frame[frame.category == cat])
        examples.append((title, r))
    helpful = frame[frame.change_type.str.startswith("helpful: caught")]
    examples.append(("Investigation changed decision", representative(helpful)))
    keep = ["subject", "window_index", "cnn_predicted_hr", "reference_hr", "absolute_error", "unreliable_probability",
            "motion_intensity_g", "motion_level", "actions"] + FINDINGS
    for title, r in examples:
        rows.append({"example": title, **{k: r[k] for k in keep}})
    examples_tbl = pd.DataFrame(rows)

    paths = [out / "category_summary.csv", out / "investigation_changes.csv", out / "category_by_motion.csv",
             out / "representative_examples.csv", out / "window_findings.csv.gz", out / "representative_examples.png"]
    summary.to_csv(paths[0], index=False)
    changes.to_csv(paths[1], index=False)
    by_motion.to_csv(paths[2])
    examples_tbl.to_csv(paths[3], index=False)
    frame[["subject", "window_index", "category", "change_type", "motion_level"] + FINDINGS].to_csv(paths[4], index=False)
    plot_examples(examples, signals, paths[5])
    write_manifest(out, config, [TEST_INPUTS, TEST_DECISIONS, D4_DECISIONS, TRACES], paths,
                   notes=[f"False alarms with absolute error between 8 and 10 BPM: {near_threshold:.1%}"])

    print(summary.round(3).to_string(index=False))
    print("\nInvestigation changed the decision (vs p >= 0.5)\n", changes.round(3).to_string(index=False))
    print("\nCategory by motion level\n", by_motion.to_string())
    print("\nRepresentative examples\n", examples_tbl.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

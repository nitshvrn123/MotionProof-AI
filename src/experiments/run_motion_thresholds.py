"""D4 Phase 1: motion-aware calibrated veto versus the frozen agent and the D3 global veto.

Success criteria and the selection rule are fixed in
configs/experiments/d4_motion_thresholds.json before this script is run.
Motion edges and thresholds are chosen on validation participants S11-S12
only; test participants S13-S15 are evaluated once, after selection.

Usage:
    python -m src.experiments.run_motion_thresholds
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.common import (
    TEST_INPUTS,
    TEST_SUBJECTS,
    VALIDATION_INPUTS,
    VALIDATION_SUBJECTS,
    assert_no_test_subjects,
    load_config,
    output_dir,
    set_seed,
    write_manifest,
)
from src.experiments.motion_aware import (
    LEVELS,
    apply_veto,
    assign_levels,
    metrics,
    motion_edges,
    per_level_metrics,
    select_global,
    select_motion_aware,
)
from src.experiments.run_fp_reduction import CALIBRATOR, KEYS, TEST_DECISIONS, VAL_DECISIONS, frozen_learned

NAME = "d4_motion_thresholds"


def motion_table(split, subjects, out):
    """Per-window motion intensity; computed from recordings, cached as a derived CSV."""

    path = out / f"motion_intensity_{split}.csv"
    try:
        from src.experiments.policy_evaluation import load_split
        from src.experiments.robustness import motion_intensity

        _, signals = load_split(split)
        frame = pd.concat([pd.DataFrame({"subject": s, "window_index": np.arange(len(sig["acc"])),
                                         "motion_intensity_g": motion_intensity(sig["acc"])})
                           for s, sig in signals.items()], ignore_index=True)
        frame.to_csv(path, index=False)
    except FileNotFoundError:
        frame = pd.read_csv(path)
        print(f"Recordings for {split} not found; using cached {path.name}")
    assert set(frame["subject"]) == set(subjects)
    return frame.sort_values(KEYS).reset_index(drop=True)


def split_arrays(inputs_path, decisions_path, motion, calibrator):
    inputs = pd.read_csv(inputs_path).sort_values(KEYS).reset_index(drop=True)
    agent = frozen_learned(decisions_path)
    frame = inputs.merge(agent[KEYS + ["predicted_unreliable", "investigations_used"]], on=KEYS) \
                  .merge(motion, on=KEYS)
    assert len(frame) == len(inputs) == len(agent)
    frame["p_cal"] = calibrator.predict(frame["unreliable_probability"].to_numpy())
    return frame


def evaluate(frame, levels, global_tau, motion_taus):
    y = frame["unreliable_target"].to_numpy()
    flags = frame["predicted_unreliable"].to_numpy()
    inv = frame["investigations_used"].to_numpy()
    p = frame["p_cal"].to_numpy()
    decisions = {
        "Frozen agent (D2)": flags,
        "Global calibrated veto": apply_veto(flags, p, levels, global_tau),
        "Motion-aware veto": apply_veto(flags, p, levels, motion_taus),
    }
    rows = [{"candidate": k, **metrics(y, v, inv)} for k, v in decisions.items()]
    per_level = pd.concat([per_level_metrics(y, v, inv, levels).assign(candidate=k) for k, v in decisions.items()],
                          ignore_index=True)
    return pd.DataFrame(rows), per_level, decisions


def select_both(frame, tolerance):
    y = frame["unreliable_target"].to_numpy()
    flags = frame["predicted_unreliable"].to_numpy()
    p = frame["p_cal"].to_numpy()
    edges = motion_edges(frame["motion_intensity_g"])
    levels = assign_levels(frame["motion_intensity_g"], edges)
    frozen_recall = metrics(y, flags, frame["investigations_used"].to_numpy())["recall"]
    floor = frozen_recall - tolerance
    return edges, levels, floor, select_global(y, flags, p, floor), select_motion_aware(y, flags, p, levels, floor)


def plot_levels(per_level_val, per_level_test, path):
    colors = {"Frozen agent (D2)": "#8a8984", "Global calibrated veto": "#2a78d6", "Motion-aware veto": "#eb6834"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey="row")
    for col, (title, table) in enumerate((("Validation S11–S12", per_level_val), ("Test S13–S15", per_level_test))):
        for row, metric in enumerate(("false_positive_rate", "recall")):
            ax = axes[row, col]
            x = np.arange(3)
            for k, (cand, color) in enumerate(colors.items()):
                part = table[table["candidate"] == cand].set_index("motion_level").loc[list(LEVELS)]
                ax.bar(x + (k - 1) * 0.27, part[metric] * 100, 0.25, color=color, label=cand)
            ax.set_xticks(x, [f"{lvl} motion" for lvl in LEVELS])
            ax.set_ylabel(metric.replace("_", " ") + " (%)")
            if row == 0:
                ax.set_title(title)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    config = load_config(NAME)
    set_seed(config["seed"])
    tol = config["selection_rule"]["recall_tolerance"]
    out = output_dir(NAME)
    calibrator = joblib.load(CALIBRATOR)

    val = split_arrays(VALIDATION_INPUTS, VAL_DECISIONS, motion_table("validation", VALIDATION_SUBJECTS, out), calibrator)
    assert_no_test_subjects(val)

    # selection on validation only
    edges, val_levels, floor, g_tau, m_taus = select_both(val, tol)
    val_rows, val_levels_tbl, _ = evaluate(val, val_levels, g_tau, m_taus)
    v = val_rows.set_index("candidate")
    c1 = bool(v.loc["Motion-aware veto", "false_positive_rate"]
              <= v.loc["Global calibrated veto", "false_positive_rate"] - 0.01)

    # leave-one-participant-out inside validation
    folds = []
    for sel_subject, held_subject in ((11, 12), (12, 11)):
        sel, held = val[val.subject == sel_subject], val[val.subject == held_subject]
        f_edges, _, f_floor, f_g, f_m = select_both(sel, tol)
        h_levels = assign_levels(held["motion_intensity_g"], f_edges)
        h_rows = evaluate(held, h_levels, f_g, f_m)[0].set_index("candidate")
        h_floor = h_rows.loc["Frozen agent (D2)", "recall"] - tol
        passed = bool(h_rows.loc["Motion-aware veto", "false_positive_rate"]
                      < h_rows.loc["Global calibrated veto", "false_positive_rate"]
                      and h_rows.loc["Motion-aware veto", "recall"] >= h_floor)
        for cand, r in h_rows.iterrows():
            folds.append({"selected_on": f"S{sel_subject}", "evaluated_on": f"S{held_subject}", "candidate": cand,
                          "global_tau": f_g, "motion_taus": json.dumps(f_m), "motion_edges": json.dumps(list(f_edges)),
                          "recall": r.recall, "false_positive_rate": r.false_positive_rate, "f1": r.f1,
                          "heldout_recall_floor": h_floor, "fold_pass": passed})
    folds = pd.DataFrame(folds)
    c2 = bool(folds.groupby("selected_on")["fold_pass"].first().all())
    supported = c1 and c2

    # single test evaluation
    test = split_arrays(TEST_INPUTS, TEST_DECISIONS, motion_table("test", TEST_SUBJECTS, out), calibrator)
    test_levels = assign_levels(test["motion_intensity_g"], edges)
    test_rows, test_levels_tbl, test_decisions = evaluate(test, test_levels, g_tau, m_taus)
    t = test_rows.set_index("candidate")
    c3 = bool(t.loc["Motion-aware veto", "false_positive_rate"] < t.loc["Global calibrated veto", "false_positive_rate"]
              and t.loc["Motion-aware veto", "recall"] >= t.loc["Frozen agent (D2)", "recall"] - tol)

    paths = {
        "validation": out / "validation_comparison.csv",
        "validation_levels": out / "validation_per_level.csv",
        "loso": out / "validation_loso_folds.csv",
        "test": out / "test_comparison.csv",
        "test_levels": out / "test_per_level.csv",
        "decisions": out / "test_decisions.csv",
        "selection": out / "selection.json",
        "figure": out / "per_level_comparison.png",
    }
    val_rows.to_csv(paths["validation"], index=False)
    val_levels_tbl.to_csv(paths["validation_levels"], index=False)
    folds.to_csv(paths["loso"], index=False)
    test_rows.to_csv(paths["test"], index=False)
    test_levels_tbl.to_csv(paths["test_levels"], index=False)
    pd.DataFrame({**{k: test[k] for k in KEYS}, "motion_intensity_g": test["motion_intensity_g"],
                  "motion_level": np.array(LEVELS)[test_levels], "p_cal": test["p_cal"],
                  "unreliable_target": test["unreliable_target"], "investigations_used": test["investigations_used"],
                  "frozen_agent": test_decisions["Frozen agent (D2)"],
                  "global_veto": test_decisions["Global calibrated veto"],
                  "motion_aware_veto": test_decisions["Motion-aware veto"]}).to_csv(paths["decisions"], index=False)
    selection = {
        "motion_edges_g": [float(e) for e in edges],
        "validation_recall_floor": float(floor),
        "global_veto_tau": g_tau,
        "motion_aware_taus": dict(zip(LEVELS, m_taus)),
        "criteria": {"C1_validation_gain": c1, "C2_out_of_participant": c2,
                     "motion_aware_supported": supported, "C3_test_confirmation": c3},
    }
    paths["selection"].write_text(json.dumps(selection, indent=2))
    plot_levels(val_levels_tbl, test_levels_tbl, paths["figure"])
    write_manifest(out, config, [VALIDATION_INPUTS, TEST_INPUTS, VAL_DECISIONS, TEST_DECISIONS, CALIBRATOR],
                   list(paths.values()) + [out / "motion_intensity_validation.csv", out / "motion_intensity_test.csv"],
                   notes=["Frozen D2 agent decisions are read, never modified.",
                          "Test participants are evaluated once, after selection on validation."])

    cols = ["candidate", "recall", "false_positive_rate", "f1", "precision", "average_investigations"]
    print("Selection (validation):", json.dumps(selection, indent=2))
    print("\nValidation\n", val_rows[cols].round(4).to_string(index=False))
    print("\nLeave-one-participant-out inside validation\n",
          folds[["selected_on", "evaluated_on", "candidate", "recall", "false_positive_rate", "fold_pass"]]
          .round(4).to_string(index=False))
    print("\nTest (single evaluation)\n", test_rows[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

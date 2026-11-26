"""Window-level decisions of every reported policy on the held-out test participants.

Frozen D2 policies come from results/metrics/final_test/test_policy_decisions.csv.
The D3 global veto and the D4 motion-aware veto come from
results/experiments/d4_motion_thresholds/test_decisions.csv (run Phase 1 first).
"""

import numpy as np
import pandas as pd

from src.experiments.common import RESULTS_DIR, TEST_INPUTS
from src.experiments.run_fp_reduction import KEYS, TEST_DECISIONS

POLICIES = ("No Investigation", "Fixed Rule", "Learned Fitted Q", "Global calibrated veto", "Motion-aware veto")
D4_DECISIONS = RESULTS_DIR / "d4_motion_thresholds" / "test_decisions.csv"


def load_test_windows():
    """One row per test window: inputs, motion level and flag/check columns per policy."""

    inputs = pd.read_csv(TEST_INPUTS).sort_values(KEYS).reset_index(drop=True)
    frozen = pd.read_csv(TEST_DECISIONS)
    frame = inputs.copy()
    for policy in POLICIES[:3]:
        part = frozen[frozen["policy"] == policy].sort_values(KEYS).reset_index(drop=True)
        assert (part[KEYS].to_numpy() == frame[KEYS].to_numpy()).all()
        frame[f"flag::{policy}"] = part["predicted_unreliable"].to_numpy()
        frame[f"checks::{policy}"] = part["investigations_used"].to_numpy()
    d4 = pd.read_csv(D4_DECISIONS).sort_values(KEYS).reset_index(drop=True)
    assert (d4[KEYS].to_numpy() == frame[KEYS].to_numpy()).all()
    assert (d4["frozen_agent"].to_numpy() == frame["flag::Learned Fitted Q"].to_numpy()).all()
    frame["motion_intensity_g"] = d4["motion_intensity_g"].to_numpy()
    frame["motion_level"] = d4["motion_level"].to_numpy()
    frame["p_cal"] = d4["p_cal"].to_numpy()
    for policy, column in (("Global calibrated veto", "global_veto"), ("Motion-aware veto", "motion_aware_veto")):
        frame[f"flag::{policy}"] = d4[column].to_numpy()
        frame[f"checks::{policy}"] = frame["checks::Learned Fitted Q"]
    return frame


def counts(y, flags):
    """TP, FP, TN, FN along the last axis (works on bootstrap index matrices)."""

    y = np.asarray(y, dtype=bool)
    flags = np.asarray(flags, dtype=bool)
    return ((flags & y).sum(-1), (flags & ~y).sum(-1), (~flags & ~y).sum(-1), (~flags & y).sum(-1))


def rates(tp, fp, tn, fn):
    tp, fp, tn, fn = (np.asarray(v, dtype=float) for v in (tp, fp, tn, fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        fpr = np.where(fp + tn > 0, fp / (fp + tn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {"recall": recall, "precision": precision, "false_positive_rate": fpr, "f1": f1}

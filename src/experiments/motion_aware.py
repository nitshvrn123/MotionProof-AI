"""Motion-aware calibrated veto: helpers shared by the D4 scripts and tests.

The frozen D2 agent is never changed. A veto only removes agent flags whose
calibrated failure probability is below a threshold; the threshold may depend
on the window's motion level. Motion levels and thresholds are chosen on
validation participants only.
"""

import numpy as np
import pandas as pd

from src.experiments.common import classification_metrics
from src.experiments.policy_evaluation import decision_rewards

LEVELS = ("low", "moderate", "high")


def tau_grid(include_zero):
    grid = np.round(np.arange(0.05, 0.95 + 1e-9, 0.01), 2)
    return np.concatenate([[0.0], grid]) if include_zero else grid


def motion_edges(motion):
    """Tertile edges of motion intensity (two inner cut points)."""

    return np.quantile(np.asarray(motion, dtype=float), [1 / 3, 2 / 3])


def assign_levels(motion, edges):
    """0 = low, 1 = moderate, 2 = high (edges come from validation only)."""

    return np.digitize(np.asarray(motion, dtype=float), edges, right=False)


def apply_veto(agent_flags, p_cal, levels, taus):
    """Keep an agent flag only if p_cal >= the threshold of the window's motion level."""

    taus = np.asarray(taus, dtype=float)
    threshold = taus[np.asarray(levels, dtype=int)] if taus.ndim else np.full(len(p_cal), float(taus))
    return (np.asarray(agent_flags, dtype=int) & (np.asarray(p_cal) >= threshold)).astype(int)


def metrics(y, flagged, investigations):
    return classification_metrics(y, flagged, investigations, decision_rewards(flagged, y, investigations))


def _level_counts(y, agent_flags, p_cal, levels, grid):
    """TP/FP counts per level and threshold; FN/TN follow from totals."""

    tp = np.zeros((3, len(grid)))
    fp = np.zeros((3, len(grid)))
    for lvl in range(3):
        m = levels == lvl
        flags, prob, target = agent_flags[m], p_cal[m], y[m]
        for j, tau in enumerate(grid):
            keep = flags & (prob >= tau)
            tp[lvl, j] = np.sum(keep & (target == 1))
            fp[lvl, j] = np.sum(keep & (target == 0))
    return tp, fp


def select_global(y, agent_flags, p_cal, floor):
    """Lowest FPR with recall >= floor; ties by higher F1 then lower tau."""

    best = None
    pos, neg = y.sum(), (1 - y).sum()
    for tau in tau_grid(include_zero=False):
        flags = agent_flags & (p_cal >= tau)
        tp = np.sum(flags & (y == 1))
        fp = np.sum(flags & (y == 0))
        recall = tp / pos
        if recall < floor:
            continue
        fpr = fp / neg
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
        key = (fpr, -f1, tau)
        if best is None or key < best[0]:
            best = (key, float(tau))
    return best[1] if best else None


def select_motion_aware(y, agent_flags, p_cal, levels, floor):
    """Exhaustive search over one threshold per motion level (92^3 combinations, vectorised)."""

    grid = tau_grid(include_zero=True)
    tp, fp = _level_counts(y, agent_flags, p_cal, levels, grid)
    pos, neg = y.sum(), (1 - y).sum()
    TP = tp[0][:, None, None] + tp[1][None, :, None] + tp[2][None, None, :]
    FP = fp[0][:, None, None] + fp[1][None, :, None] + fp[2][None, None, :]
    recall = TP / pos
    fpr = FP / neg
    prec = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) > 0)
    f1 = np.divide(2 * prec * recall, prec + recall, out=np.zeros_like(TP), where=(prec + recall) > 0)
    tau_sum = grid[:, None, None] + grid[None, :, None] + grid[None, None, :]
    ok = recall >= floor
    if not ok.any():
        return None
    order = np.lexsort((tau_sum[ok], -f1[ok], fpr[ok]))
    idx = np.argwhere(ok)[order[0]]
    return tuple(float(grid[i]) for i in idx)


def per_level_metrics(y, flagged, investigations, levels):
    rows = []
    for lvl, name in enumerate(LEVELS):
        m = levels == lvl
        rows.append({"motion_level": name, "unreliable_rate": float(np.mean(y[m])),
                     **metrics(y[m], flagged[m], investigations[m])})
    return pd.DataFrame(rows)

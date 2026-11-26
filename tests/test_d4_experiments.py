"""Tests for the D4 supplementary experiments (motion-aware veto, bootstrap, usability scoring)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.experiments.common import RESULTS_DIR, TEST_SUBJECTS, VALIDATION_SUBJECTS
from src.experiments.motion_aware import (
    apply_veto,
    assign_levels,
    motion_edges,
    select_global,
    select_motion_aware,
    tau_grid,
)
from src.experiments.run_bootstrap import block_indices
from src.experiments.score_usability import score, sus_score

PHASE1 = RESULTS_DIR / "d4_motion_thresholds"


def test_tau_grid_includes_no_veto_option():
    assert tau_grid(True)[0] == 0.0
    assert tau_grid(False)[0] == 0.05
    assert len(tau_grid(True)) == len(tau_grid(False)) + 1 == 92


def test_motion_levels_use_tertiles():
    motion = np.arange(300, dtype=float)
    levels = assign_levels(motion, motion_edges(motion))
    assert np.bincount(levels).tolist() == [100, 100, 100]


def test_uniform_motion_thresholds_equal_global_veto():
    rng = np.random.default_rng(0)
    flags, p, levels = rng.integers(0, 2, 500), rng.random(500), rng.integers(0, 3, 500)
    assert (apply_veto(flags, p, levels, (0.3, 0.3, 0.3)) == apply_veto(flags, p, levels, 0.3)).all()


def test_veto_only_removes_flags():
    rng = np.random.default_rng(1)
    flags, p, levels = rng.integers(0, 2, 500), rng.random(500), rng.integers(0, 3, 500)
    vetoed = apply_veto(flags, p, levels, (0.9, 0.1, 0.5))
    assert ((vetoed == 1) <= (flags == 1)).all()


def test_motion_aware_search_never_worse_than_global_on_selection_data():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.4 * y + 0.6 * rng.random(2000), 0, 1)
    flags = (rng.random(2000) < 0.6).astype(int) | y
    levels = rng.integers(0, 3, 2000)
    floor = 0.8 * ((flags == 1) & (y == 1)).sum() / y.sum()
    g = select_global(y, flags, p, floor)
    m = select_motion_aware(y, flags, p, levels, floor)
    fpr = lambda f: ((f == 1) & (y == 0)).sum() / (y == 0).sum()
    assert fpr(apply_veto(flags, p, levels, m)) <= fpr(apply_veto(flags, p, levels, g)) + 1e-12


def test_saved_selection_used_validation_only():
    sel = json.loads((PHASE1 / "selection.json").read_text())
    val_motion = pd.read_csv(PHASE1 / "motion_intensity_validation.csv")
    assert set(val_motion.subject) == set(VALIDATION_SUBJECTS)
    assert np.allclose(sel["motion_edges_g"], motion_edges(val_motion["motion_intensity_g"]))
    assert set(pd.read_csv(PHASE1 / "motion_intensity_test.csv").subject) == set(TEST_SUBJECTS)


def test_global_veto_reproduces_d3_result():
    sel = json.loads((PHASE1 / "selection.json").read_text())
    assert sel["global_veto_tau"] == 0.26
    d4 = pd.read_csv(PHASE1 / "test_comparison.csv").set_index("candidate").loc["Global calibrated veto"]
    d3 = pd.read_csv(RESULTS_DIR / "phase2_fp_reduction" / "test_comparison.csv").set_index("variant").loc["tau=0.26"]
    assert np.isclose(d4.recall, d3.recall) and np.isclose(d4.false_positive_rate, d3.false_positive_rate)


def test_block_bootstrap_stays_within_each_participant():
    rng = np.random.default_rng(42)
    idx = block_indices([100, 50], 30, 200, rng)
    assert idx.shape == (200, 150)
    assert idx[:, :100].max() < 100 and idx[:, 100:].min() >= 100 and idx.max() < 150


def test_block_bootstrap_is_deterministic():
    a = block_indices([80, 60], 15, 10, np.random.default_rng(7))
    b = block_indices([80, 60], 15, 10, np.random.default_rng(7))
    assert (a == b).all()


@pytest.mark.parametrize("answer,expected", [(3, 50.0)])
def test_sus_neutral_answers_score_50(answer, expected):
    assert sus_score({f"sus{i}": answer for i in range(1, 11)}) == expected


def test_sus_extremes():
    best = {f"sus{i}": (5 if i % 2 else 1) for i in range(1, 11)}
    worst = {f"sus{i}": (1 if i % 2 else 5) for i in range(1, 11)}
    assert sus_score(best) == 100.0 and sus_score(worst) == 0.0


def test_usability_summary_and_validation():
    row = {"participant": "P1", **{f"sus{i}": 3 for i in range(1, 11)}, "t1": "yes", "t2": "partly", "t3": "no",
           "t4": "yes", "q1": 4, "q2": 3, "q3": 5, "q4": 4, "comments": ""}
    _, summary = score(pd.DataFrame([row]))
    assert summary["sus_mean"] == 50.0 and summary["t2_success_rate"] == 0.5
    with pytest.raises(ValueError):
        score(pd.DataFrame([{**row, "t1": "maybe"}]))

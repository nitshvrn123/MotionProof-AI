"""Regression tests for Phase 1 calibration utilities and saved results."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    calibration_metrics,
    expected_calibration_error,
    leave_one_subject_out,
    reliability_table,
)

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "results" / "experiments" / "phase1_calibration"


def test_ece_is_zero_for_perfectly_calibrated_bins():
    p = np.array([0.2] * 10 + [0.8] * 10)
    y = np.array([1, 1] + [0] * 8 + [1] * 8 + [0, 0])
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0)


def test_ece_matches_hand_computation():
    p = np.array([0.9, 0.9, 0.1, 0.1])
    y = np.array([0, 0, 0, 0])
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.5)


def test_reliability_table_counts_every_window():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=500)
    y = rng.integers(0, 2, size=500)
    assert sum(r["count"] for r in reliability_table(y, p, 15)) == 500


def test_platt_recovers_known_logistic_map():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.01, 0.99, size=20000)
    true_p = 1 / (1 + np.exp(-(0.5 * np.log(raw / (1 - raw)) - 0.5)))
    y = rng.uniform(size=raw.size) < true_p
    cal = PlattCalibrator().fit(raw, y)
    assert cal.coef_ == pytest.approx(0.5, abs=0.05)
    assert cal.intercept_ == pytest.approx(-0.5, abs=0.05)


def test_platt_preserves_ranking_and_isotonic_is_monotone():
    rng = np.random.default_rng(2)
    p = rng.uniform(size=2000)
    y = rng.uniform(size=2000) < p ** 2
    platt = PlattCalibrator().fit(p, y)
    iso = IsotonicCalibrator().fit(p, y)
    order = np.argsort(p)
    assert np.all(np.diff(platt.predict(p[order])) >= -1e-12)
    assert np.all(np.diff(iso.predict(p[order])) >= -1e-12)
    m = calibration_metrics(y, platt.predict(p))
    assert m["roc_auc"] == pytest.approx(calibration_metrics(y, p)["roc_auc"])


def test_leave_one_subject_out_never_scores_on_fitting_subject():
    frame = pd.DataFrame({"subject": [11] * 50 + [12] * 50,
                          "p": np.linspace(0.01, 0.99, 100),
                          "y": [0, 1] * 50})
    preds, folds = leave_one_subject_out(frame, "platt", "p", "y")
    assert {f["held_out_subject"] for f in folds} == {11, 12}
    assert len(preds) == 100 and np.isfinite(preds).all()


@pytest.mark.skipif(not (PHASE1 / "selection.json").exists(), reason="phase 1 not run")
def test_saved_phase1_results_are_consistent():
    pooled = pd.read_csv(PHASE1 / "validation_selection_pooled.csv")
    test = pd.read_csv(PHASE1 / "test_calibration_metrics.csv")
    selected = pd.read_json(PHASE1 / "selection.json", typ="series")["selected_method"]
    assert selected == pooled.sort_values(["brier", "ece"]).iloc[0]["method"]
    assert set(test["method"]) == {"original", "platt", "isotonic"}
    assert (test["n"] == 13007).all()
    original = test.set_index("method").loc["original"]
    frozen = pd.read_csv(ROOT / "results/metrics/final_test/test_policy_inputs_nonlinear.csv")
    assert original["mean_predicted"] == pytest.approx(frozen["unreliable_probability"].mean())

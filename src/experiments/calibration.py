"""Probability calibration utilities for the reliability classifier.

Calibrators are fitted only on validation participants (S11, S12). The
original classifier checkpoint is never modified: a calibrator is a separate
post-hoc map from the classifier's probability to a calibrated probability.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def _clip(p):
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)


def logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


def _bin_index(p, n_bins):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1), edges


def expected_calibration_error(y, p, n_bins=15):
    """Equal-width ECE: sum over bins of (n_b / N) * |observed_b - predicted_b|."""

    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("y and p must be 1-D arrays of equal length.")

    bins, _ = _bin_index(p, n_bins)
    ece = 0.0

    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())

    return float(ece)


def reliability_table(y, p, n_bins=15):
    """Per-bin counts, mean predicted probability and observed failure rate."""

    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    bins, edges = _bin_index(p, n_bins)
    rows = []

    for b in range(n_bins):
        mask = bins == b
        rows.append({
            "bin": b,
            "bin_lower": float(edges[b]),
            "bin_upper": float(edges[b + 1]),
            "count": int(mask.sum()),
            "mean_predicted": float(p[mask].mean()) if mask.any() else np.nan,
            "observed_rate": float(y[mask].mean()) if mask.any() else np.nan,
        })

    return rows


def calibration_metrics(y, p, n_bins=15):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=np.float64)
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "ece": expected_calibration_error(y, p, n_bins),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, _clip(p), labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
    }


@dataclass
class IdentityCalibrator:
    name: str = "original"

    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.asarray(p, dtype=np.float64)

    def describe(self):
        return {"method": self.name}


@dataclass
class PlattCalibrator:
    """Logistic regression on the logit of the original probability."""

    name: str = "platt"
    coef_: float = None
    intercept_: float = None

    def fit(self, p, y):
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(logit(p).reshape(-1, 1), np.asarray(y, dtype=int))
        self.coef_ = float(model.coef_[0, 0])
        self.intercept_ = float(model.intercept_[0])
        return self

    def predict(self, p):
        z = self.coef_ * logit(p) + self.intercept_
        return 1.0 / (1.0 + np.exp(-z))

    def describe(self):
        return {"method": self.name, "coef": self.coef_, "intercept": self.intercept_}


@dataclass
class IsotonicCalibrator:
    name: str = "isotonic"
    model_: IsotonicRegression = field(default=None, repr=False)

    def fit(self, p, y):
        self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.model_.fit(np.asarray(p, dtype=np.float64), np.asarray(y, dtype=float))
        return self

    def predict(self, p):
        return self.model_.predict(np.asarray(p, dtype=np.float64))

    def describe(self):
        return {"method": self.name, "breakpoints": int(len(self.model_.X_thresholds_))}


CALIBRATORS = {
    "original": IdentityCalibrator,
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
}


def make_calibrator(name):
    return CALIBRATORS[name]()


def leave_one_subject_out(frame, method, prob_col, target_col, subject_col="subject", n_bins=15):
    """Fit on all-but-one validation subject and score the held-out one.

    Returns pooled out-of-subject predictions and per-fold metrics.
    """

    preds = np.empty(len(frame), dtype=np.float64)
    folds = []

    for subject in sorted(frame[subject_col].unique()):
        held = (frame[subject_col] == subject).to_numpy()
        cal = make_calibrator(method).fit(
            frame.loc[~held, prob_col].to_numpy(), frame.loc[~held, target_col].to_numpy()
        )
        preds[held] = cal.predict(frame.loc[held, prob_col].to_numpy())
        metrics = calibration_metrics(frame.loc[held, target_col], preds[held], n_bins)
        folds.append({"method": method, "held_out_subject": int(subject), **metrics})

    return preds, folds

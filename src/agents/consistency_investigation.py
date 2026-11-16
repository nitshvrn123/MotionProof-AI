"""Causal heart-rate consistency investigation for MotionProof AI.

Compares the current CNN prediction against earlier predictions
from the same participant.

Never uses future predictions, ECG reference HR, or error labels.
"""

from dataclasses import dataclass, asdict

import numpy as np


MAX_HISTORY_WINDOWS = 5
MIN_HISTORY_WINDOWS = 2


@dataclass(frozen=True)
class ConsistencyInvestigationResult:
    history_available: bool
    history_count: int
    previous_hr_bpm: float
    historical_median_hr_bpm: float
    hr_jump_bpm: float
    median_deviation_bpm: float
    historical_std_bpm: float

    def to_dict(self):
        return asdict(self)


def investigate_consistency(
    current_hr,
    previous_hr_predictions,
    max_history=MAX_HISTORY_WINDOWS,
):
    """Investigate consistency using previous CNN predictions.

    Parameters
    ----------
    current_hr:
        Current CNN heart-rate prediction in BPM.

    previous_hr_predictions:
        Chronologically ordered predictions from earlier windows
        belonging to the same participant.

    max_history:
        Maximum number of recent predictions to examine.

    Returns
    -------
    ConsistencyInvestigationResult
        Causal temporal-consistency measurements.
    """

    current_hr = float(current_hr)

    history = np.asarray(
        previous_hr_predictions,
        dtype=np.float64,
    )

    if history.ndim != 1:
        raise ValueError(
            "Previous HR predictions must be one-dimensional."
        )

    if not np.isfinite(current_hr):
        raise ValueError(
            "Current HR must be finite."
        )

    if not np.isfinite(history).all():
        raise ValueError(
            "Previous HR predictions contain nonfinite values."
        )

    if isinstance(max_history, (bool, np.bool_)) or not isinstance(
        max_history, (int, np.integer)
    ) or max_history < 1:
        raise ValueError(
            "max_history must be a positive integer."
        )

    # Only the most recent previous predictions are retained.
    recent_history = history[-max_history:]

    history_count = len(recent_history)

    # No previous prediction is available.
    if history_count == 0:
        return ConsistencyInvestigationResult(
            history_available=False,
            history_count=0,
            previous_hr_bpm=float("nan"),
            historical_median_hr_bpm=float("nan"),
            hr_jump_bpm=float("nan"),
            median_deviation_bpm=float("nan"),
            historical_std_bpm=float("nan"),
        )

    previous_hr = float(
        recent_history[-1]
    )

    historical_median = float(
        np.median(recent_history)
    )

    hr_jump = abs(
        current_hr - previous_hr
    )

    median_deviation = abs(
        current_hr - historical_median
    )

    historical_std = float(
        np.std(recent_history)
    )

    return ConsistencyInvestigationResult(
        history_available=(
            history_count >= MIN_HISTORY_WINDOWS
        ),
        history_count=history_count,
        previous_hr_bpm=previous_hr,
        historical_median_hr_bpm=historical_median,
        hr_jump_bpm=float(hr_jump),
        median_deviation_bpm=float(median_deviation),
        historical_std_bpm=historical_std,
    )

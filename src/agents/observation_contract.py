"""Shared policy observation contract for MotionProof AI.

Defines which information is available before and after
an investigation.

Both fixed-rule and learned policies must follow this contract.
"""

from dataclasses import dataclass

import numpy as np

from src.agents.investigation_actions import InvestigationAction


INITIAL_OBSERVATION_FEATURES = (
    "cnn_predicted_hr",
    "unreliable_probability",
)

INVESTIGATION_RESULT_FEATURES = {
    InvestigationAction.CHECK_PPG: (
        "dominant_frequency_hz",
        "spectral_hr_bpm",
        "spectral_concentration",
        "spectral_entropy",
        "autocorrelation_strength",
        "cnn_predicted_hr_bpm",
        "cnn_spectral_disagreement_bpm",
    ),
    InvestigationAction.CHECK_MOTION: (
        "motion_intensity",
        "magnitude_mean",
        "magnitude_range",
        "acc_x_std",
        "acc_y_std",
        "acc_z_std",
        "dominant_motion_frequency_hz",
        "motion_spectral_concentration",
    ),
    InvestigationAction.CHECK_CONSISTENCY: (
        "history_available",
        "history_count",
        "previous_hr_bpm",
        "historical_median_hr_bpm",
        "hr_jump_bpm",
        "median_deviation_bpm",
        "historical_std_bpm",
    ),
}


@dataclass(frozen=True)
class PolicyObservation:
    cnn_predicted_hr: float
    unreliable_probability: float
    remaining_budget: int
    completed_actions: tuple
    investigation_results: dict


def build_policy_observation(environment):
    """Construct the information visible to a policy."""

    if environment.remaining_budget < 0:
        raise ValueError("Negative remaining budget.")

    visible_results = {}

    for action in (
        InvestigationAction.CHECK_PPG,
        InvestigationAction.CHECK_MOTION,
        InvestigationAction.CHECK_CONSISTENCY,
    ):
        if action.value not in environment.results:
            continue

        result = environment.results[action.value]

        allowed_features = INVESTIGATION_RESULT_FEATURES[action]

        visible_results[action.value] = {
            name: result[name]
            for name in allowed_features
        }

    return PolicyObservation(
        cnn_predicted_hr=float(
            environment.observation.cnn_predicted_hr
        ),
        unreliable_probability=float(
            environment.observation.unreliable_probability
        ),
        remaining_budget=int(
            environment.remaining_budget
        ),
        completed_actions=tuple(
            environment.action_history
        ),
        investigation_results=visible_results,
    )


def validate_initial_observation(observation):
    """Verify that initial observations contain finite values."""

    values = np.asarray(
        [
            observation.cnn_predicted_hr,
            observation.unreliable_probability,
        ],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Initial observation contains nonfinite values."
        )

    if not 0 <= observation.unreliable_probability <= 1:
        raise ValueError(
            "Reliability probability must be in [0, 1]."
        )

    return True

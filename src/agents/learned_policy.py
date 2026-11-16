"""Learned adaptive investigation policy components for MotionProof AI.

Step 50.13G:
    State encoding and valid-action masking.

The policy receives only the shared observation contract.
Ground-truth labels and ECG reference HR are never policy inputs.
"""

import numpy as np

from src.agents.investigation_actions import InvestigationAction
from src.agents.observation_contract import (
    INVESTIGATION_RESULT_FEATURES,
    build_policy_observation,
    validate_initial_observation,
)


ACTION_ORDER = tuple(InvestigationAction)

INVESTIGATION_ORDER = (
    InvestigationAction.CHECK_PPG,
    InvestigationAction.CHECK_MOTION,
    InvestigationAction.CHECK_CONSISTENCY,
)


def get_state_feature_names():
    """Return the fixed ordering of policy state features."""

    names = [
        "cnn_predicted_hr",
        "unreliable_probability",
        "remaining_budget",
    ]

    for action in INVESTIGATION_ORDER:
        prefix = action.value.lower()

        names.append(f"{prefix}_completed")

        for feature in INVESTIGATION_RESULT_FEATURES[action]:
            names.append(f"{prefix}_{feature}")

    return tuple(names)


STATE_FEATURE_NAMES = get_state_feature_names()
STATE_DIM = len(STATE_FEATURE_NAMES)
ACTION_DIM = len(ACTION_ORDER)


def encode_policy_state(environment):
    """Encode the permitted policy observation into a finite vector."""

    observation = build_policy_observation(environment)
    validate_initial_observation(observation)

    features = [
        float(observation.cnn_predicted_hr),
        float(observation.unreliable_probability),
        float(observation.remaining_budget),
    ]

    for action in INVESTIGATION_ORDER:
        result = observation.investigation_results.get(action.value)

        completed = result is not None
        features.append(float(completed))

        for feature_name in INVESTIGATION_RESULT_FEATURES[action]:
            if not completed:
                features.append(0.0)
                continue

            value = result[feature_name]

            if isinstance(value, (bool, np.bool_)):
                features.append(float(value))
                continue

            value = float(value)

            if np.isnan(value):
                if (
                    action == InvestigationAction.CHECK_CONSISTENCY
                    and not result["history_available"]
                ):
                    features.append(0.0)
                    continue

                raise ValueError(
                    f"Unexpected NaN in {action.value}: "
                    f"{feature_name}"
                )

            if not np.isfinite(value):
                raise ValueError(
                    f"Nonfinite feature in {action.value}: "
                    f"{feature_name}"
                )

            features.append(value)

    state = np.asarray(features, dtype=np.float32)

    if state.shape != (STATE_DIM,):
        raise RuntimeError("Unexpected policy state dimensions.")

    if not np.isfinite(state).all():
        raise ValueError("Policy state contains nonfinite values.")

    return state


def build_action_mask(environment):
    """Return a boolean mask of actions allowed by the environment."""

    available = set(environment.available_actions())

    mask = np.asarray(
        [action in available for action in ACTION_ORDER],
        dtype=np.bool_,
    )

    if mask.shape != (ACTION_DIM,):
        raise RuntimeError("Unexpected action mask dimensions.")

    return mask


def mask_action_scores(scores, action_mask):
    """Assign negative infinity to unavailable actions."""

    scores = np.asarray(scores, dtype=np.float64)
    action_mask = np.asarray(action_mask, dtype=np.bool_)

    if scores.shape != (ACTION_DIM,):
        raise ValueError("Incorrect action-score dimensions.")

    if action_mask.shape != (ACTION_DIM,):
        raise ValueError("Incorrect action-mask dimensions.")

    if not np.isfinite(scores).all():
        raise ValueError("Action scores must be finite.")

    if not action_mask.any():
        raise ValueError("No available actions.")

    masked = scores.copy()
    masked[~action_mask] = -np.inf

    return masked

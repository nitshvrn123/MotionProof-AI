import numpy as np
import pytest

from src.agents.investigation_actions import InvestigationAction
from src.agents.investigation_environment import InvestigationEnvironment
from src.agents.learned_policy import (
    ACTION_ORDER,
    ACTION_DIM,
    STATE_DIM,
    STATE_FEATURE_NAMES,
    encode_policy_state,
    build_action_mask,
    mask_action_scores,
)


def make_environment(budget=2, history=None):
    rng = np.random.default_rng(42)

    t = np.arange(512) / 64.0
    ppg = np.sin(2 * np.pi * 1.2 * t)

    acc = rng.normal(size=(256, 3))

    if history is None:
        history = []

    return InvestigationEnvironment(
        ppg_window=ppg,
        acc_window=acc,
        cnn_predicted_hr=72.0,
        unreliable_probability=0.45,
        previous_hr_predictions=history,
        budget=budget,
    )


def test_state_dimensions_and_finiteness():
    env = make_environment()

    state = encode_policy_state(env)

    assert state.shape == (STATE_DIM,)
    assert STATE_DIM == 28
    assert np.isfinite(state).all()


def test_initial_investigation_features_hidden():
    env = make_environment()

    state = encode_policy_state(env)

    for action in (
        "check_ppg",
        "check_motion",
        "check_consistency",
    ):
        index = STATE_FEATURE_NAMES.index(
            f"{action}_completed"
        )

        assert state[index] == 0


def test_ppg_investigation_updates_state():
    env = make_environment()

    before = encode_policy_state(env)

    env.step(InvestigationAction.CHECK_PPG)

    after = encode_policy_state(env)

    assert not np.array_equal(before, after)

    index = STATE_FEATURE_NAMES.index(
        "check_ppg_completed"
    )

    assert after[index] == 1
    assert np.isfinite(after).all()


def test_repeated_action_is_masked():
    env = make_environment()

    env.step(InvestigationAction.CHECK_PPG)

    mask = build_action_mask(env)

    index = ACTION_ORDER.index(
        InvestigationAction.CHECK_PPG
    )

    assert not mask[index]


def test_budget_zero_masks_investigations():
    env = make_environment(budget=0)

    mask = build_action_mask(env)

    assert mask[
        ACTION_ORDER.index(InvestigationAction.ACCEPT)
    ]

    assert mask[
        ACTION_ORDER.index(
            InvestigationAction.FLAG_UNRELIABLE
        )
    ]

    for action in (
        InvestigationAction.CHECK_PPG,
        InvestigationAction.CHECK_MOTION,
        InvestigationAction.CHECK_CONSISTENCY,
    ):
        assert not mask[ACTION_ORDER.index(action)]


def test_missing_history_produces_finite_state():
    env = make_environment(history=[])

    env.step(InvestigationAction.CHECK_CONSISTENCY)

    state = encode_policy_state(env)

    assert np.isfinite(state).all()

    index = STATE_FEATURE_NAMES.index(
        "check_consistency_history_available"
    )

    assert state[index] == 0


def test_available_history_is_encoded():
    env = make_environment(
        history=[68.0, 70.0, 71.0]
    )

    env.step(InvestigationAction.CHECK_CONSISTENCY)

    state = encode_policy_state(env)

    index = STATE_FEATURE_NAMES.index(
        "check_consistency_history_available"
    )

    assert state[index] == 1
    assert np.isfinite(state).all()


def test_mask_action_scores():
    scores = np.arange(ACTION_DIM, dtype=float)

    mask = np.array(
        [True, True, False, True, False]
    )

    masked = mask_action_scores(scores, mask)

    assert np.isneginf(masked[2])
    assert np.isneginf(masked[4])

    assert np.isfinite(masked[0])
    assert np.isfinite(masked[1])
    assert np.isfinite(masked[3])


def test_terminated_environment_masks_all_actions():
    env = make_environment()

    env.step(InvestigationAction.ACCEPT)

    mask = build_action_mask(env)

    assert not mask.any()


def test_environment_rejects_repeated_investigation():
    env = make_environment()

    env.step(InvestigationAction.CHECK_PPG)

    with pytest.raises(ValueError):
        env.step(InvestigationAction.CHECK_PPG)

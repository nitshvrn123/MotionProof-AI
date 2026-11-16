"""Offline reward calculation for investigation policies.

Ground-truth labels are used only after the policy terminates.
They must never be included in policy observations.
"""

from dataclasses import dataclass

import numpy as np

from src.agents.investigation_actions import (
    InvestigationAction,
    TERMINAL_ACTIONS,
)


@dataclass(frozen=True)
class InvestigationRewardConfig:
    true_positive_reward: float = 2.0
    false_positive_penalty: float = -1.5
    false_negative_penalty: float = -2.0
    true_negative_reward: float = 0.0
    investigation_cost: float = 0.1


def calculate_investigation_reward(
    final_decision,
    unreliable_target,
    investigations_used,
    config=None,
):
    """Calculate reward from a completed investigation.

    Parameters
    ----------
    final_decision:
        ACCEPT or FLAG_UNRELIABLE.

    unreliable_target:
        Offline ground-truth label.
        1 means CNN absolute error > 10 BPM.
        0 means CNN absolute error <= 10 BPM.

    investigations_used:
        Number of investigation budget units consumed.

    config:
        Optional reward configuration.
    """

    config = config or InvestigationRewardConfig()

    decision = InvestigationAction(final_decision)

    if decision not in TERMINAL_ACTIONS:
        raise ValueError(
            "Reward requires a terminal decision."
        )

    if isinstance(unreliable_target, (bool, np.bool_)):
        target = int(unreliable_target)
    elif (
        isinstance(unreliable_target, (int, np.integer))
        and unreliable_target in (0, 1)
    ):
        target = int(unreliable_target)
    else:
        raise ValueError(
            "unreliable_target must be 0 or 1."
        )

    if (
        isinstance(investigations_used, (bool, np.bool_))
        or not isinstance(
            investigations_used, (int, np.integer)
        )
        or investigations_used < 0
    ):
        raise ValueError(
            "investigations_used must be a nonnegative integer."
        )

    if decision == InvestigationAction.FLAG_UNRELIABLE:
        if target == 1:
            decision_reward = config.true_positive_reward
        else:
            decision_reward = config.false_positive_penalty
    else:
        if target == 1:
            decision_reward = config.false_negative_penalty
        else:
            decision_reward = config.true_negative_reward

    investigation_penalty = (
        config.investigation_cost * investigations_used
    )

    total_reward = (
        decision_reward - investigation_penalty
    )

    return {
        "final_decision": decision.value,
        "unreliable_target": target,
        "decision_reward": float(decision_reward),
        "investigation_penalty": float(investigation_penalty),
        "total_reward": float(total_reward),
    }

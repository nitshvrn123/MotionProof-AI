"""Shared investigation environment for MotionProof AI.

Both fixed-rule and learned policies use this environment.

The environment enforces per-window investigation budgets.
It never exposes ECG reference HR or error labels.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.agents.investigation_actions import (
    InvestigationAction,
    TERMINAL_ACTIONS,
    get_action_cost,
)

from src.agents.ppg_investigation import investigate_ppg
from src.agents.motion_investigation import investigate_motion
from src.agents.consistency_investigation import investigate_consistency


@dataclass(frozen=True)
class InvestigationObservation:
    cnn_predicted_hr: float
    unreliable_probability: float


class InvestigationEnvironment:

    def __init__(
        self,
        ppg_window,
        acc_window,
        cnn_predicted_hr,
        unreliable_probability,
        previous_hr_predictions,
        budget=2,
    ):
        self._ppg = np.asarray(
            ppg_window, dtype=np.float64
        ).copy()

        self._acc = np.asarray(
            acc_window, dtype=np.float64
        ).copy()

        self._history = np.asarray(
            previous_hr_predictions, dtype=np.float64
        ).copy()

        if self._ppg.shape != (512,):
            raise ValueError("Expected PPG shape (512,).")

        if self._acc.shape != (256, 3):
            raise ValueError("Expected ACC shape (256, 3).")

        if self._history.ndim != 1:
            raise ValueError("History must be one-dimensional.")

        if not np.isfinite(self._ppg).all():
            raise ValueError("Nonfinite PPG input.")

        if not np.isfinite(self._acc).all():
            raise ValueError("Nonfinite ACC input.")

        if not np.isfinite(self._history).all():
            raise ValueError("Nonfinite history.")

        if not np.isfinite(cnn_predicted_hr):
            raise ValueError("Nonfinite CNN prediction.")

        if not np.isfinite(unreliable_probability):
            raise ValueError("Nonfinite reliability probability.")

        if not 0 <= unreliable_probability <= 1:
            raise ValueError(
                "Reliability probability must be between 0 and 1."
            )

        if isinstance(budget, (bool, np.bool_)) or not isinstance(
            budget, (int, np.integer)
        ) or budget < 0:
            raise ValueError(
                "Budget must be a nonnegative integer."
            )

        self.initial_budget = int(budget)
        self.remaining_budget = int(budget)

        self.observation = InvestigationObservation(
            cnn_predicted_hr=float(cnn_predicted_hr),
            unreliable_probability=float(unreliable_probability),
        )

        self.results: dict[str, dict[str, Any]] = {}
        self.action_history: list[str] = []

        self.terminated = False
        self.final_decision = None

    def available_actions(self):
        """Return actions permitted in the current state."""

        if self.terminated:
            return []

        actions = [
            InvestigationAction.ACCEPT,
            InvestigationAction.FLAG_UNRELIABLE,
        ]

        for action in (
            InvestigationAction.CHECK_PPG,
            InvestigationAction.CHECK_MOTION,
            InvestigationAction.CHECK_CONSISTENCY,
        ):
            if (
                action.value not in self.results
                and get_action_cost(action) <= self.remaining_budget
            ):
                actions.append(action)

        return actions

    def step(self, action):
        """Execute an investigation or terminal decision."""

        if self.terminated:
            raise RuntimeError(
                "Investigation has already terminated."
            )

        action = InvestigationAction(action)

        if action not in self.available_actions():
            raise ValueError(
                f"Action {action.value} is unavailable."
            )

        if action in TERMINAL_ACTIONS:
            self.terminated = True
            self.final_decision = action.value
            self.action_history.append(action.value)

            return {
                "action": action.value,
                "cost": 0,
                "result": None,
                "remaining_budget": self.remaining_budget,
                "terminated": True,
                "final_decision": self.final_decision,
            }

        if action == InvestigationAction.CHECK_PPG:
            result = investigate_ppg(
                self._ppg,
                self.observation.cnn_predicted_hr,
            ).to_dict()

        elif action == InvestigationAction.CHECK_MOTION:
            result = investigate_motion(
                self._acc
            ).to_dict()

        elif action == InvestigationAction.CHECK_CONSISTENCY:
            result = investigate_consistency(
                self.observation.cnn_predicted_hr,
                self._history,
            ).to_dict()

        else:
            raise ValueError(
                "Unsupported investigation action."
            )

        cost = get_action_cost(action)

        self.remaining_budget -= cost

        self.results[action.value] = result
        self.action_history.append(action.value)

        return {
            "action": action.value,
            "cost": cost,
            "result": result,
            "remaining_budget": self.remaining_budget,
            "terminated": False,
            "final_decision": None,
        }

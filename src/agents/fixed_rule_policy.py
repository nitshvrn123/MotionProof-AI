"""Fixed-rule investigation policy for MotionProof AI.

Uses predetermined decision rules rather than learned actions.

The policy receives only the shared observation contract.
It does not access ECG reference HR or ground-truth error labels.
"""

from dataclasses import dataclass

from src.agents.investigation_actions import (
    InvestigationAction,
)

from src.agents.observation_contract import (
    build_policy_observation,
    validate_initial_observation,
)


@dataclass(frozen=True)
class FixedRulePolicyConfig:
    investigation_threshold: float = 0.30
    flag_probability_threshold: float = 0.60
    disagreement_threshold_bpm: float = 15.0


class FixedRuleInvestigationPolicy:
    """Deterministic baseline with investigation and final decisions."""

    def __init__(self, config=None):
        self.config = config or FixedRulePolicyConfig()

    def select_action(self, environment):
        """Choose the next action using the shared observation contract."""

        if environment.terminated:
            raise RuntimeError(
                "Environment already terminated."
            )

        observation = build_policy_observation(environment)

        validate_initial_observation(observation)

        available = environment.available_actions()

        probability = observation.unreliable_probability

        ppg_result = observation.investigation_results.get(
            InvestigationAction.CHECK_PPG.value
        )

        # -----------------------------------------
        # 1. Low-risk predictions
        # -----------------------------------------

        if probability < self.config.investigation_threshold:
            return InvestigationAction.ACCEPT

        # -----------------------------------------
        # 2. First investigation: PPG
        # -----------------------------------------

        if (
            ppg_result is None
            and InvestigationAction.CHECK_PPG in available
        ):
            return InvestigationAction.CHECK_PPG

        # -----------------------------------------
        # 3. Second investigation
        # -----------------------------------------

        if ppg_result is not None:
            disagreement = ppg_result[
                "cnn_spectral_disagreement_bpm"
            ]

            if (
                disagreement >= self.config.disagreement_threshold_bpm
                and InvestigationAction.CHECK_MOTION in available
            ):
                return InvestigationAction.CHECK_MOTION

            if (
                disagreement < self.config.disagreement_threshold_bpm
                and InvestigationAction.CHECK_CONSISTENCY in available
            ):
                return InvestigationAction.CHECK_CONSISTENCY

        # -----------------------------------------
        # 4. Final reliability decision
        # -----------------------------------------

        if probability >= self.config.flag_probability_threshold:
            return InvestigationAction.FLAG_UNRELIABLE

        if ppg_result is not None:
            disagreement = ppg_result[
                "cnn_spectral_disagreement_bpm"
            ]

            if disagreement >= self.config.disagreement_threshold_bpm:
                return InvestigationAction.FLAG_UNRELIABLE

        return InvestigationAction.ACCEPT


def run_fixed_rule_policy(environment, policy=None):
    """Execute the fixed-rule policy until a terminal decision."""

    policy = policy or FixedRuleInvestigationPolicy()

    while not environment.terminated:
        action = policy.select_action(environment)
        environment.step(action)

    return {
        "cnn_predicted_hr": (
            environment.observation.cnn_predicted_hr
        ),
        "unreliable_probability": (
            environment.observation.unreliable_probability
        ),
        "final_decision": environment.final_decision,
        "actions": list(environment.action_history),
        "investigations_used": (
            environment.initial_budget
            - environment.remaining_budget
        ),
        "remaining_budget": environment.remaining_budget,
        "investigation_results": dict(environment.results),
    }

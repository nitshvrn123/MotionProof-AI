"""Action definitions and investigation budget rules."""

from dataclasses import dataclass
from enum import Enum


class InvestigationAction(str, Enum):
    ACCEPT = "ACCEPT"
    FLAG_UNRELIABLE = "FLAG_UNRELIABLE"
    CHECK_PPG = "CHECK_PPG"
    CHECK_MOTION = "CHECK_MOTION"
    CHECK_CONSISTENCY = "CHECK_CONSISTENCY"


@dataclass(frozen=True)
class ActionDefinition:
    name: InvestigationAction
    cost: int
    description: str


ACTION_DEFINITIONS = {
    InvestigationAction.ACCEPT: ActionDefinition(
        InvestigationAction.ACCEPT,
        0,
        "Accept the current CNN heart-rate estimate.",
    ),
    InvestigationAction.FLAG_UNRELIABLE: ActionDefinition(
        InvestigationAction.FLAG_UNRELIABLE,
        0,
        "Flag the current CNN estimate as potentially unreliable.",
    ),
    InvestigationAction.CHECK_PPG: ActionDefinition(
        InvestigationAction.CHECK_PPG,
        1,
        "Inspect PPG periodicity and signal quality.",
    ),
    InvestigationAction.CHECK_MOTION: ActionDefinition(
        InvestigationAction.CHECK_MOTION,
        1,
        "Inspect accelerometer motion and artifacts.",
    ),
    InvestigationAction.CHECK_CONSISTENCY: ActionDefinition(
        InvestigationAction.CHECK_CONSISTENCY,
        1,
        "Compare with earlier heart-rate estimates.",
    ),
}


TERMINAL_ACTIONS = frozenset({
    InvestigationAction.ACCEPT,
    InvestigationAction.FLAG_UNRELIABLE,
})


def get_action_cost(action):
    action = InvestigationAction(action)
    return ACTION_DEFINITIONS[action].cost


def validate_action_budget(actions, budget):
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise ValueError("Budget must be an integer.")

    if budget < 0:
        raise ValueError("Budget cannot be negative.")

    return sum(get_action_cost(a) for a in actions) <= budget

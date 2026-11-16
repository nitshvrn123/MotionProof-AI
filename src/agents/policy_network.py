"""Neural action-value network for MotionProof AI."""

import numpy as np
import torch
from torch import nn

from src.agents.learned_policy import (
    ACTION_DIM,
    ACTION_ORDER,
    STATE_DIM,
    build_action_mask,
    encode_policy_state,
)


class PolicyQNetwork(nn.Module):
    """Estimate the value of each possible investigation action."""

    def __init__(self, hidden_dim=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )

    def forward(self, states):
        if states.shape[-1] != STATE_DIM:
            raise ValueError(
                f"Expected {STATE_DIM} state features, "
                f"received {states.shape[-1]}."
            )

        return self.network(states)


def select_policy_action(
    network,
    environment,
    device="cpu",
):
    """Select the highest-valued valid action.

    The environment provides only policy-visible observations.
    Invalid actions are excluded before selection.
    """

    state = encode_policy_state(environment)
    action_mask = build_action_mask(environment)

    if not action_mask.any():
        raise ValueError(
            "Cannot select an action from a terminated environment."
        )

    device = torch.device(device)

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    mask_tensor = torch.as_tensor(
        action_mask,
        dtype=torch.bool,
        device=device,
    )

    was_training = network.training
    network.eval()

    try:
        with torch.no_grad():
            scores = network(state_tensor).squeeze(0)

            if scores.shape != (ACTION_DIM,):
                raise ValueError("Invalid network output dimensions.")

            if not torch.isfinite(scores).all():
                raise ValueError("Network produced nonfinite action scores.")

            masked_scores = scores.masked_fill(
                ~mask_tensor,
                float("-inf"),
            )

            action_index = int(
                torch.argmax(masked_scores).item()
            )
    finally:
        network.train(was_training)

    return ACTION_ORDER[action_index]


def run_learned_policy(
    network,
    environment,
    device="cpu",
):
    """Execute the learned policy until it makes a terminal decision."""

    while not environment.terminated:
        action = select_policy_action(
            network,
            environment,
            device=device,
        )

        environment.step(action)

    return {
        "final_decision": environment.final_decision,
        "actions": list(environment.action_history),
        "investigations_used": (
            environment.initial_budget
            - environment.remaining_budget
        ),
        "remaining_budget": environment.remaining_budget,
        "investigation_results": dict(environment.results),
    }

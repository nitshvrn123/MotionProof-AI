"""Inference engine for MotionProof AI's finite-horizon fitted Q policy."""

from pathlib import Path

import numpy as np
import torch

from src.agents.investigation_actions import InvestigationAction
from src.agents.learned_policy import (
    ACTION_ORDER,
    STATE_DIM,
    STATE_FEATURE_NAMES,
    build_action_mask,
    encode_policy_state,
)
from src.agents.policy_network import PolicyQNetwork


class FittedQInvestigationPolicy:
    """Stage-specific fitted Q networks with training-only normalization."""

    def __init__(self, checkpoint_path, device="cpu"):

        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)

        # ====================================================
        # 1. VALIDATE CHECKPOINT
        # ====================================================

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        # The existing pilot checkpoint contains NumPy arrays.
        #
        # weights_only=False is required for this checkpoint.
        # Only load checkpoints generated locally and trusted.
        #
        # Future checkpoints should store normalization
        # parameters as PyTorch tensors so weights_only=True
        # can be used safely.

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        # ====================================================
        # 2. LOAD CHECKPOINT METADATA
        # ====================================================

        self.budget = int(checkpoint["budget"])

        self.action_names = tuple(
            checkpoint["action_names"]
        )

        self.feature_names = tuple(
            checkpoint["state_feature_names"]
        )

        expected_actions = tuple(
            action.value for action in ACTION_ORDER
        )

        if self.action_names != expected_actions:
            raise ValueError(
                "Checkpoint action ordering does not match code."
            )

        if self.feature_names != tuple(STATE_FEATURE_NAMES):
            raise ValueError(
                "Checkpoint state features do not match code."
            )

        if int(checkpoint["state_dim"]) != STATE_DIM:
            raise ValueError(
                "Checkpoint state dimension mismatch."
            )

        if int(checkpoint["action_dim"]) != len(ACTION_ORDER):
            raise ValueError(
                "Checkpoint action dimension mismatch."
            )

        if self.budget < 0:
            raise ValueError(
                "Checkpoint investigation budget must be nonnegative."
            )

        # ====================================================
        # 3. LOAD FEATURE NORMALIZATION
        # ====================================================

        self.feature_mean = np.asarray(
            checkpoint["feature_mean"],
            dtype=np.float32,
        )

        self.feature_std = np.asarray(
            checkpoint["feature_std"],
            dtype=np.float32,
        )

        if self.feature_mean.shape != (STATE_DIM,):
            raise ValueError(
                "Invalid normalization mean shape."
            )

        if self.feature_std.shape != (STATE_DIM,):
            raise ValueError(
                "Invalid normalization standard deviation shape."
            )

        if not np.isfinite(self.feature_mean).all():
            raise ValueError(
                "Nonfinite normalization means."
            )

        if not np.isfinite(self.feature_std).all():
            raise ValueError(
                "Nonfinite normalization standard deviations."
            )

        if np.any(self.feature_std <= 0):
            raise ValueError(
                "Normalization standard deviations must be positive."
            )

        # ====================================================
        # 4. LOAD STAGE-SPECIFIC NETWORKS
        # ====================================================

        hidden_dim = int(checkpoint["hidden_dim"])

        self.networks = {}

        for stage in range(self.budget + 1):

            weights = checkpoint[
                "stage_model_state_dicts"
            ][stage]

            network = PolicyQNetwork(
                hidden_dim=hidden_dim
            )

            network.load_state_dict(
                weights,
                strict=True,
            )

            network.to(self.device)
            network.eval()

            self.networks[stage] = network

        if len(self.networks) != self.budget + 1:
            raise RuntimeError(
                "Not all stage-specific networks were loaded."
            )

    # ========================================================
    # 5. SELECT NEXT ACTION
    # ========================================================

    def select_action(self, environment):
        """Select the highest-valued currently valid action."""

        if environment.terminated:
            raise RuntimeError(
                "Cannot act in a terminated environment."
            )

        if environment.initial_budget != self.budget:
            raise ValueError(
                "Environment budget differs from trained policy budget."
            )

        stage = (
            self.budget - environment.remaining_budget
        )

        if stage not in self.networks:
            raise ValueError(
                f"No trained network for stage {stage}."
            )

        # ----------------------------------------------------
        # Encode policy-visible observations.
        # ----------------------------------------------------

        state = encode_policy_state(
            environment
        )

        if state.shape != (STATE_DIM,):
            raise ValueError(
                "Encoded state has incorrect dimensions."
            )

        if not np.isfinite(state).all():
            raise ValueError(
                "Encoded state contains nonfinite values."
            )

        # ----------------------------------------------------
        # Apply training normalization.
        # ----------------------------------------------------

        normalized = (
            state - self.feature_mean
        ) / self.feature_std

        if not np.isfinite(normalized).all():
            raise ValueError(
                "Normalized state contains nonfinite values."
            )

        state_tensor = torch.as_tensor(
            normalized,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        # ----------------------------------------------------
        # Predict action values.
        # ----------------------------------------------------

        network = self.networks[stage]
        network.eval()

        with torch.no_grad():

            q_values = (
                network(state_tensor)
                .squeeze(0)
                .cpu()
                .numpy()
            )

        if q_values.shape != (len(ACTION_ORDER),):
            raise ValueError(
                "Network returned incorrect action dimensions."
            )

        if not np.isfinite(q_values).all():
            raise ValueError(
                "Network produced nonfinite Q values."
            )

        # ----------------------------------------------------
        # Mask unavailable actions.
        # ----------------------------------------------------

        action_mask = build_action_mask(
            environment
        )

        if action_mask.shape != (len(ACTION_ORDER),):
            raise ValueError(
                "Action mask has incorrect dimensions."
            )

        if not action_mask.any():
            raise RuntimeError(
                "No valid actions available."
            )

        masked_values = np.where(
            action_mask,
            q_values,
            -np.inf,
        )

        action_index = int(
            np.argmax(masked_values)
        )

        selected_action = ACTION_ORDER[
            action_index
        ]

        if not action_mask[action_index]:
            raise RuntimeError(
                "Policy selected an invalid action."
            )

        return selected_action

    # ========================================================
    # 6. EXECUTE COMPLETE INVESTIGATION
    # ========================================================

    def run(self, environment):
        """Run the policy until a terminal reliability decision."""

        if environment.terminated:
            raise RuntimeError(
                "Environment has already terminated."
            )

        if environment.initial_budget != self.budget:
            raise ValueError(
                "Environment budget differs from trained policy budget."
            )

        while not environment.terminated:

            action = self.select_action(
                environment
            )

            environment.step(
                action
            )

        final_decision = environment.final_decision

        if final_decision not in (
            InvestigationAction.ACCEPT.value,
            InvestigationAction.FLAG_UNRELIABLE.value,
        ):
            raise RuntimeError(
                "Policy did not produce a valid terminal decision."
            )

        investigations_used = (
            environment.initial_budget
            - environment.remaining_budget
        )

        if investigations_used > self.budget:
            raise RuntimeError(
                "Policy exceeded investigation budget."
            )

        return {
            "final_decision": final_decision,

            "actions": list(
                environment.action_history
            ),

            "investigations_used": investigations_used,

            "remaining_budget": environment.remaining_budget,

            "investigation_results": dict(
                environment.results
            ),
        }


# ============================================================
# 7. CHECKPOINT LOADER
# ============================================================

def load_fitted_q_policy(
    checkpoint_path,
    device="cpu",
):
    """Load a trained fitted Q-learning investigation policy."""

    return FittedQInvestigationPolicy(
        checkpoint_path=checkpoint_path,
        device=device,
    )
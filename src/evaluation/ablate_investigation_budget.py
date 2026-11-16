"""Post-training investigation-budget ablation for frozen fitted Q policy."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.agents.fitted_q_inference import load_fitted_q_policy
from src.agents.investigation_environment import InvestigationEnvironment
from src.agents.investigation_actions import InvestigationAction
from src.agents.learned_policy import (
    ACTION_ORDER,
    STATE_DIM,
    build_action_mask,
    encode_policy_state,
)
from src.evaluation import evaluate_validation_policies as evaluator


ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = ROOT / "results/metrics/final_test"

INPUT_PATH = TEST_DIR / "test_policy_inputs_nonlinear.csv"

ORIGINAL_DECISIONS_PATH = TEST_DIR / "test_policy_decisions.csv"

CHECKPOINT_PATH = (
    ROOT / "checkpoints/learned_policy_fitted_q_full.pt"
)

SUMMARY_PATH = TEST_DIR / "investigation_budget_ablation.csv"

DETAIL_PATH = TEST_DIR / "investigation_budget_ablation_details.csv"

SUBJECTS = (13, 14, 15)
BUDGETS = (0, 1, 2)

EXPECTED_WINDOWS = {
    13: 4565,
    14: 4476,
    15: 3966,
}

TERMINAL_ACTIONS = {
    InvestigationAction.ACCEPT.value,
    InvestigationAction.FLAG_UNRELIABLE.value,
}


def action_value(action):
    """Normalize action enum/string representation."""
    return getattr(action, "value", action)


def select_restricted_action(policy, environment, effective_budget):
    """
    Preserve trained Q-networks and state encoding.

    Restrict investigation actions after effective_budget
    investigations have been performed.
    """

    assert environment.initial_budget == policy.budget == 2

    stage = (
        environment.initial_budget
        - environment.remaining_budget
    )

    assert 0 <= stage <= 2

    # With remaining investigation capacity, use the exact
    # original action-selection implementation.
    if stage < effective_budget:
        return policy.select_action(environment)

    # At the effective budget limit, only terminal actions
    # are permitted. Evaluate them using the trained
    # stage-specific Q-network.
    state = encode_policy_state(environment)

    assert state.shape == (STATE_DIM,)
    assert np.isfinite(state).all()

    normalized = (
        state - policy.feature_mean
    ) / policy.feature_std

    assert np.isfinite(normalized).all()

    state_tensor = torch.as_tensor(
        normalized,
        dtype=torch.float32,
        device=policy.device,
    ).unsqueeze(0)

    network = policy.networks[stage]
    network.eval()

    with torch.no_grad():
        q_values = (
            network(state_tensor)
            .squeeze(0)
            .cpu()
            .numpy()
        )

    assert q_values.shape == (len(ACTION_ORDER),)
    assert np.isfinite(q_values).all()

    original_mask = build_action_mask(environment)

    restricted_mask = np.array(
        [
            bool(original_mask[index])
            and action_value(action) in TERMINAL_ACTIONS
            for index, action in enumerate(ACTION_ORDER)
        ],
        dtype=bool,
    )

    if not restricted_mask.any():
        raise RuntimeError(
            "No terminal actions available at budget limit."
        )

    masked_values = np.where(
        restricted_mask,
        q_values,
        -np.inf,
    )

    return ACTION_ORDER[int(np.argmax(masked_values))]


def run_restricted_policy(policy, environment, effective_budget):
    """Execute the frozen policy under an effective investigation limit."""

    if effective_budget == policy.budget:
        # Preserve exact original inference for budget 2.
        return policy.run(environment)

    if environment.initial_budget != policy.budget:
        raise ValueError("Environment must retain trained budget.")

    while not environment.terminated:

        action = select_restricted_action(
            policy,
            environment,
            effective_budget,
        )

        environment.step(action)

    investigations_used = (
        environment.initial_budget
        - environment.remaining_budget
    )

    if investigations_used > effective_budget:
        raise RuntimeError(
            "Restricted policy exceeded effective budget."
        )

    return {
        "final_decision": environment.final_decision,
        "investigations_used": investigations_used,
        "actions": list(environment.action_history),
    }


def decision_to_binary(decision):

    value = action_value(decision)

    if value == InvestigationAction.FLAG_UNRELIABLE.value:
        return 1

    if value == InvestigationAction.ACCEPT.value:
        return 0

    raise ValueError(f"Unexpected terminal decision: {decision}")


def evaluate_subject(subject, rows, ppg, acc, policy, budget):

    predictions = rows[
        "cnn_predicted_hr"
    ].to_numpy(dtype=np.float64)

    records = []

    for index, row in rows.iterrows():

        history = predictions[
            max(0, index - evaluator.HISTORY_LENGTH):index
        ]

        # Keep the ORIGINAL trained environment budget.
        environment = InvestigationEnvironment(
            ppg_window=ppg[index],
            acc_window=acc[index],
            cnn_predicted_hr=float(
                row["cnn_predicted_hr"]
            ),
            unreliable_probability=float(
                row["unreliable_probability"]
            ),
            previous_hr_predictions=list(history),
            budget=2,
        )

        result = run_restricted_policy(
            policy,
            environment,
            budget,
        )

        prediction = decision_to_binary(
            result["final_decision"]
        )

        investigations = int(
            result["investigations_used"]
        )

        target = int(row["unreliable_target"])

        reward_result = (
            evaluator.calculate_investigation_reward(
                final_decision=action_value(
                    result["final_decision"]
                ),
                unreliable_target=target,
                investigations_used=investigations,
            )
        )

        records.append({
            "subject": subject,
            "window_index": index,
            "budget": budget,
            "policy": "Learned Fitted Q",
            "unreliable_target": target,
            "predicted_unreliable": prediction,
            "unreliable_probability": float(
                row["unreliable_probability"]
            ),
            "investigations_used": investigations,
            "reward": float(
                reward_result["total_reward"]
            ),
            "correct": int(prediction == target),
        })

        if (index + 1) % 1000 == 0:
            print(
                f"S{subject} | budget={budget}: "
                f"{index + 1:,}/{len(rows):,}",
                flush=True,
            )

    return pd.DataFrame(records)


def verify_original_budget_two(details):
    """
    Verify budget=2 ablation reproduces the previously
    frozen final-test evaluation exactly.
    """

    original = pd.read_csv(ORIGINAL_DECISIONS_PATH)

    original = original.loc[
        original["policy"] == "Learned Fitted Q"
    ].copy()

    reproduced = details.loc[
        details["budget"] == 2
    ].copy()

    keys = ["subject", "window_index"]

    original = original.sort_values(keys).reset_index(drop=True)
    reproduced = reproduced.sort_values(keys).reset_index(drop=True)

    assert len(original) == len(reproduced) == 13007

    assert original[keys].equals(reproduced[keys])

    for column in (
        "unreliable_target",
        "predicted_unreliable",
        "investigations_used",
    ):
        assert np.array_equal(
            original[column].to_numpy(),
            reproduced[column].to_numpy(),
        ), f"Budget=2 mismatch: {column}"

    assert np.allclose(
        original["reward"].to_numpy(),
        reproduced["reward"].to_numpy(),
        rtol=0,
        atol=1e-10,
    ), "Budget=2 reward mismatch"

    print(
        "\nPASS: Budget=2 exactly reproduces "
        "the frozen final-test decisions."
    )


def main():

    print("=" * 75)
    print("MOTIONPROOF AI — STEP 37.2C")
    print("RESTRICTED INVESTIGATION BUDGET ABLATION")
    print("=" * 75)

    for path in (SUMMARY_PATH, DETAIL_PATH):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing results: {path}"
            )

    inputs = pd.read_csv(INPUT_PATH)

    assert len(inputs) == 13007
    assert set(inputs["subject"]) == set(SUBJECTS)

    policy = load_fitted_q_policy(
        CHECKPOINT_PATH,
        device="cpu",
    )

    assert policy.budget == 2

    print("\nPASS: Frozen budget=2 policy loaded.")

    all_results = []

    for subject in SUBJECTS:

        path = ROOT / f"data/processed/test/S{subject}.npz"

        with np.load(path) as data:
            ppg = data["ppg"]
            acc = data["acc"]

        rows = (
            inputs.loc[inputs["subject"] == subject]
            .sort_values("window_index")
            .reset_index(drop=True)
        )

        assert len(rows) == EXPECTED_WINDOWS[subject]
        assert len(ppg) == len(acc) == len(rows)

        assert np.array_equal(
            rows["window_index"].to_numpy(),
            np.arange(len(rows)),
        )

        for budget in BUDGETS:

            print(
                f"\nEvaluating S{subject}, "
                f"effective budget={budget}...",
                flush=True,
            )

            result = evaluate_subject(
                subject,
                rows,
                ppg,
                acc,
                policy,
                budget,
            )

            all_results.append(result)

    details = pd.concat(
        all_results,
        ignore_index=True,
    )

    assert len(details) == 13007 * len(BUDGETS)

    assert not details.duplicated(
        ["subject", "window_index", "budget"]
    ).any()

    verify_original_budget_two(details)

    records = []

    for budget, group in details.groupby("budget"):

        metrics = evaluator.calculate_metrics(group)
        metrics["budget"] = int(budget)

        records.append(metrics)

    summary = pd.DataFrame(records)

    display_columns = [
        "budget",
        "windows",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "average_investigations",
        "average_reward",
    ]

    print("\n" + "=" * 75)
    print("INVESTIGATION BUDGET ABLATION RESULTS")
    print("=" * 75)

    print(
        summary[display_columns]
        .round(4)
        .to_string(index=False)
    )

    details.to_csv(
        DETAIL_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\nSaved:", DETAIL_PATH)
    print("Saved:", SUMMARY_PATH)

    print("\nPASS: Step 37.2C completed.")


if __name__ == "__main__":
    main()

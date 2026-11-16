"""Leave-one-investigation-action-out ablation of frozen fitted Q policy."""

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
ORIGINAL_PATH = TEST_DIR / "test_policy_decisions.csv"

CHECKPOINT_PATH = (
    ROOT / "checkpoints/learned_policy_fitted_q_full.pt"
)

SUMMARY_PATH = TEST_DIR / "investigation_action_ablation.csv"
DETAIL_PATH = TEST_DIR / "investigation_action_ablation_details.csv"

SUBJECTS = (13, 14, 15)

EXPECTED_WINDOWS = {
    13: 4565,
    14: 4476,
    15: 3966,
}

CONFIGURATIONS = {
    "Full Agent": None,
    "Without PPG": InvestigationAction.CHECK_PPG,
    "Without Motion": InvestigationAction.CHECK_MOTION,
    "Without Consistency": InvestigationAction.CHECK_CONSISTENCY,
}


def action_value(action):
    return getattr(action, "value", action)


def select_masked_action(policy, environment, excluded_action):
    """Select highest-Q valid action excluding one investigation tool."""

    if environment.terminated:
        raise RuntimeError("Cannot act in terminated environment.")

    if environment.initial_budget != policy.budget:
        raise ValueError("Environment budget differs from checkpoint.")

    stage = (
        policy.budget - environment.remaining_budget
    )

    if stage not in policy.networks:
        raise RuntimeError(f"Missing trained network for stage {stage}.")

    state = encode_policy_state(environment)

    if state.shape != (STATE_DIM,):
        raise RuntimeError("Unexpected policy state dimensions.")

    if not np.isfinite(state).all():
        raise RuntimeError("Nonfinite policy state.")

    normalized = (
        state - policy.feature_mean
    ) / policy.feature_std

    if not np.isfinite(normalized).all():
        raise RuntimeError("Nonfinite normalized state.")

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

    if q_values.shape != (len(ACTION_ORDER),):
        raise RuntimeError("Unexpected Q-value dimensions.")

    if not np.isfinite(q_values).all():
        raise RuntimeError("Nonfinite Q-values.")

    action_mask = build_action_mask(environment)

    if action_mask.shape != (len(ACTION_ORDER),):
        raise RuntimeError("Unexpected action-mask dimensions.")

    excluded_value = action_value(excluded_action)

    restricted_mask = np.asarray(
        [
            bool(action_mask[index])
            and action_value(action) != excluded_value
            for index, action in enumerate(ACTION_ORDER)
        ],
        dtype=bool,
    )

    if not restricted_mask.any():
        raise RuntimeError("No valid actions after exclusion.")

    masked_q_values = np.where(
        restricted_mask,
        q_values,
        -np.inf,
    )

    selected_index = int(np.argmax(masked_q_values))

    return ACTION_ORDER[selected_index]


def run_ablation_policy(policy, environment, excluded_action):
    """Run original policy or action-restricted policy."""

    if excluded_action is None:
        return policy.run(environment)

    while not environment.terminated:

        action = select_masked_action(
            policy,
            environment,
            excluded_action,
        )

        if action_value(action) == action_value(excluded_action):
            raise RuntimeError("Excluded action was selected.")

        environment.step(action)

    investigations_used = (
        environment.initial_budget
        - environment.remaining_budget
    )

    if investigations_used > policy.budget:
        raise RuntimeError("Investigation budget exceeded.")

    return {
        "final_decision": environment.final_decision,
        "investigations_used": investigations_used,
        "actions": list(environment.action_history),
    }


def decision_to_binary(decision):

    value = action_value(decision)

    if value == InvestigationAction.ACCEPT.value:
        return 0

    if value == InvestigationAction.FLAG_UNRELIABLE.value:
        return 1

    raise ValueError(f"Invalid terminal decision: {decision}")


def evaluate_subject(
    subject,
    rows,
    ppg,
    acc,
    policy,
    configuration,
    excluded_action,
):

    predictions = rows[
        "cnn_predicted_hr"
    ].to_numpy(dtype=np.float64)

    records = []

    for index, row in rows.iterrows():

        history = predictions[
            max(0, index - evaluator.HISTORY_LENGTH):index
        ]

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

        result = run_ablation_policy(
            policy,
            environment,
            excluded_action,
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
            "configuration": configuration,
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
                f"S{subject} | {configuration}: "
                f"{index + 1:,}/{len(rows):,}",
                flush=True,
            )

    return pd.DataFrame(records)


def verify_full_agent(details):
    """Ensure full-agent condition reproduces frozen test decisions."""

    original = pd.read_csv(ORIGINAL_PATH)

    original = original.loc[
        original["policy"] == "Learned Fitted Q"
    ].copy()

    reproduced = details.loc[
        details["configuration"] == "Full Agent"
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
        ), f"Full-agent mismatch: {column}"

    assert np.allclose(
        original["reward"].to_numpy(),
        reproduced["reward"].to_numpy(),
        rtol=0,
        atol=1e-10,
    ), "Full-agent reward mismatch"

    print(
        "\nPASS: Full-agent ablation reproduces "
        "all 13,007 frozen final-test decisions."
    )


def main():

    print("=" * 75)
    print("MOTIONPROOF AI — STEP 37.3B")
    print("LEAVE-ONE-INVESTIGATION-ACTION-OUT ABLATION")
    print("=" * 75)

    for path in (SUMMARY_PATH, DETAIL_PATH):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}"
            )

    inputs = pd.read_csv(INPUT_PATH)

    assert len(inputs) == 13007
    assert set(inputs["subject"]) == set(SUBJECTS)

    policy = load_fitted_q_policy(
        CHECKPOINT_PATH,
        device="cpu",
    )

    assert policy.budget == 2

    print("\nPASS: Frozen fitted Q policy loaded.")

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

        for configuration, excluded_action in CONFIGURATIONS.items():

            print(
                f"\nEvaluating S{subject}: {configuration}",
                flush=True,
            )

            result = evaluate_subject(
                subject,
                rows,
                ppg,
                acc,
                policy,
                configuration,
                excluded_action,
            )

            all_results.append(result)

    details = pd.concat(
        all_results,
        ignore_index=True,
    )

    assert len(details) == 13007 * len(CONFIGURATIONS)

    assert not details.duplicated(
        ["subject", "window_index", "configuration"]
    ).any()

    verify_full_agent(details)

    records = []

    for configuration in CONFIGURATIONS:

        group = details.loc[
            details["configuration"] == configuration
        ]

        metrics = evaluator.calculate_metrics(group)

        metrics["configuration"] = configuration

        records.append(metrics)

    summary = pd.DataFrame(records)

    display_columns = [
        "configuration",
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
    print("INVESTIGATION ACTION ABLATION RESULTS")
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

    print("\nPASS: Step 37.3B completed.")


if __name__ == "__main__":
    main()

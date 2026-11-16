"""Offline investigation-state generation for MotionProof AI."""

from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.investigation_actions import InvestigationAction
from src.agents.investigation_environment import InvestigationEnvironment
from src.agents.investigation_reward import calculate_investigation_reward
from src.agents.learned_policy import (
    ACTION_ORDER,
    STATE_FEATURE_NAMES,
    build_action_mask,
    encode_policy_state,
)


INVESTIGATIONS = (
    InvestigationAction.CHECK_PPG,
    InvestigationAction.CHECK_MOTION,
    InvestigationAction.CHECK_CONSISTENCY,
)


def generate_action_sequences(budget=2):
    """Generate all valid investigation prefixes."""
    sequences = [()]

    for length in range(1, min(budget, len(INVESTIGATIONS)) + 1):
        sequences.extend(permutations(INVESTIGATIONS, length))

    return sequences


def make_environment(row, ppg, acc, history, budget=2):
    return InvestigationEnvironment(
        ppg_window=ppg,
        acc_window=acc,
        cnn_predicted_hr=float(row["cnn_predicted_hr"]),
        unreliable_probability=float(row["unreliable_probability"]),
        previous_hr_predictions=np.asarray(history, dtype=float),
        budget=budget,
    )


def generate_training_states(
    project_root,
    max_windows_per_subject=None,
    budget=2,
):
    """Generate offline states and terminal rewards for training subjects."""

    project_root = Path(project_root)

    df = pd.read_csv(
        project_root / "results/metrics/policy_training_dataset.csv"
    )

    assert set(df["subject"].unique()) == set(range(1, 11))

    sequences = generate_action_sequences(budget)

    records = []

    for subject in range(1, 11):
        subject_rows = (
            df.loc[df["subject"] == subject]
            .sort_values("window_index")
        )

        if max_windows_per_subject is not None:
            subject_rows = subject_rows.head(max_windows_per_subject)

        with np.load(
            project_root / f"data/processed/train/S{subject}.npz"
        ) as participant:
            ppg_windows = participant["ppg"]
            acc_windows = participant["acc"]

            for _, row in subject_rows.iterrows():
                index = int(row["window_index"])

                history = (
                    df.loc[
                        (df["subject"] == subject)
                        & (df["window_index"] < index),
                        "cnn_predicted_hr",
                    ]
                    .tail(5)
                    .to_numpy(dtype=float)
                )

                for sequence in sequences:
                    env = make_environment(
                        row,
                        ppg_windows[index],
                        acc_windows[index],
                        history,
                        budget=budget,
                    )

                    for action in sequence:
                        env.step(action)

                    state = encode_policy_state(env)
                    mask = build_action_mask(env)

                    used = budget - env.remaining_budget
                    target = int(row["unreliable_target"])

                    accept_reward = calculate_investigation_reward(
                        InvestigationAction.ACCEPT,
                        target,
                        used,
                    )

                    flag_reward = calculate_investigation_reward(
                        InvestigationAction.FLAG_UNRELIABLE,
                        target,
                        used,
                    )

                    record = {
                        "subject": subject,
                        "window_index": index,
                        "sequence": "|".join(a.value for a in sequence),
                        "investigations_used": used,
                        "unreliable_target": target,
                        "accept_reward": accept_reward,
                        "flag_reward": flag_reward,
                    }

                    record.update(
                        dict(zip(STATE_FEATURE_NAMES, state.tolist()))
                    )

                    for action, allowed in zip(ACTION_ORDER, mask):
                        record[f"valid_{action.value}"] = int(allowed)

                    records.append(record)

        print(
            f"S{subject}: generated states for "
            f"{len(subject_rows)} windows"
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]

    output = root / "results/metrics/policy_investigation_states.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset = generate_training_states(
        project_root=root,
        max_windows_per_subject=100,
        budget=2,
    )

    dataset.to_csv(output, index=False)

    print("\nDataset shape:", dataset.shape)
    print("Saved:", output)
    print("PASS: Offline investigation states generated.")

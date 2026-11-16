"""Evaluate three reliability decision policies on unseen S11-S12 windows."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from src.agents.investigation_environment import (
    InvestigationEnvironment,
)

from src.agents.fixed_rule_policy import (
    run_fixed_rule_policy,
)

from src.agents.fitted_q_inference import (
    load_fitted_q_policy,
)

from src.agents.investigation_reward import (
    calculate_investigation_reward,
)


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "results/metrics/validation_policy_inputs.csv"
)

CHECKPOINT_PATH = (
    ROOT / "checkpoints/learned_policy_fitted_q_full.pt"
)

OUTPUT_DIRECTORY = ROOT / "results/metrics"

BUDGET = 2
HISTORY_LENGTH = 5


def make_environment(ppg, acc, row, history):
    return InvestigationEnvironment(
        ppg_window=ppg,
        acc_window=acc,
        cnn_predicted_hr=float(row["cnn_predicted_hr"]),
        unreliable_probability=float(row["unreliable_probability"]),
        previous_hr_predictions=list(history),
        budget=BUDGET,
    )


def decision_to_binary(decision):

    if hasattr(decision, "value"):
        decision = decision.value

    decision = str(decision)

    if decision == "ACCEPT":
        return 0

    if decision == "FLAG_UNRELIABLE":
        return 1

    raise ValueError(f"Unexpected decision: {decision}")


def run_fixed(environment):

    result = run_fixed_rule_policy(environment)

    if isinstance(result, dict):
        decision = result["final_decision"]
        investigations = result["investigations_used"]
    else:
        decision = environment.final_decision
        investigations = (
            environment.initial_budget
            - environment.remaining_budget
        )

    return (
        decision_to_binary(decision),
        int(investigations),
    )


def run_learned(environment, policy):

    result = policy.run(environment)

    return (
        decision_to_binary(result["final_decision"]),
        int(result["investigations_used"]),
    )


def evaluate_subject(subject, validation, learned_policy):

    path = ROOT / f"data/processed/val/S{subject}.npz"

    with np.load(path) as data:
        ppg = data["ppg"]
        acc = data["acc"]

    rows = (
        validation.loc[validation["subject"] == subject]
        .sort_values("window_index")
        .reset_index(drop=True)
    )

    assert len(rows) == len(ppg)

    assert np.array_equal(
        rows["window_index"].to_numpy(),
        np.arange(len(rows)),
    )

    predictions = rows[
        "cnn_predicted_hr"
    ].to_numpy(dtype=np.float64)

    records = []

    for index, row in rows.iterrows():

        history = predictions[
            max(0, index - HISTORY_LENGTH):index
        ]

        baseline_prediction = int(
            row["unreliable_probability"] >= 0.5
        )

        fixed_environment = make_environment(
            ppg[index],
            acc[index],
            row,
            history,
        )

        fixed_prediction, fixed_investigations = (
            run_fixed(fixed_environment)
        )

        learned_environment = make_environment(
            ppg[index],
            acc[index],
            row,
            history,
        )

        learned_prediction, learned_investigations = (
            run_learned(
                learned_environment,
                learned_policy,
            )
        )

        target = int(row["unreliable_target"])

        policy_outputs = {
            "No Investigation": (
                baseline_prediction,
                0,
            ),
            "Fixed Rule": (
                fixed_prediction,
                fixed_investigations,
            ),
            "Learned Fitted Q": (
                learned_prediction,
                learned_investigations,
            ),
        }

        for policy_name, (
            prediction,
            investigations,
        ) in policy_outputs.items():

            assert prediction in (0, 1)
            assert 0 <= investigations <= BUDGET

            decision = (
                "FLAG_UNRELIABLE"
                if prediction == 1
                else "ACCEPT"
            )

            reward_result = (
                calculate_investigation_reward(
                    final_decision=decision,
                    unreliable_target=target,
                    investigations_used=investigations,
                )
            )

            reward = float(
                reward_result["total_reward"]
            )

            records.append({
                "subject": subject,
                "window_index": index,
                "policy": policy_name,
                "unreliable_target": target,
                "predicted_unreliable": prediction,
                "unreliable_probability": float(
                    row["unreliable_probability"]
                ),
                "investigations_used": investigations,
                "reward": reward,
                "correct": int(prediction == target),
            })

        if (index + 1) % 1000 == 0:
            print(
                f"S{subject}: {index + 1:,}/{len(rows):,} "
                "windows evaluated.",
                flush=True,
            )

    return pd.DataFrame(records)


def calculate_metrics(group):

    y_true = group[
        "unreliable_target"
    ].to_numpy(dtype=int)

    y_pred = group[
        "predicted_unreliable"
    ].to_numpy(dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else np.nan
    )

    return {
        "windows": len(group),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "average_investigations": group[
            "investigations_used"
        ].mean(),
        "average_reward": group["reward"].mean(),
        "flag_rate": group[
            "predicted_unreliable"
        ].mean(),
    }


def main():

    print("=" * 72)
    print("MOTIONPROOF AI — FULL VALIDATION POLICY EVALUATION")
    print("=" * 72)

    validation = pd.read_csv(INPUT_PATH)

    assert len(validation) == 8475
    assert set(validation["subject"]) == {11, 12}

    assert not validation.duplicated(
        ["subject", "window_index"]
    ).any()

    assert validation[
        "unreliable_probability"
    ].between(0, 1).all()

    print("PASS: Validation dataset verified.")

    learned_policy = load_fitted_q_policy(
        CHECKPOINT_PATH,
        device="cpu",
    )

    print("PASS: Full learned policy loaded.")

    results = []

    for subject in (11, 12):

        print(
            f"\nEvaluating S{subject}...",
            flush=True,
        )

        subject_results = evaluate_subject(
            subject,
            validation,
            learned_policy,
        )

        results.append(subject_results)

        print(
            f"PASS: S{subject} completed.",
            flush=True,
        )

    decisions = pd.concat(
        results,
        ignore_index=True,
    )

    assert len(decisions) == 8475 * 3

    expected_policies = {
        "No Investigation",
        "Fixed Rule",
        "Learned Fitted Q",
    }

    assert set(decisions["policy"]) == expected_policies

    assert not decisions.duplicated(
        ["subject", "window_index", "policy"]
    ).any()

    # --------------------------------------------------------
    # Overall policy metrics
    # --------------------------------------------------------

    overall_records = []

    for policy_name, group in decisions.groupby("policy"):

        metrics = calculate_metrics(group)

        metrics["policy"] = policy_name

        overall_records.append(metrics)

    overall = pd.DataFrame(overall_records)

    # --------------------------------------------------------
    # Per-subject policy metrics
    # --------------------------------------------------------

    subject_records = []

    for (subject, policy_name), group in decisions.groupby(
        ["subject", "policy"]
    ):

        metrics = calculate_metrics(group)

        metrics["subject"] = subject
        metrics["policy"] = policy_name

        subject_records.append(metrics)

    per_subject = pd.DataFrame(subject_records)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    decisions_path = (
        OUTPUT_DIRECTORY
        / "validation_policy_decisions.csv"
    )

    overall_path = (
        OUTPUT_DIRECTORY
        / "validation_policy_comparison.csv"
    )

    subject_path = (
        OUTPUT_DIRECTORY
        / "validation_policy_per_subject.csv"
    )

    decisions.to_csv(
        decisions_path,
        index=False,
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    per_subject.to_csv(
        subject_path,
        index=False,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    display_columns = [
        "policy",
        "windows",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "average_investigations",
        "average_reward",
    ]

    print("\n" + "=" * 72)
    print("OVERALL VALIDATION COMPARISON")
    print("=" * 72)

    print(
        overall[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("PER-SUBJECT VALIDATION COMPARISON")
    print("=" * 72)

    print(
        per_subject[
            ["subject"] + display_columns
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("SAVED FILES")
    print("=" * 72)

    print(decisions_path)
    print(overall_path)
    print(subject_path)

    print(
        "\nPASS: Full unseen-subject policy evaluation completed."
    )


if __name__ == "__main__":
    main()

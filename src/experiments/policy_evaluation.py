"""Replay a fitted Q policy on validation or test windows.

Environment construction, history length and inputs match
src/evaluation/evaluate_final_test_policies.py, so a replay of the original
checkpoint reproduces the frozen decisions (checked in tests).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.fitted_q_inference import load_fitted_q_policy
from src.agents.investigation_reward import InvestigationRewardConfig, calculate_investigation_reward
from src.evaluation import evaluate_validation_policies as evaluator
from src.experiments.common import ROOT, TEST_INPUTS, VALIDATION_INPUTS

SPLITS = {
    "validation": {"subjects": (11, 12), "inputs": VALIDATION_INPUTS, "folder": "val"},
    "test": {"subjects": (13, 14, 15), "inputs": TEST_INPUTS, "folder": "test"},
}


def load_split(split):
    spec = SPLITS[split]
    inputs = pd.read_csv(spec["inputs"])
    signals = {}
    for subject in spec["subjects"]:
        with np.load(ROOT / "data" / "processed" / spec["folder"] / f"S{subject}.npz") as data:
            signals[subject] = {"ppg": data["ppg"].copy(), "acc": data["acc"].copy()}
    return inputs, signals


def replay_policy(policy, inputs, signals, reward_config=None):
    """Run a policy on every window. Rewards use the given (default original) config."""

    reward_config = reward_config or InvestigationRewardConfig()
    if isinstance(policy, (str, Path)):
        policy = load_fitted_q_policy(policy, device="cpu")

    records = []
    for subject, sig in signals.items():
        rows = inputs[inputs["subject"] == subject].sort_values("window_index").reset_index(drop=True)
        preds = rows["cnn_predicted_hr"].to_numpy(dtype=np.float64)
        for i, row in rows.iterrows():
            w = int(row["window_index"])
            history = preds[max(0, i - evaluator.HISTORY_LENGTH):i]
            result = policy.run(evaluator.make_environment(sig["ppg"][w], sig["acc"][w], row, history))
            target = int(row["unreliable_target"])
            used = int(result["investigations_used"])
            records.append({
                "subject": int(subject),
                "window_index": w,
                "unreliable_target": target,
                "unreliable_probability": float(row["unreliable_probability"]),
                "predicted_unreliable": int(result["final_decision"] == "FLAG_UNRELIABLE"),
                "investigations_used": used,
                "actions": "|".join(result["actions"]),
                "reward": float(calculate_investigation_reward(
                    result["final_decision"], target, used, reward_config)["total_reward"]),
            })
    return pd.DataFrame(records)


def decision_rewards(flagged, target, investigations, reward_config=None):
    """Vectorized original-reward computation for decision-layer variants."""

    c = reward_config or InvestigationRewardConfig()
    flagged = np.asarray(flagged, dtype=bool)
    target = np.asarray(target, dtype=bool)
    base = np.select(
        [flagged & target, flagged & ~target, ~flagged & target],
        [c.true_positive_reward, c.false_positive_penalty, c.false_negative_penalty],
        default=c.true_negative_reward,
    )
    return base - c.investigation_cost * np.asarray(investigations, dtype=float)


def bayes_threshold(reward_config):
    """Probability above which flagging has higher expected reward than accepting."""

    c = reward_config
    gain_fp = c.true_negative_reward - c.false_positive_penalty
    gain_tp = c.true_positive_reward - c.false_negative_penalty
    return gain_fp / (gain_fp + gain_tp)

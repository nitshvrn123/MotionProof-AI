"""Regression tests for Phase 2-3 experiment code."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agents.investigation_reward import InvestigationRewardConfig, calculate_investigation_reward
from src.experiments.common import classification_metrics
from src.experiments.policy_evaluation import bayes_threshold, decision_rewards
from src.experiments.robustness import PERTURBATIONS, apply, motion_intensity

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "metrics" / "final_test"


def windows(n=6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(512) / 64
    ppg = (np.sin(2 * np.pi * 1.2 * t)[None, :] * 50 + rng.normal(0, 5, (n, 512))).astype(np.float32)
    acc = rng.normal(0, 0.1, (n, 256, 3)).astype(np.float32)
    acc[..., 2] += 1.0
    return ppg, acc


@pytest.mark.parametrize("name", sorted(PERTURBATIONS))
def test_perturbations_are_deterministic_and_never_modify_inputs(name):
    ppg, acc = windows()
    ppg0, acc0 = ppg.copy(), acc.copy()
    level = {"amplitude_attenuation": 0.5, "signal_dropout": 1.0, "clipping": 0.1}.get(name, 0.2)
    a = apply(name, ppg, acc, level, seed=7)
    b = apply(name, ppg, acc, level, seed=7)
    assert np.array_equal(ppg, ppg0) and np.array_equal(acc, acc0)
    assert np.array_equal(a[0], b[0]) and a[0].shape == ppg.shape and a[1].shape == acc.shape
    assert not np.array_equal(a[0], ppg)


def test_clean_condition_is_an_exact_copy():
    ppg, acc = windows()
    new_ppg, new_acc = apply("clean", ppg, acc, 0, seed=1)
    assert np.array_equal(new_ppg, ppg) and new_ppg is not ppg


def test_gaussian_noise_scale_matches_level():
    ppg, acc = windows(n=200)
    noisy, _ = apply("gaussian_noise", ppg, acc, 0.2, seed=3)
    ratio = (noisy - ppg).std(axis=1) / ppg.std(axis=1)
    assert ratio.mean() == pytest.approx(0.2, abs=0.01)


def test_dropout_flattens_requested_duration():
    ppg, acc = windows()
    out, _ = apply("signal_dropout", ppg, acc, 2.0, seed=4)
    flat = (np.diff(out, axis=1) == 0).sum(axis=1)
    assert (flat >= 2 * 64 - 1).all()


def test_motion_intensity_is_zero_for_static_gravity():
    acc = np.zeros((3, 256, 3))
    acc[..., 2] = 1.0
    assert np.allclose(motion_intensity(acc), 0.0)


def test_bayes_threshold_matches_expected_reward_break_even():
    cfg = InvestigationRewardConfig()
    tau = bayes_threshold(cfg)
    assert tau == pytest.approx(1.5 / 5.5)
    flag = tau * cfg.true_positive_reward + (1 - tau) * cfg.false_positive_penalty
    accept = tau * cfg.false_negative_penalty + (1 - tau) * cfg.true_negative_reward
    assert flag == pytest.approx(accept)


def test_vectorized_rewards_match_project_reward_function():
    rng = np.random.default_rng(5)
    flagged = rng.integers(0, 2, 300)
    target = rng.integers(0, 2, 300)
    used = rng.integers(0, 3, 300)
    expected = [calculate_investigation_reward("FLAG_UNRELIABLE" if f else "ACCEPT", int(t), int(u))
                ["total_reward"] for f, t, u in zip(flagged, target, used)]
    assert np.allclose(decision_rewards(flagged, target, used), expected)


def test_classification_metrics_reproduce_frozen_comparison():
    decisions = pd.read_csv(FINAL / "test_policy_decisions.csv")
    frozen = pd.read_csv(FINAL / "test_policy_comparison.csv").set_index("policy")
    for policy, group in decisions.groupby("policy"):
        m = classification_metrics(group["unreliable_target"], group["predicted_unreliable"],
                                   group["investigations_used"], group["reward"])
        for key in ("accuracy", "precision", "recall", "f1", "false_positive_rate",
                    "average_investigations", "average_reward"):
            assert m[key] == pytest.approx(frozen.loc[policy, key], abs=1e-12), (policy, key)


def test_fitted_q_reward_reconstruction_matches_project_reward():
    from src.agents.fitted_q_training import terminal_rewards
    cfg = InvestigationRewardConfig(false_positive_penalty=-2.5, investigation_cost=0.2)
    accept, flag = terminal_rewards(np.array([0, 1, 0, 1]), np.array([0, 1, 2, 2]), cfg)
    for i, (t, u) in enumerate([(0, 0), (1, 1), (0, 2), (1, 2)]):
        assert accept[i] == pytest.approx(calculate_investigation_reward("ACCEPT", t, u, cfg)["total_reward"])
        assert flag[i] == pytest.approx(
            calculate_investigation_reward("FLAG_UNRELIABLE", t, u, cfg)["total_reward"])


def test_training_module_refuses_to_overwrite_original_checkpoint():
    from src.agents.fitted_q_training import ORIGINAL_CHECKPOINT, save_checkpoint
    with pytest.raises(RuntimeError):
        save_checkpoint({}, ORIGINAL_CHECKPOINT)


@pytest.mark.skipif(not (ROOT / "data/processed/test/S15.npz").exists(), reason="test data missing")
def test_replay_of_original_policy_reproduces_frozen_decisions():
    from src.experiments.policy_evaluation import load_split, replay_policy
    inputs, signals = load_split("test")
    inputs = inputs[(inputs["subject"] == 15) & (inputs["window_index"] < 300)]
    replay = replay_policy(ROOT / "checkpoints/learned_policy_fitted_q_full.pt", inputs, {15: signals[15]})
    frozen = pd.read_csv(FINAL / "test_policy_decisions.csv")
    frozen = frozen[(frozen["policy"] == "Learned Fitted Q") & (frozen["subject"] == 15)
                    & (frozen["window_index"] < 300)]
    merged = replay.merge(frozen, on=["subject", "window_index"], suffixes=("", "_frozen"))
    assert len(merged) == 300
    assert (merged["predicted_unreliable"] == merged["predicted_unreliable_frozen"]).all()
    assert (merged["investigations_used"] == merged["investigations_used_frozen"]).all()
    assert np.allclose(merged["reward"], merged["reward_frozen"])


def test_frozen_d2_artifacts_are_unchanged():
    from src.experiments.verify_frozen import check
    count, problems = check()
    assert count >= 15 and problems == []


PHASE2 = ROOT / "results" / "experiments" / "phase2_fp_reduction"
PHASE3 = ROOT / "results" / "experiments" / "phase3_robustness"


@pytest.mark.skipif(not (PHASE2 / "selection.json").exists(), reason="phase 2 not run")
def test_phase2_selection_is_reproducible_from_validation_only():
    import json
    from src.experiments.run_fp_reduction import select, seed_means
    from src.experiments.common import load_config
    validation = pd.read_csv(PHASE2 / "validation_candidates.csv")
    assert set(validation["split"]) == {"validation"}
    floor, per_family, overall = select(validation, load_config("phase2_fp_reduction"))
    saved = json.loads((PHASE2 / "selection.json").read_text())
    assert floor == pytest.approx(saved["recall_floor_validation"])
    assert overall[["family", "variant"]].to_dict(orient="records") == saved["selected_overall"]
    means = seed_means(validation)
    assert (means.loc[means["family"] == "reward_retrained_policy", "seed"] == "mean").all()


@pytest.mark.skipif(not (PHASE2 / "test_comparison.csv").exists(), reason="phase 2 not run")
def test_phase2_frozen_baselines_are_copied_unchanged():
    comparison = pd.read_csv(PHASE2 / "test_comparison.csv")
    frozen = pd.read_csv(FINAL / "test_policy_comparison.csv").set_index("policy")
    rows = comparison[comparison["family"] == "frozen_d2_baseline"].set_index("variant")
    for policy in frozen.index:
        for key in ("recall", "false_positive_rate", "f1", "average_reward"):
            assert rows.loc[policy, key] == pytest.approx(frozen.loc[policy, key], abs=1e-12)


@pytest.mark.skipif(not (PHASE3 / "synthetic_summary.csv").exists(), reason="phase 3 not run")
def test_phase3_clean_condition_matches_frozen_results():
    summary = pd.read_csv(PHASE3 / "synthetic_summary.csv")
    clean = summary[summary["perturbation"] == "clean"].set_index("policy")
    frozen = pd.read_csv(FINAL / "test_policy_comparison.csv").set_index("policy")
    assert clean.loc["Learned Fitted Q", "recall"] == pytest.approx(frozen.loc["Learned Fitted Q", "recall"], abs=0.002)
    assert clean.loc["Learned Fitted Q", "false_positive_rate"] == pytest.approx(
        frozen.loc["Learned Fitted Q", "false_positive_rate"], abs=0.002)
    assert clean.loc["Learned Fitted Q", "cnn_mae_bpm"] == pytest.approx(5.4005, abs=0.01)
    assert (summary["windows"] == 13007).all()


def test_fitted_q_training_produces_a_checkpoint_the_production_loader_accepts(tmp_path, monkeypatch):
    import torch
    from itertools import permutations
    import src.agents.fitted_q_training as fqt
    from src.agents.fitted_q_inference import load_fitted_q_policy
    from src.agents.learned_policy import STATE_DIM

    monkeypatch.setattr(fqt, "EPOCHS_PER_STAGE", 2)
    checks = ["CHECK_PPG", "CHECK_MOTION", "CHECK_CONSISTENCY"]
    seqs = [()] + [(a,) for a in checks] + list(permutations(checks, 2))
    rng = np.random.default_rng(0)
    rows, index = [], {}
    for w in range(20):
        for seq in seqs:
            index[(w, seq)] = len(rows)
            rows.append((w, seq))
    n = len(rows)
    masks = np.ones((n, 5), dtype=bool)
    depth = np.array([len(s) for _, s in rows], dtype=np.int8)
    transition = np.full((n, 5), -1, dtype=np.int32)
    for i, (w, seq) in enumerate(rows):
        for a, name in enumerate(checks, start=2):
            if name in seq or len(seq) == 2:
                masks[i, a] = False
            else:
                transition[i, a] = index[(w, seq + (name,))]
    data = {"states": rng.normal(size=(n, STATE_DIM)).astype(np.float32), "masks": masks, "depth": depth,
            "transition": transition, "target": np.array([w % 2 for w, _ in rows], dtype=np.int8)}
    checkpoint, history = fqt.train_fitted_q(data=data, log=None)
    path = tmp_path / "tiny.pt"
    torch.save(checkpoint, path)
    policy = load_fitted_q_policy(path)
    assert sorted(policy.networks) == [0, 1, 2]
    assert set(history["stage"]) == {0, 1, 2}
    assert checkpoint["reward_config"]["false_positive_penalty"] == -1.5

"""Phase 2: false-positive reduction experiments.

Every candidate is scored on validation (S11, S12). The selection rule from
configs/experiments/phase2_fp_reduction.json picks one variant per family
and one overall. Only selected variants are then evaluated on test, next to
the frozen D2 baselines read from results/metrics/final_test/.

Usage:
    python -m src.experiments.train_reward_variants   # ~5 min per variant on CPU
    python -m src.experiments.run_fp_reduction
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agents.investigation_reward import InvestigationRewardConfig
from src.experiments.common import (
    EXPERIMENTAL_CHECKPOINTS,
    FINAL_TEST,
    ROOT,
    classification_metrics,
    load_config,
    output_dir,
    set_seed,
    write_manifest,
)
from src.experiments.policy_evaluation import (
    bayes_threshold,
    decision_rewards,
    load_split,
    replay_policy,
)
from src.experiments.train_reward_variants import variant_path

VAL_DECISIONS = ROOT / "results/metrics/nonlinear_validation/validation_policy_decisions.csv"
TEST_DECISIONS = FINAL_TEST / "test_policy_decisions.csv"
TEST_COMPARISON = FINAL_TEST / "test_policy_comparison.csv"
CALIBRATOR = EXPERIMENTAL_CHECKPOINTS / "calibration" / "platt_calibrator.joblib"
KEYS = ["subject", "window_index"]


def frozen_learned(path):
    frame = pd.read_csv(path)
    return frame[frame["policy"] == "Learned Fitted Q"].sort_values(KEYS).reset_index(drop=True)


def grid(spec):
    lo, hi, step = spec
    return np.round(np.arange(lo, hi + step / 2, step), 4)


def candidate_rows(split, inputs, agent, calibrator, config, retrained):
    """All candidate decision rules evaluated on one split."""

    inputs = inputs.sort_values(KEYS).reset_index(drop=True)
    assert (inputs[KEYS].to_numpy() == agent[KEYS].to_numpy()).all()
    y = inputs["unreliable_target"].to_numpy()
    p_cal = calibrator.predict(inputs["unreliable_probability"].to_numpy())
    zeros = np.zeros(len(y))
    fam = config["families"]
    rows = []

    def add(family, variant, flagged, investigations, **extra):
        rewards = decision_rewards(flagged, y, investigations)
        rows.append({"split": split, "family": family, "variant": variant, **extra,
                     **classification_metrics(y, flagged, investigations, rewards)})

    agent_flags = agent["predicted_unreliable"].to_numpy()
    agent_inv = agent["investigations_used"].to_numpy()
    add("original_learned", "original_learned", agent_flags, agent_inv)

    for tau in grid(fam["probability_threshold"]["tau_grid"]):
        add("probability_threshold", f"tau={tau:.2f}", (p_cal >= tau).astype(int), zeros, tau=tau)

    for tau in grid(fam["agent_calibrated_veto"]["tau_grid"]):
        add("agent_calibrated_veto", f"tau={tau:.2f}", (agent_flags & (p_cal >= tau)).astype(int),
            agent_inv, tau=tau)

    for fp in fam["bayes_cost_rule"]["false_positive_penalties"]:
        tau = bayes_threshold(InvestigationRewardConfig(false_positive_penalty=fp))
        add("bayes_cost_rule", f"fp={fp}", (p_cal >= tau).astype(int), zeros, tau=round(tau, 4))

    for (name, seed), frame in retrained.items():
        frame = frame.sort_values(KEYS).reset_index(drop=True)
        add("reward_retrained_policy", name, frame["predicted_unreliable"].to_numpy(),
            frame["investigations_used"].to_numpy(), seed=seed)

    return pd.DataFrame(rows)


def seed_means(frame):
    """Collapse per-seed retrained rows to one mean row per variant (plus seed spread)."""

    seeded = frame[frame["family"] == "reward_retrained_policy"]
    if seeded.empty:
        return frame
    numeric = seeded.select_dtypes("number").columns.drop("seed", errors="ignore")
    means = seeded.groupby(["split", "family", "variant"], as_index=False)[list(numeric)].mean()
    spread = seeded.groupby(["split", "family", "variant"])[["recall", "false_positive_rate", "f1"]].std()
    spread.columns = [f"{c}_seed_sd" for c in spread.columns]
    means = means.merge(spread.reset_index(), on=["split", "family", "variant"])
    means["seed"] = "mean"
    return pd.concat([frame[frame["family"] != "reward_retrained_policy"], means], ignore_index=True)


def select(validation, config):
    rule = config["selection_rule"]
    floor = validation.loc[validation["family"] == "original_learned", "recall"].item() - rule["recall_tolerance"]
    validation = seed_means(validation)
    eligible = validation[(validation["recall"] >= floor) & (validation["family"] != "original_learned")]
    ranked = eligible.sort_values(["false_positive_rate", "f1"], ascending=[True, False])
    per_family = ranked.groupby("family", sort=False).head(1)
    return floor, per_family, ranked.head(1)


def plot_tradeoff(validation, selected, floor, path):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    styles = {"probability_threshold": ("#C99A65", "-"), "agent_calibrated_veto": ("#B96547", "-")}
    for family, (color, ls) in styles.items():
        part = validation[validation["family"] == family].sort_values("false_positive_rate")
        ax.plot(part["false_positive_rate"], part["recall"], color=color, ls=ls, lw=1.8,
                label=family.replace("_", " "))
    for family, marker, color in (("bayes_cost_rule", "s", "#8A7062"),
                                  ("reward_retrained_policy", "D", "#3B2B25"),
                                  ("original_learned", "*", "#B96547")):
        part = validation[validation["family"] == family]
        ax.scatter(part["false_positive_rate"], part["recall"], marker=marker, s=90 if marker != "*" else 260,
                   color=color, label=family.replace("_", " "), zorder=3)
    ax.axhline(floor, ls=":", color="#948277")
    ax.text(0.02, floor + 0.01, f"recall floor {floor:.3f}", color="#948277", fontsize=9)
    ax.scatter(selected["false_positive_rate"], selected["recall"], s=260, facecolors="none",
               edgecolors="#55693F", linewidths=2, label="selected per family", zorder=4)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("Recall")
    ax.set_title("Validation S11–S12: candidate decision rules")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    config = load_config("phase2_fp_reduction")
    set_seed(config["seed"])
    out = output_dir("phase2_fp_reduction")
    calibrator = joblib.load(CALIBRATOR)
    variants = config["families"]["reward_retrained_policy"]["variants"]
    seeds = config["families"]["reward_retrained_policy"]["seeds"]
    missing = [(n, s) for n in variants for s in seeds if not variant_path(n, s).exists()]
    if missing:
        raise SystemExit(f"Train variants first (python -m src.experiments.train_reward_variants): {missing}")

    val_inputs, val_signals = load_split("validation")
    val_retrained = {(n, s): replay_policy(variant_path(n, s), val_inputs, val_signals)
                     for n in variants for s in seeds}
    validation = candidate_rows("validation", val_inputs, frozen_learned(VAL_DECISIONS), calibrator,
                                config, val_retrained)
    validation.to_csv(out / "validation_candidates.csv", index=False)

    floor, per_family, overall = select(validation, config)
    plot_tradeoff(seed_means(validation), per_family, floor, out / "validation_tradeoff.png")

    test_inputs, test_signals = load_split("test")
    chosen_retrained = per_family.loc[per_family["family"] == "reward_retrained_policy", "variant"].tolist()
    test_retrained = {(n, s): replay_policy(variant_path(n, s), test_inputs, test_signals)
                      for n in chosen_retrained for s in seeds}
    for (name, seed), frame in test_retrained.items():
        frame.to_csv(out / f"test_decisions_{name}_seed{seed}.csv", index=False)
    test_all = candidate_rows("test", test_inputs, frozen_learned(TEST_DECISIONS), calibrator, config,
                              test_retrained)
    test_all.to_csv(out / "test_all_candidates_reporting_only.csv", index=False)
    test_all = seed_means(test_all)

    picks = per_family[["family", "variant"]].assign(selected_overall=False)
    picks.loc[picks["variant"].isin(overall["variant"]) & picks["family"].isin(overall["family"]),
              "selected_overall"] = True
    test_selected = test_all.merge(picks, on=["family", "variant"])

    frozen = pd.read_csv(TEST_COMPARISON).assign(family="frozen_d2_baseline", variant=lambda d: d["policy"])
    columns = ["family", "variant", "accuracy", "precision", "recall", "f1", "false_positive_rate",
               "average_investigations", "average_reward"]
    extra = [c for c in ("recall_seed_sd", "false_positive_rate_seed_sd", "f1_seed_sd") if c in test_selected]
    comparison = pd.concat([frozen[columns], test_selected[columns + extra + ["selected_overall"]]],
                           ignore_index=True)
    comparison.to_csv(out / "test_comparison.csv", index=False)

    # Recompute the frozen learned row from frozen decisions as an integrity check.
    learned_check = test_all[test_all["family"] == "original_learned"].iloc[0]
    frozen_learned_row = frozen[frozen["policy"] == "Learned Fitted Q"].iloc[0]
    for key in ("recall", "false_positive_rate", "f1", "average_reward"):
        assert abs(learned_check[key] - frozen_learned_row[key]) < 1e-9, key

    reward_table = pd.concat([
        validation[validation["family"].isin(["reward_retrained_policy", "original_learned"])],
        seed_means(validation[validation["family"] == "reward_retrained_policy"]),
        test_all[test_all["family"] == "reward_retrained_policy"],
    ])
    reward_table.to_csv(out / "reward_sensitivity.csv", index=False)

    summary = {"recall_floor_validation": floor,
               "selected_per_family": per_family[["family", "variant", "recall", "false_positive_rate", "f1"]]
               .to_dict(orient="records"),
               "selected_overall": overall[["family", "variant"]].to_dict(orient="records")}
    with open(out / "selection.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=float)

    inputs = [VAL_DECISIONS, TEST_DECISIONS, TEST_COMPARISON, CALIBRATOR] + [
        variant_path(n, s) for n in variants for s in seeds]
    write_manifest(out, config, inputs, list(out.glob("*.csv")) + list(out.glob("*.png")) + [out / "selection.json"],
                   notes=["Selection used validation only; test rows of non-selected candidates are saved "
                          "for transparency and were not used for any choice."])

    pd.set_option("display.width", 200)
    print(f"Validation recall floor: {floor:.4f}")
    print(per_family[["family", "variant", "recall", "false_positive_rate", "f1", "average_investigations"]]
          .round(4).to_string(index=False))
    print("Test comparison:")
    print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

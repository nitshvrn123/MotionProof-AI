"""D4 Phase 3: moving-block bootstrap confidence intervals for test metrics and paired policy differences.

Blocks of consecutive windows are resampled within each test participant, so
each participant keeps its size and short-range dependence between
overlapping windows is preserved. All policies are scored on the same
resampled windows, so differences are paired. With three participants the
intervals describe variation within these recordings, not the population.

Usage:
    python -m src.experiments.run_bootstrap
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.common import TEST_INPUTS, load_config, output_dir, write_manifest
from src.experiments.d4_common import D4_DECISIONS, POLICIES, counts, load_test_windows, rates
from src.experiments.run_fp_reduction import TEST_DECISIONS

NAME = "d4_bootstrap"


def block_indices(sizes, block, replicates, rng):
    """Moving-block resample indices, concatenated over participants (each keeps its own size)."""

    parts, offset = [], 0
    for n in sizes:
        n_blocks = int(np.ceil(n / block))
        starts = rng.integers(0, n - block + 1, size=(replicates, n_blocks))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(replicates, -1)[:, :n]
        parts.append(idx + offset)
        offset += n
    return np.concatenate(parts, axis=1)


def bootstrap(frame, block, replicates, seed, policies=POLICIES, by_participant=False):
    frame = frame.sort_values(["subject", "window_index"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    if by_participant:
        out = {}
        for subject, part in frame.groupby("subject"):
            out[subject] = bootstrap(part, block, replicates, seed + int(subject), policies)
        return out
    sizes = frame.groupby("subject").size().to_numpy()
    idx = block_indices(sizes, block, replicates, rng)
    y = frame["unreliable_target"].to_numpy()[idx]
    results = {}
    for policy in policies:
        flags = frame[f"flag::{policy}"].to_numpy()[idx]
        results[policy] = rates(*counts(y, flags))
    return results


def interval(values, confidence):
    lo, hi = np.percentile(values, [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100])
    return float(lo), float(hi)


def point(frame, policy):
    return {k: float(v) for k, v in rates(*counts(frame["unreliable_target"], frame[f"flag::{policy}"])).items()}


def main():
    config = load_config("d4_evaluation")["bootstrap"]
    out = output_dir(NAME)
    frame = load_test_windows()
    conf, B = config["confidence"], config["replicates"]

    rows, diffs = [], []
    for block in config["block_sensitivity_windows"]:
        boot = bootstrap(frame, block, B, config["seed"])
        for policy in POLICIES:
            est = point(frame, policy)
            for metric in config["metrics"]:
                lo, hi = interval(boot[policy][metric], conf)
                rows.append({"block_windows": block, "policy": policy, "metric": metric,
                             "estimate": est[metric], "ci_low": lo, "ci_high": hi})
        for a, b in config["comparisons"]:
            ea, eb = point(frame, a), point(frame, b)
            for metric in config["metrics"]:
                d = boot[a][metric] - boot[b][metric]
                lo, hi = interval(d, conf)
                diffs.append({"block_windows": block, "comparison": f"{a} − {b}", "metric": metric,
                              "estimate": ea[metric] - eb[metric], "ci_low": lo, "ci_high": hi,
                              "excludes_zero": bool(lo > 0 or hi < 0)})
    rows, diffs = pd.DataFrame(rows), pd.DataFrame(diffs)

    # per-participant paired differences (primary block length)
    per_part = []
    boots = bootstrap(frame, config["primary_block_windows"], B, config["seed"], by_participant=True)
    for subject, boot in boots.items():
        part = frame[frame.subject == subject]
        for a, b in config["comparisons"]:
            ea, eb = point(part, a), point(part, b)
            for metric in ("recall", "false_positive_rate", "f1"):
                lo, hi = interval(boot[a][metric] - boot[b][metric], conf)
                per_part.append({"participant": f"S{subject}", "comparison": f"{a} − {b}", "metric": metric,
                                 "estimate": ea[metric] - eb[metric], "ci_low": lo, "ci_high": hi,
                                 "excludes_zero": bool(lo > 0 or hi < 0)})
    per_part = pd.DataFrame(per_part)

    # forest plot of paired differences at the primary block length
    prim = diffs[diffs.block_windows == config["primary_block_windows"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    comps = [f"{a} − {b}" for a, b in config["comparisons"]]
    for ax, metric in zip(axes, ("recall", "false_positive_rate", "f1")):
        part = prim[prim.metric == metric].set_index("comparison").loc[comps]
        ypos = np.arange(len(comps))[::-1]
        ax.errorbar(part.estimate * 100, ypos, xerr=[(part.estimate - part.ci_low) * 100, (part.ci_high - part.estimate) * 100],
                    fmt="o", color="#2a78d6", ecolor="#2a78d6", capsize=3)
        ax.axvline(0, color="#8a8984", lw=0.8)
        ax.set_title(metric.replace("_", " ") + " difference (points)")
        ax.set_yticks(ypos, comps)
    fig.tight_layout()
    paths = [out / "metric_intervals.csv", out / "paired_differences.csv", out / "paired_differences_by_participant.csv",
             out / "paired_differences.png"]
    fig.savefig(paths[3], dpi=150)
    plt.close(fig)
    rows.to_csv(paths[0], index=False)
    diffs.to_csv(paths[1], index=False)
    per_part.to_csv(paths[2], index=False)
    write_manifest(out, config, [TEST_INPUTS, TEST_DECISIONS, D4_DECISIONS], paths)

    show = prim.copy()
    for c in ("estimate", "ci_low", "ci_high"):
        show[c] = (show[c] * 100).round(1)
    print(f"Paired differences, block = {config['primary_block_windows']} windows, B = {B} (percentage points)\n",
          show.to_string(index=False))
    sens = diffs[(diffs.comparison == "Learned Fitted Q − Fixed Rule") & (diffs.metric == "recall")]
    print("\nBlock-length sensitivity (Learned − Fixed recall)\n", sens.round(4).to_string(index=False))
    pp = per_part[per_part.comparison == "Learned Fitted Q − Fixed Rule"]
    print("\nPer participant (Learned − Fixed)\n", pp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

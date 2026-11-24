"""Finite-horizon fitted Q training for the investigation policy.

Consolidated from notebooks/setup_2.ipynb (cell "FULL FITTED Q TRAINING"),
which produced checkpoints/learned_policy_fitted_q_full.pt. Hyperparameters,
normalization, backward induction and the checkpoint format are unchanged.
The one addition is an optional reward configuration so that reward
sensitivity can be studied without editing code.

Training data: results/metrics/policy_investigation_states_full.csv, or the
compressed .csv.gz copy in the repository (S1-S10 only, every investigation
prefix of every window).

Usage (writes to checkpoints/experimental/, never over the original):
    python -m src.agents.fitted_q_training --fp-penalty -2.0 --name fp2.0
"""

import argparse
import ast
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.agents.investigation_reward import InvestigationRewardConfig
from src.agents.learned_policy import ACTION_ORDER, STATE_DIM, STATE_FEATURE_NAMES
from src.agents.policy_network import PolicyQNetwork

ROOT = Path(__file__).resolve().parents[2]
STATES_CSV = ROOT / "results" / "metrics" / "policy_investigation_states_full.csv"

CACHE = ROOT / "results" / "experiments" / "cache" / "fitted_q_states.npz"
ORIGINAL_CHECKPOINT = ROOT / "checkpoints" / "learned_policy_fitted_q_full.pt"

SEED = 42
BUDGET = 2
EPOCHS_PER_STAGE = 50
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
INVESTIGATION_INDICES = (2, 3, 4)
ACTION_NAMES = [a.value for a in ACTION_ORDER]


def _total_reward(value):
    return float(ast.literal_eval(value)["total_reward"])


def build_state_cache(states_csv=STATES_CSV, cache=CACHE):
    """Parse the 432,150-state table once into compact arrays."""

    states_csv = Path(states_csv)
    if not states_csv.exists() and Path(f"{states_csv}.gz").exists():
        states_csv = Path(f"{states_csv}.gz")
    df = pd.read_csv(states_csv, keep_default_na=False)
    assert df.shape == (432150, 40), df.shape
    assert set(df["subject"].unique()) == set(range(1, 11))
    assert ACTION_NAMES == ["ACCEPT", "FLAG_UNRELIABLE", "CHECK_PPG", "CHECK_MOTION",
                            "CHECK_CONSISTENCY"]

    sequences = df["sequence"].astype(str).to_numpy()
    depth = df["investigations_used"].to_numpy(dtype=np.int8)
    subjects = df["subject"].to_numpy(dtype=np.int16)
    windows = df["window_index"].to_numpy(dtype=np.int32)
    masks = df[[f"valid_{a}" for a in ACTION_NAMES]].to_numpy(dtype=np.bool_)

    lookup = {(int(s), int(w), q): i for i, (s, w, q) in enumerate(zip(subjects, windows, sequences))}
    assert len(lookup) == len(df), "duplicate states"
    transition = np.full((len(df), len(ACTION_NAMES)), -1, dtype=np.int32)

    for i in range(len(df)):
        if depth[i] == BUDGET:
            continue
        for a in INVESTIGATION_INDICES:
            if not masks[i, a]:
                continue
            nxt = ACTION_NAMES[a] if sequences[i] == "" else f"{sequences[i]}|{ACTION_NAMES[a]}"
            j = lookup[(int(subjects[i]), int(windows[i]), nxt)]
            assert depth[j] == depth[i] + 1
            transition[i, a] = j

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        states=df[list(STATE_FEATURE_NAMES)].to_numpy(dtype=np.float32),
        masks=masks,
        depth=depth,
        subjects=subjects,
        windows=windows,
        target=df["unreliable_target"].to_numpy(dtype=np.int8),
        transition=transition,
        stored_accept=np.array([_total_reward(v) for v in df["accept_reward"]], dtype=np.float32),
        stored_flag=np.array([_total_reward(v) for v in df["flag_reward"]], dtype=np.float32),
    )
    return load_state_cache(cache)


def load_state_cache(cache=CACHE):
    if not Path(cache).exists():
        return build_state_cache(cache=cache)
    with np.load(cache) as data:
        return {key: data[key] for key in data.files}


def terminal_rewards(target, investigations_used, config):
    """Vectorized equivalent of calculate_investigation_reward."""

    target = np.asarray(target).astype(bool)
    cost = config.investigation_cost * np.asarray(investigations_used, dtype=np.float32)
    accept = np.where(target, config.false_negative_penalty, config.true_negative_reward) - cost
    flag = np.where(target, config.true_positive_reward, config.false_positive_penalty) - cost
    return accept.astype(np.float32), flag.astype(np.float32)


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_stage(x, y, mask, device, log=None):
    network = PolicyQNetwork(hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    x, y, mask = (torch.from_numpy(a).to(device) for a in (x, y, mask))
    losses = []

    for epoch in range(EPOCHS_PER_STAGE):
        network.train()
        order = torch.randperm(len(x), device=device)
        total, count = 0.0, 0
        for start in range(0, len(x), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            sq = (network(x[batch]) - y[batch]).square()
            bmask = mask[batch]
            loss = sq[bmask].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n = int(bmask.sum())
            total += loss.item() * n
            count += n
        losses.append(total / count)
        if log and (epoch == 0 or (epoch + 1) % 10 == 0):
            log(f"epoch {epoch + 1:02d}/{EPOCHS_PER_STAGE} loss {losses[-1]:.5f}")

    network.eval()
    return network, losses


def train_fitted_q(reward_config=None, seed=SEED, device="cpu", log=print, data=None):
    """Backward-induction fitted Q training. Returns (checkpoint dict, history frame)."""

    reward_config = reward_config or InvestigationRewardConfig()
    data = data or load_state_cache()
    _set_seed(seed)
    device = torch.device(device)

    states = data["states"]
    mean = states.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = states.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    normalized = (states - mean) / std
    masks, depth, transition = data["masks"], data["depth"], data["transition"]
    reward_accept, reward_flag = terminal_rewards(data["target"], depth, reward_config)

    networks, history = {}, []
    for stage in range(BUDGET, -1, -1):
        idx = np.flatnonzero(depth == stage)
        targets = np.zeros((len(idx), len(ACTION_NAMES)), dtype=np.float32)
        targets[:, 0] = reward_accept[idx]
        targets[:, 1] = reward_flag[idx]

        if stage < BUDGET:
            nxt_net = networks[stage + 1]
            for a in INVESTIGATION_INDICES:
                local = np.flatnonzero(masks[idx, a])
                if not len(local):
                    continue
                nxt = transition[idx[local], a]
                assert (nxt >= 0).all()
                values = np.empty(len(nxt), dtype=np.float32)
                with torch.no_grad():
                    for start in range(0, len(nxt), BATCH_SIZE):
                        sl = nxt[start:start + BATCH_SIZE]
                        q = nxt_net(torch.from_numpy(normalized[sl]).to(device))
                        q = q.masked_fill(~torch.from_numpy(masks[sl]).to(device), -torch.inf)
                        values[start:start + len(sl)] = q.max(dim=1).values.cpu().numpy()
                targets[local, a] = values

        stage_log = (lambda m, s=stage: log(f"stage {s} | {m}")) if log else None
        network, losses = _train_stage(normalized[idx], targets, masks[idx], device, stage_log)
        networks[stage] = network
        history += [{"stage": stage, "epoch": e, "loss": l, "states": len(idx)}
                    for e, l in enumerate(losses, start=1)]

    checkpoint = {
        "stage_model_state_dicts": {
            s: {k: v.detach().cpu() for k, v in n.state_dict().items()} for s, n in networks.items()
        },
        "feature_mean": torch.from_numpy(mean.copy()),
        "feature_std": torch.from_numpy(std.copy()),
        "state_feature_names": tuple(STATE_FEATURE_NAMES),
        "action_names": tuple(ACTION_NAMES),
        "state_dim": STATE_DIM,
        "action_dim": len(ACTION_NAMES),
        "hidden_dim": HIDDEN_DIM,
        "budget": BUDGET,
        "training_method": "finite_horizon_fitted_q_learning",
        "training_subjects": list(range(1, 11)),
        "training_windows": int((depth == 0).sum()),
        "training_states": int(len(depth)),
        "seed": seed,
        "reward_config": asdict(reward_config),
    }
    return checkpoint, pd.DataFrame(history)


def save_checkpoint(checkpoint, path):
    path = Path(path).resolve()
    if path == ORIGINAL_CHECKPOINT.resolve():
        raise RuntimeError("Refusing to overwrite the original fitted Q checkpoint.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--fp-penalty", type=float, default=-1.5)
    parser.add_argument("--fn-penalty", type=float, default=-2.0)
    parser.add_argument("--tp-reward", type=float, default=2.0)
    parser.add_argument("--investigation-cost", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    config = InvestigationRewardConfig(
        true_positive_reward=args.tp_reward,
        false_positive_penalty=args.fp_penalty,
        false_negative_penalty=args.fn_penalty,
        investigation_cost=args.investigation_cost,
    )
    start = time.time()
    checkpoint, history = train_fitted_q(config, seed=args.seed)
    out = ROOT / "checkpoints" / "experimental" / "policies" / f"fitted_q_{args.name}.pt"
    save_checkpoint(checkpoint, out)
    history.to_csv(out.with_suffix(".history.csv"), index=False)
    print(f"Saved {out.relative_to(ROOT)} in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()

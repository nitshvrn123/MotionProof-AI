"""Train the Phase 2 reward-variant policies listed in the experiment config.

Skips variants whose checkpoint already exists. Never touches the original
checkpoint.

Usage:
    python -m src.experiments.train_reward_variants
"""

import time

from src.agents.fitted_q_training import load_state_cache, save_checkpoint, train_fitted_q
from src.agents.investigation_reward import InvestigationRewardConfig
from src.experiments.common import EXPERIMENTAL_CHECKPOINTS, load_config

POLICY_DIR = EXPERIMENTAL_CHECKPOINTS / "policies"


def variant_path(name, seed=42):
    suffix = "" if seed == 42 else f"_seed{seed}"
    return POLICY_DIR / f"fitted_q_{name}{suffix}.pt"


def main():
    config = load_config("phase2_fp_reduction")
    variants = config["families"]["reward_retrained_policy"]["variants"]
    data = load_state_cache()

    seeds = config["families"]["reward_retrained_policy"]["seeds"]
    for name, spec in variants.items():
        for seed in seeds:
            path = variant_path(name, seed)
            if path.exists():
                print(f"{name} seed {seed}: exists, skipping")
                continue
            start = time.time()
            reward = InvestigationRewardConfig(**spec)
            checkpoint, history = train_fitted_q(reward, seed=seed, data=data, log=None)
            save_checkpoint(checkpoint, path)
            history.to_csv(path.with_suffix(".history.csv"), index=False)
            print(f"{name} seed {seed}: trained in {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

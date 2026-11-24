#!/usr/bin/env bash
# Re-run every post-D2 experiment from the project root, in order.
# Requires data/processed/{val,test}/*.npz (not in the repository). Training states are read from
# results/metrics/policy_investigation_states_full.csv.gz, which is included.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.experiments.verify_frozen
python -m src.evaluation.build_agent_traces          # Verified agent replay for the app (~10 s)
python -m src.experiments.run_calibration            # Phase 1 (~5 s)
python -m src.experiments.train_reward_variants      # Phase 2 policies, 18 runs (~1-2 min each on CPU)
python -m src.experiments.run_fp_reduction           # Phase 2 selection + test (~3 min)
python -m src.experiments.run_robustness             # Phase 3 (~15-25 min on CPU)
python -m src.experiments.verify_frozen
python -m pytest -q

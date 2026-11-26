#!/usr/bin/env bash
# Re-run every post-D2 experiment (D3 Phases 1-3, then the D4 extended evaluation) from the project root, in order.
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
python -m src.experiments.run_motion_thresholds      # D4.1 motion-aware veto, selected on validation only
python -m src.experiments.run_participant_analysis   # D4.2 per-participant test results
python -m src.experiments.run_bootstrap              # D4.3 moving-block bootstrap intervals
python -m src.experiments.run_failure_cases          # D4.4 failure-case analysis
python -m src.experiments.verify_frozen
python -m pytest -q

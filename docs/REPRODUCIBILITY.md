# Reproducibility guide

How to rebuild the MotionProof AI environment, check that the frozen D2 results are intact, and rerun every post-D2 experiment.

## 1. Environment

The checkpoints were verified with Python 3.12 and the exact versions in `requirements.txt`. scikit-learn must be 1.4.0, because the reliability classifier is a pickled `HistGradientBoostingClassifier`.

```bash
cd MotionProof-AI-VSCode
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.experiments.verify_frozen
python -m pytest -q
```

On a machine without a GPU everything runs on CPU. No step needs CUDA.

## 2. Data and splits

The splits are by participant and never mix windows from one person across splits.

| Split | Participants | Windows | Used for |
|---|---|---|---|
| Training | S1–S10 | 43,215 | CNN fold training, reliability classifier, fitted Q policy |
| Validation | S11, S12 | 8,475 (4,521 + 3,954) | Threshold choices, calibration fitting, Phase 2 selection |
| Held-out test | S13, S14, S15 | 13,007 (4,565 + 4,476 + 3,966) | Reporting only |

Windows are 8 seconds long. PPG is 512 samples at 64 Hz and the accelerometer is 256 × 3 samples at 32 Hz. A prediction is unreliable when the CNN's absolute error against the ECG reference exceeds 10 BPM.

Processed files expected locally:

```
data/processed/val/S11.npz, S12.npz      keys: ppg, acc, hr, subject, split
data/processed/test/S13.npz … S15.npz
```

Included in the repository: `results/metrics/policy_investigation_states_full.csv.gz` (S1–S10 training states, 11.9 MB; the training code reads it directly), the derived validation and test inputs, and all frozen evaluation files. The `.npz` participant windows are not included; see the README.

Raw PPG-DaLiA recordings (`data/raw/PPG_FieldStudy/S*/S*.pkl`, CC BY 4.0, from the UCI repository) are needed only to rebuild the `.npz` files, using `notebooks/setup.ipynb`. The processed files do not contain activity labels.

## 3. Random seeds

| Component | Seed | Where |
|---|---|---|
| Fitted Q policy (production) | 42 | `src/agents/fitted_q_training.py` (from `notebooks/setup_2.ipynb`) |
| Reliability classifier | `random_state=42` | `src/evaluation/reliability_nonlinear.py` |
| Phase 2 reward variants | 42, 43, 44 | `configs/experiments/phase2_fp_reduction.json` |
| Phase 3 perturbations | 42 + 1000 × condition + subject | `configs/experiments/phase3_robustness.json` |

## 4. Checkpoints

Production checkpoints. Never overwrite these; `src/agents/fitted_q_training.py` refuses to.

| File | Role | SHA-256 (first 12) |
|---|---|---|
| `checkpoints/ppg_cnn_best.pt` | Heart-rate CNN | `af193b6d5fe3` |
| `checkpoints/reliability_classifier_nonlinear_full.joblib` | Reliability classifier (S1–S10) | `82f98d732a09` |
| `checkpoints/learned_policy_fitted_q_full.pt` | Fitted Q investigation policy (S1–S10) | `7058bb0d94f7` |

These hashes match `results/metrics/final_test/reproducibility_manifest.json` from D2. All frozen hashes, including every file in `results/metrics/final_test/`, are listed in `configs/frozen_artifacts.json` and checked by `python -m src.experiments.verify_frozen` and by the test suite.

Experimental artifacts live separately:

```
checkpoints/experimental/calibration/{platt,isotonic}_calibrator.joblib
checkpoints/experimental/policies/fitted_q_<variant>[_seed43|_seed44].pt (+ .history.csv)
```

Other files in `checkpoints/` (pilot policies, multimodal CNNs, fold CNNs) are earlier development artifacts and are not used by the final system.

## 5. Rerunning experiments

Run everything in order:

```bash
bash scripts/reproduce_experiments.sh
```

Or run one step at a time:

```bash
python -m src.experiments.verify_frozen          # frozen D2 files unchanged
python -m src.evaluation.build_agent_traces      # app trace cache, verified against frozen decisions
python -m src.experiments.run_calibration        # Phase 1
python -m src.experiments.train_reward_variants  # Phase 2 policies (skips existing)
python -m src.experiments.run_fp_reduction       # Phase 2 selection and test evaluation
python -m src.experiments.run_robustness         # Phase 3
python -m pytest -q
```

Retrain a single policy with a custom reward:

```bash
python -m src.agents.fitted_q_training --name my_variant --fp-penalty -2.0 --investigation-cost 0.1 --seed 42
```

## 6. Configuration tracking and outputs

Each experiment reads one JSON file in `configs/experiments/`. Each run writes `manifest.json` next to its outputs, with the config, the SHA-256 of every input file, the list of outputs and the package versions.

```
results/
  metrics/            frozen D2 evaluation (read-only)
  demo/               learned_agent_traces.csv for the app
  experiments/
    phase1_calibration/
    phase2_fp_reduction/
    phase3_robustness/
    cache/            parsed training states (rebuilt automatically, safe to delete)
```

## 7. Known sources of nondeterminism

Retraining the production policy with its original procedure on this CPU reproduces the training losses closely but not exactly: the largest per-epoch difference is 0.058. 91.2% of held-out decisions agree with the production checkpoint. Floating-point order differs between the original GPU run and CPU, and the Q-values of different investigation actions are often nearly tied. This is why Phase 2 compares reward variants against a policy retrained with the original reward under the same conditions (`original_reward_retrain`), over three seeds, rather than against the production checkpoint.

Live inference in the app matches the frozen evaluation: CNN heart rate within 0.009 BPM and classifier probability within 0.031. Frozen agent decisions are reproduced on every test window checked (13,007 of 13,007 in the trace cache) and on all 8,475 validation windows.

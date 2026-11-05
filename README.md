# MotionProof AI

## Adaptive Agent Investigation for Reliable Heart Rate Estimation Using Multimodal Deep Learning

MotionProof AI investigates the reliability of heart rate estimates from wrist-worn photoplethysmography (PPG) signals, particularly during physical movement.

The project combines multimodal deep learning, signal quality analysis, and an adaptive investigation agent to identify potentially unreliable heart rate predictions.

## Research Question

Can a learned adaptive investigation policy improve heart rate estimation reliability compared with a fixed-rule policy when both have access to the same information and investigation tools?

## Dataset

**PPG-DaLiA: PPG Dataset for Motion Artifact Detection and Heart Rate Estimation**

The dataset contains recordings from 15 participants performing everyday activities.

The primary signals used are:

- Wrist PPG sampled at 64 Hz
- Wrist accelerometer sampled at 32 Hz
- ECG-derived reference heart rate

The signals are segmented into 8-second windows with a 2-second stride, following the official dataset documentation.

ECG-derived heart rate is used as an offline reference label and is not provided as an input to the deployed prediction model.

## Proposed Architecture

1. **Data preprocessing:** Load synchronized PPG and accelerometer recordings and generate aligned windows.
2. **Heart rate estimation:** Train a multimodal deep learning model to estimate heart rate.
3. **Reliability assessment:** Estimate whether a prediction is likely to exceed an error threshold of 10 BPM.
4. **Adaptive investigation:** Learn a policy that selects whether to accept a prediction, investigate further, use an alternative estimator, or flag uncertainty.
5. **Evaluation:** Compare the learned policy against a fixed-rule policy using subject-independent evaluation.

The deep learning models and adaptive investigation policy are planned components and have not yet been implemented.

## Current Implementation

The following components have been implemented:

- PPG-DaLiA dataset loading
- Synchronized PPG and accelerometer window generation
- Signal-processing heart rate baseline
- Motion intensity analysis
- Motion cadence estimation
- PPG spectral quality features
- Reliability feature extraction
- Automated preprocessing tests

### Preliminary Results

Initial experiments were conducted using participant S1.

| Metric | Result |
|---|---:|
| Baseline MAE | 10.15 BPM |
| Baseline RMSE | 16.56 BPM |
| Absolute error >10 BPM | 32.61% |
| Spearman correlation between motion and absolute error | 0.4965 |
| Candidate cadence-locking windows | 611 |

These results are exploratory findings from a single participant and are not representative of final cross-subject model performance.

## Project Structure

```text
MotionProof-AI/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── setup.ipynb
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── agents/
│   └── evaluation/
├── tests/
│   └── test_preprocessing.py
├── ui/
├── docs/
├── results/
│   ├── metrics/
│   └── plots/
├── checkpoints/
├── requirements.txt
└── README.md
```

## Running the Preprocessing Tests

From the project root:

```bash
python -m pytest tests/test_preprocessing.py -v
```

The initial preprocessing test suite contains nine tests.

## Evaluation Plan

The project will use participant-independent evaluation, including leave-one-subject-out testing and nested validation where required for model selection.

Planned evaluation metrics include:

- Mean absolute error (MAE)
- Root mean squared error (RMSE)
- Percentage of predictions exceeding 10 BPM error
- Reliability classification performance
- Selective MAE versus prediction coverage
- Performance under different motion intensities
- Cadence-locking analysis

The adaptive and fixed-rule policies will be compared using matched information and investigation budgets.

## Responsible AI Considerations

MotionProof AI is a research prototype and is not intended for medical diagnosis or clinical decision-making.

The system will report uncertainty, investigate performance differences across participants and activities, and avoid using ECG-derived reference labels during inference.

## Status

**In development — November 2026**

Current milestone: Deliverable 1.2, Foundation and Technical Design.
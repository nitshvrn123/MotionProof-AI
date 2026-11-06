# MotionProof AI

## Adaptive Agent Investigation for Reliable Heart Rate Estimation Using Multimodal Deep Learning

MotionProof AI investigates the reliability of heart rate estimates from wrist-worn photoplethysmography (PPG) signals, particularly during physical movement.

The project combines multimodal deep learning, signal quality analysis, and an adaptive investigation agent to identify potentially unreliable heart rate predictions.

## Author

**Nitishwar Vasantha Kumar**  
M.S. Applied Data Science  
University of Florida  
Email: supermanspider2511@gmail.com

## Research Question

Can a learned adaptive investigation policy improve heart rate estimation reliability compared with a fixed-rule policy when both have access to the same information and investigation tools?

## Dataset

**PPG-DaLiA: PPG Dataset for Motion Artifact Detection and Heart Rate Estimation**

The project uses the publicly available PPG-DaLiA dataset from the UCI Machine Learning Repository.

The dataset contains recordings from 15 participants performing everyday activities.

The primary signals used are:

- Wrist PPG sampled at 64 Hz
- Wrist accelerometer sampled at 32 Hz
- ECG-derived reference heart rate

The signals are segmented into 8-second windows with a 2-second stride, following the official dataset documentation.

Each window contains:

- 512 PPG samples
- 256 accelerometer samples across three axes
- One ECG-derived reference heart rate label

ECG-derived heart rate is used as an offline reference label for training and evaluation. It is not provided as an input to the deployed prediction model.

### Dataset Format and Access

The dataset is distributed as compressed archives containing participant recordings.

Each participant recording is stored as a Python pickle file.

Example:

`data/raw/PPG_FieldStudy/S1/S1.pkl`

The recordings contain synchronized physiological signals, activity information, and reference heart rate labels.

The full dataset is not included in this GitHub repository because of its size.

Dataset source:

https://archive.ics.uci.edu/dataset/495/ppg+dalia

**Security note:** Python pickle files should only be loaded from trusted sources.

## Installation and Setup

### 1. Clone the Repository

Clone the project repository:

```bash
git clone https://github.com/nitshvrn123/MotionProof-AI.git
cd MotionProof-AI
```

### 2. Install Dependencies

Python 3.10 or later is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

The initial environment uses:

- NumPy
- SciPy
- Pandas
- Matplotlib
- pytest
- pypdf

JupyterLab is required to open and execute the setup notebook. Install it if it is not already available:

```bash
python -m pip install jupyterlab
```

Additional deep learning and interface dependencies will be introduced during subsequent implementation stages.

### 3. Download and Prepare the Dataset

Download PPG-DaLiA from the UCI Machine Learning Repository.

Extract the participant recordings into:

```text
MotionProof-AI/
└── data/
    └── raw/
        └── PPG_FieldStudy/
            ├── S1/
            │   └── S1.pkl
            ├── S2/
            │   └── S2.pkl
            └── ...
```

For the initial setup notebook, participant S1 is sufficient.

The required file is:

```text
data/raw/PPG_FieldStudy/S1/S1.pkl
```

The complete dataset will be used for participant-independent training and evaluation during later project stages.

### 4. Run the Setup Notebook

Launch JupyterLab:

```bash
jupyter lab
```

Open:

```text
notebooks/setup.ipynb
```

Execute the notebook cells sequentially.

The notebook performs the following operations:

1. Loads the PPG-DaLiA participant S1 recording.
2. Verifies signal structure, dimensions, and sampling frequencies.
3. Computes summary statistics for ECG-derived reference heart rate.
4. Visualizes PPG, accelerometer, and reference heart rate signals.
5. Generates synchronized 8-second windows with a 2-second stride.
6. Implements an initial signal-processing heart rate estimator.
7. Evaluates baseline prediction errors.
8. Analyzes relationships between motion intensity and prediction error.
9. Estimates motion cadence and identifies candidate cadence-locking windows.
10. Extracts preliminary reliability features for future model development.

Generated figures and numerical outputs are stored in:

```text
results/plots/
results/metrics/
```

**HiPerGator environment note:**

The current notebook uses an absolute project path configured for the University of Florida HiPerGator environment:

```python
PROJECT_ROOT = Path("/blue/eee6778/nvasanthakumar/MotionProof-AI")
```

Users running the notebook in another environment must update `PROJECT_ROOT` to their local repository location before execution.

### 5. Run Automated Tests

From the project root, execute:

```bash
python -m pytest tests/test_preprocessing.py -v
```

The initial preprocessing test suite contains nine tests covering:

- Window dimensions and synchronization
- PPG-based heart rate estimation
- Accelerometer motion intensity
- Motion cadence estimation
- PPG spectral quality analysis
- Reliability feature extraction
- Invalid signal handling

All nine initial preprocessing tests passed in the development environment.

## Proposed Architecture

MotionProof AI is designed as a multimodal heart rate estimation and reliability investigation system.

The planned architecture consists of five primary components.

### 1. Data Preprocessing

Load synchronized PPG and accelerometer recordings, validate signal dimensions, and generate aligned windows.

Preprocessing also includes signal filtering, motion analysis, and feature extraction.

### 2. Heart Rate Estimation

Train and evaluate a multimodal deep learning model using wrist PPG and accelerometer signals.

Candidate architectures include:

- PPG-only 1D convolutional neural network
- Multimodal 1D convolutional neural network
- Attention-enhanced multimodal model

The models will be compared against the initial signal-processing baseline.

### 3. Reliability Assessment

Estimate whether a predicted heart rate is likely to exceed an absolute error threshold of 10 beats per minute.

Potential reliability inputs include:

- Predicted heart rate
- Motion intensity
- Motion cadence
- Difference between predicted heart rate and motion cadence
- PPG spectral concentration
- PPG spectral entropy
- Change in predicted heart rate across consecutive windows

The reliability model will be trained using offline reference labels while ensuring that ground-truth heart rate and prediction error are not used as inference features.

### 4. Adaptive Investigation Agent

Develop a learned investigation policy that selects an action based on the estimated reliability of each prediction.

Possible actions include:

1. Accept the current prediction.
2. Investigate additional signal quality information.
3. Use an alternative heart rate estimator.
4. Flag the prediction as unreliable.

The investigation policy will be compared against a fixed-rule baseline.

Both policies will receive the same available information and investigation tools, with matched investigation budgets.

### 5. User Interface

Develop an interactive Streamlit application for visualizing heart rate predictions and reliability decisions.

The planned interface will display:

- Selected participant and signal window
- PPG waveform
- Accelerometer motion information
- Estimated heart rate
- Reliability assessment
- Agent investigation decision
- Supporting signal quality indicators
- Comparison between learned and fixed-rule policies

The interface will be used to demonstrate system behavior and improve the interpretability of agent decisions.

**Implementation status:** The deep learning models, learned investigation policy, and Streamlit interface are planned components and have not yet been implemented.

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

The cadence-locking count represents candidate cases identified using motion frequency and offline prediction error. It should not be interpreted as a confirmed count of cadence-locking events.

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
│   │   ├── loader.py
│   │   ├── windows.py
│   │   └── features.py
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

## Evaluation Plan

The project will use participant-independent evaluation, including leave-one-subject-out testing and nested validation for model selection.

Planned evaluation metrics include:

- Mean absolute error (MAE)
- Root mean squared error (RMSE)
- Percentage of predictions exceeding 10 BPM error
- Reliability classification performance
- Selective MAE versus prediction coverage
- Performance under different motion intensities
- Cadence-locking analysis

The adaptive and fixed-rule policies will be compared using matched information and investigation budgets.

All model selection, reliability calibration, and policy tuning will be performed using training participants without accessing the held-out test participant.

## Implementation Timeline

| Week | Dates | Focus | Expected Outcome |
|---|---|---|---|
| 1 | Nov 5–11 | Foundation and design | GitHub repository, setup notebook, preprocessing pipeline, technical blueprint |
| 2 | Nov 12–18 | Baseline training and UI prototype | Initial multimodal model, preliminary evaluation, working interface |
| 3 | Nov 19–25 | Reliability modeling and adaptive investigation | Reliability classifier and initial learned versus fixed-rule comparison |
| 4 | Nov 26–Dec 2 | Model evaluation and refinement | Participant-independent evaluation, ablation studies, improved agent behavior |
| 5 | Dec 3–9 | Integration and final reporting | Stable interactive demonstration, final report, presentation |

## Anticipated Technical Challenges

### 1. Motion Artifacts

Physical movement can introduce strong artifacts into wrist PPG recordings, resulting in inaccurate heart rate predictions.

**Mitigation:** Combine PPG and accelerometer information, analyze motion cadence, and compare signal-processing and deep learning estimators.

### 2. Generalization Across Participants

Physiological differences and activity patterns may reduce model performance on unseen participants.

**Mitigation:** Use participant-independent evaluation and nested validation to prevent information leakage.

### 3. Reliability and Agent Decision Quality

A learned investigation policy may make unnecessary decisions or fail to identify unreliable predictions.

**Mitigation:** Compare the learned policy against fixed rules using identical information, available tools, and investigation budgets. Evaluate both prediction accuracy and coverage.

## Responsible AI Considerations

MotionProof AI is a research prototype and is not intended for medical diagnosis or clinical decision-making.

The project considers the following responsible AI principles:

- **Transparency:** Display reliability estimates, investigation decisions, and supporting signal quality indicators.
- **Fairness:** Examine performance differences across participants and activities.
- **Privacy:** Use the publicly available research dataset without collecting additional personal physiological data.
- **Reliability:** Flag uncertain predictions rather than presenting every estimate as trustworthy.
- **Data leakage prevention:** Never provide ECG-derived reference heart rate or ground-truth prediction error as inference inputs.
- **Environmental responsibility:** Use computationally efficient baselines and limit unnecessary training experiments.

The system will not claim clinical accuracy or medical reliability based on experimental results alone.

## Status

**In development — November 2026**

Current milestone: Deliverable 1.2, Foundation and Technical Design.

### Completed

- GitHub repository initialized and initial commit pushed
- PPG-DaLiA participant S1 loaded successfully
- Signal-processing baseline implemented
- Motion and cadence analysis completed
- Reliability features extracted
- Reusable preprocessing modules implemented
- Nine preprocessing tests passed

### Upcoming

- Technical blueprint PDF
- Multimodal deep learning model
- Reliability prediction model
- Adaptive investigation policy
- Fixed-rule policy comparison
- Streamlit interface
- Participant-independent evaluation
- Final demonstration and report
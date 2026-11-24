"""Data loading and inference for the D3 interface.

The inference path is the same one used by app.py: CNN -> reliability
features -> nonlinear classifier -> InvestigationEnvironment -> fitted Q
policy. The reference ECG heart rate is never passed to any of these steps.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.agents.fitted_q_inference import load_fitted_q_policy
from src.agents.investigation_environment import InvestigationEnvironment
from src.evaluation.generate_validation_predictions import load_cnn, predict_cnn
from src.evaluation.reliability_baseline import FEATURE_COLUMNS
from src.evaluation.reliability_features import extract_reliability_features

ROOT = Path(__file__).resolve().parents[1]
FINAL_TEST = ROOT / "results" / "metrics" / "final_test"
TRACES_PATH = ROOT / "results" / "demo" / "learned_agent_traces.csv"

SUBJECTS = (13, 14, 15)
PPG_RATE_HZ = 64
ACC_RATE_HZ = 32
BUDGET = 2
HISTORY_LENGTH = 5


@st.cache_data(show_spinner=False)
def load_subject(subject):
    path = ROOT / "data" / "processed" / "test" / f"S{subject}.npz"

    if not path.exists():
        raise FileNotFoundError(f"Missing processed recording: {path}")

    with np.load(path) as data:
        return {
            "ppg": data["ppg"].copy(),
            "acc": data["acc"].copy(),
            "hr": data["hr"].copy(),
        }


@st.cache_data(show_spinner=False)
def load_results():
    """Frozen held-out evaluation outputs. Read-only."""

    def read(name):
        path = FINAL_TEST / name
        return pd.read_csv(path) if path.exists() else None

    return {
        "comparison": read("test_policy_comparison.csv"),
        "inputs": read("test_policy_inputs_nonlinear.csv"),
        "per_subject": read("test_policy_per_subject.csv"),
        "budget_ablation": read("investigation_budget_ablation.csv"),
        "action_ablation": read("investigation_action_ablation.csv"),
    }


@st.cache_data(show_spinner=False)
def load_traces():
    """Verified replay of the learned policy on every held-out window.

    Built by src/evaluation/build_agent_traces.py. Returns None if absent.
    """

    if not TRACES_PATH.exists():
        return None

    return pd.read_csv(TRACES_PATH)


@st.cache_resource(show_spinner=False)
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cnn, mean, std = load_cnn(device)

    classifier = joblib.load(
        ROOT / "checkpoints" / "reliability_classifier_nonlinear_full.joblib"
    )

    agent = load_fitted_q_policy(
        ROOT / "checkpoints" / "learned_policy_fitted_q_full.pt",
        device="cpu",
    )

    return cnn, mean, std, classifier, agent, device


def get_history(subject, window, inputs):
    """Previous CNN predictions (never reference HR), as in evaluation."""

    rows = inputs[
        (inputs["subject"] == subject)
        & (inputs["window_index"] < window)
    ].sort_values("window_index")

    return rows.tail(HISTORY_LENGTH)["cnn_predicted_hr"].tolist()


def run_inference(ppg, acc, history):
    cnn, mean, std, classifier, agent, device = load_models()

    prediction = predict_cnn(
        model=cnn,
        ppg=ppg.reshape(1, -1),
        mean=mean,
        std=std,
        device=device,
    )
    predicted_hr = float(np.asarray(prediction).reshape(-1)[0])

    features = extract_reliability_features(
        ppg.reshape(1, -1),
        acc.reshape(1, 256, 3),
    )
    feature_frame = pd.DataFrame(
        np.column_stack([features, [predicted_hr]]),
        columns=FEATURE_COLUMNS,
    )
    probability = float(classifier.predict_proba(feature_frame)[0, 1])

    environment = InvestigationEnvironment(
        ppg_window=ppg,
        acc_window=acc,
        cnn_predicted_hr=predicted_hr,
        unreliable_probability=probability,
        previous_hr_predictions=history,
        budget=BUDGET,
    )

    result = agent.run(environment)

    return predicted_hr, probability, result


def analyze_window(subject, window):
    """Run the full pipeline for one window and package the outputs.

    Reference HR is attached afterwards, for evaluation display only.
    """

    data = load_subject(subject)
    inputs = load_results()["inputs"]
    history = get_history(subject, window, inputs)

    predicted_hr, probability, agent_result = run_inference(
        data["ppg"][window], data["acc"][window], history
    )

    return {
        "subject": subject,
        "window": window,
        "predicted_hr": predicted_hr,
        "probability": probability,
        "history": list(history),
        "agent_result": agent_result,
        "reference_hr": float(data["hr"][window]),
    }


EXPERIMENTS = ROOT / "results" / "experiments"


@st.cache_data(show_spinner=False)
def load_experiments():
    """Post-D2 experiment outputs. Each entry is None if that phase has not been run."""

    def read(path):
        path = EXPERIMENTS / path
        return pd.read_csv(path) if path.exists() else None

    return {
        "calibration_test": read("phase1_calibration/test_calibration_metrics.csv"),
        "calibration_bins": read("phase1_calibration/test_reliability_bins.csv"),
        "fp_comparison": read("phase2_fp_reduction/test_comparison.csv"),
        "robustness": read("phase3_robustness/synthetic_summary.csv"),
        "motion": read("phase3_robustness/natural_motion_subgroups.csv"),
    }

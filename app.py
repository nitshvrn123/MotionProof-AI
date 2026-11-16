from pathlib import Path
import html
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".python_deps"))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from src.evaluation.generate_validation_predictions import load_cnn, predict_cnn
from src.evaluation.reliability_features import extract_reliability_features
from src.evaluation.reliability_baseline import FEATURE_COLUMNS
from src.agents.fitted_q_inference import load_fitted_q_policy
from src.agents.investigation_environment import InvestigationEnvironment


st.set_page_config(
    page_title="MotionProof AI | Wearable Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics" / "final_test"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #080e19;
    color: #eaf1fa;
}

[data-testid="stSidebar"] {
    background: #0d1726;
    border-right: 1px solid #26354a;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
}

h1, h2, h3 {
    letter-spacing: -0.035em;
}

.hero {
    padding: 28px 32px;
    border: 1px solid #293b51;
    border-radius: 18px;
    background: linear-gradient(120deg, #14263a, #101827);
    margin-bottom: 26px;
}

.eyebrow {
    color: #51dfc5;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    font-size: 36px;
    font-weight: 800;
    margin: 10px 0;
}

.hero-subtitle {
    color: #9eafc4;
    font-size: 15px;
}

.panel {
    background: #111d2d;
    border: 1px solid #29394f;
    border-radius: 15px;
    padding: 24px;
    margin-bottom: 18px;
}

.panel-label {
    font-size: 12px;
    color: #93a6bd;
    text-transform: uppercase;
    letter-spacing: 1.4px;
}

.panel-value {
    font-size: 37px;
    font-weight: 800;
    color: #f2f7fd;
    margin-top: 8px;
}

.panel-detail {
    font-size: 12px;
    color: #9aacc0;
    margin-top: 7px;
}

.section-heading {
    color: #e9f2fb;
    font-size: 21px;
    font-weight: 750;
    margin: 25px 0 14px;
}

.timeline-card {
    border-left: 3px solid #51dfc5;
    padding: 15px 20px;
    margin: 12px 0;
    background: #121e2d;
    border-radius: 0 12px 12px 0;
}

.timeline-number {
    font-size: 11px;
    color: #51dfc5;
    letter-spacing: 1px;
}

.timeline-title {
    font-size: 17px;
    font-weight: 700;
    margin: 6px 0;
}

.timeline-description {
    font-size: 13px;
    color: #a2b2c5;
}

.decision-accept {
    background: #102b27;
    border: 1px solid #2b8b75;
    border-radius: 14px;
    padding: 25px;
}

.decision-flag {
    background: #321d29;
    border: 1px solid #d46d85;
    border-radius: 14px;
    padding: 25px;
}

.decision-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.decision-value {
    font-size: 28px;
    font-weight: 800;
    margin-top: 8px;
}

div.stButton > button[kind="primary"] {
    background: #43d8b8;
    color: #08121b;
    border: none;
    font-weight: 800;
    border-radius: 10px;
    min-height: 48px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_subject(subject):
    # Use the same data-location convention as the original app.
    candidates = [
        ROOT / "data" / "processed" / "test" / f"S{subject}.npz",
        ROOT / "data" / "processed" / f"S{subject}.npz",
        ROOT / "data" / f"S{subject}.npz",
    ]

    path = next((p for p in candidates if p.exists()), None)

    if path is None:
        raise FileNotFoundError(
            f"Could not locate S{subject}.npz. "
            "Check the original app.py DATA path."
        )

    with np.load(path) as data:
        return {
            "ppg": data["ppg"].copy(),
            "acc": data["acc"].copy(),
            "hr": data["hr"].copy(),
        }


@st.cache_data
def load_results():
    return {
        "comparison": pd.read_csv(
            METRICS / "test_policy_comparison.csv"
        ),
        "inputs": pd.read_csv(
            METRICS / "test_policy_inputs_nonlinear.csv"
        ),
    }


@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cnn, mean, std = load_cnn(device)

    classifier = joblib.load(
        ROOT / "checkpoints/reliability_classifier_nonlinear_full.joblib"
    )

    agent = load_fitted_q_policy(
        ROOT / "checkpoints/learned_policy_fitted_q_full.pt",
        device="cpu",
    )

    return cnn, mean, std, classifier, agent, device


def get_history(subject, window, inputs):
    rows = inputs[
        (inputs["subject"] == subject)
        & (inputs["window_index"] < window)
    ].sort_values("window_index")

    return rows.tail(5)["cnn_predicted_hr"].tolist()


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

    probability = float(
        classifier.predict_proba(feature_frame)[0, 1]
    )

    environment = InvestigationEnvironment(
        ppg_window=ppg,
        acc_window=acc,
        cnn_predicted_hr=predicted_hr,
        unreliable_probability=probability,
        previous_hr_predictions=history,
        budget=2,
    )

    result = agent.run(environment)

    return predicted_hr, probability, result


def metric_card(label, value, detail):
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-label">{html.escape(label)}</div>
            <div class="panel-value">{html.escape(str(value))}</div>
            <div class="panel-detail">{html.escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def signal_chart(values, rate, title):
    values = np.asarray(values)
    times = np.arange(len(values)) / rate

    fig = go.Figure()

    if values.ndim == 1:
        fig.add_trace(go.Scatter(
            x=times,
            y=values,
            mode="lines",
            line=dict(color="#51dfc5", width=2),
            name="PPG",
        ))
    else:
        colors = ["#51dfc5", "#64b5f6", "#ae91fa"]

        for axis in range(values.shape[1]):
            fig.add_trace(go.Scatter(
                x=times,
                y=values[:, axis],
                mode="lines",
                line=dict(color=colors[axis], width=1.6),
                name=f"ACC {'XYZ'[axis]}",
            ))

    fig.update_layout(
        title=title,
        height=310,
        paper_bgcolor="#111d2d",
        plot_bgcolor="#111d2d",
        font=dict(color="#a4b5c8"),
        xaxis=dict(
            title="Time (seconds)",
            gridcolor="#29394f",
        ),
        yaxis=dict(gridcolor="#29394f"),
        margin=dict(l=35, r=20, t=55, b=45),
        legend=dict(orientation="h", y=1.15),
    )

    return fig


ACTION_NAMES = {
    "CHECK_PPG": "PPG Signal Investigation",
    "CHECK_MOTION": "Motion Artifact Investigation",
    "CHECK_CONSISTENCY": "Heart Rate Consistency Check",
    "ACCEPT": "Prediction Accepted",
    "FLAG_UNRELIABLE": "Prediction Flagged",
}

ACTION_DESCRIPTIONS = {
    "CHECK_PPG": "Examined the PPG waveform and pulse-related signal characteristics.",
    "CHECK_MOTION": "Examined accelerometer measurements for movement-related interference.",
    "CHECK_CONSISTENCY": "Compared the current heart rate prediction with recent model predictions.",
    "ACCEPT": "The learned policy selected ACCEPT as its final decision.",
    "FLAG_UNRELIABLE": "The learned policy selected FLAG_UNRELIABLE as its final decision.",
}


def show_investigation(result):
    actions = result.get("actions", [])
    findings = result.get("investigation_results", {})

    for index, action in enumerate(actions, start=1):
        name = ACTION_NAMES.get(action, action.replace("_", " ").title())
        description = ACTION_DESCRIPTIONS.get(
            action,
            "Agent action recorded by the learned policy.",
        )

        st.markdown(
            f"""
            <div class="timeline-card">
                <div class="timeline-number">STEP {index:02d}</div>
                <div class="timeline-title">{html.escape(name)}</div>
                <div class="timeline-description">
                    {html.escape(description)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if action in findings:
            with st.expander(f"View findings: {name}"):
                st.json(findings[action])

    st.caption(
        f"Investigations used: {result.get('investigations_used', 0)} / 2 "
        f"| Remaining budget: {result.get('remaining_budget', 0)}"
    )


st.sidebar.markdown("## MOTIONPROOF AI")
st.sidebar.caption("WEARABLE INTELLIGENCE")

page = st.sidebar.radio(
    "NAVIGATION",
    ["Interactive Prototype", "Research Evaluation"],
)

st.sidebar.divider()

subject = st.sidebar.selectbox(
    "Select Subject",
    [13, 14, 15],
    format_func=lambda value: f"Subject S{value}",
)

subject_data = load_subject(subject)

window = st.sidebar.slider(
    "Sensor Window",
    min_value=0,
    max_value=len(subject_data["ppg"]) - 1,
    value=0,
)

ppg = subject_data["ppg"][window]
acc = subject_data["acc"][window]
reference_hr = float(subject_data["hr"][window])

st.sidebar.caption("Recorded PPG-DaLiA test data")
st.sidebar.caption("8-second sensor windows")
st.sidebar.caption("Research demonstration only")


if page == "Interactive Prototype":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">AI-POWERED WEARABLE MONITORING</div>
            <div class="hero-title">MotionProof AI</div>
            <div class="hero-subtitle">
                From raw wearable signals to an investigated heart rate prediction.
                Explore how a trained AI agent evaluates prediction reliability.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"**Active recording:** Subject S{subject} &nbsp; | &nbsp; "
        f"Window {window} &nbsp; | &nbsp; 8-second sample"
    )

    saved = st.session_state.get("motionproof_prototype_result")

    current = (
        saved is not None
        and saved["subject"] == subject
        and saved["window"] == window
    )

    st.markdown(
        '<div class="section-heading">Wearable Signal Monitor</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            signal_chart(ppg, 64, "Photoplethysmography (PPG)"),
            width="stretch",
        )

    with right:
        st.plotly_chart(
            signal_chart(acc, 32, "Three-Axis Accelerometer"),
            width="stretch",
        )

    st.markdown(
        '<div class="section-heading">AI Analysis</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "Run AI Analysis",
        type="primary",
        width="stretch",
    ):
        with st.spinner("Running CNN, reliability model, and AI agent..."):
            inputs = load_results()["inputs"]
            history = get_history(subject, window, inputs)

            predicted_hr, probability, agent_result = run_inference(
                ppg, acc, history
            )

        st.session_state["motionproof_prototype_result"] = {
            "subject": subject,
            "window": window,
            "predicted_hr": predicted_hr,
            "probability": probability,
            "agent_result": agent_result,
        }

        st.rerun()

    if current:
        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "CNN Heart Rate",
                f"{saved['predicted_hr']:.1f} BPM",
                "Predicted from PPG",
            )

        with c2:
            metric_card(
                "Estimated Failure Probability",
                f"{saved['probability'] * 100:.1f}%",
                "Nonlinear reliability classifier",
            )

        with c3:
            metric_card(
                "Reference Heart Rate",
                f"{reference_hr:.1f} BPM",
                "Evaluation only; not used by the agent",
            )

        st.markdown(
            '<div class="section-heading">AI Agent Investigation</div>',
            unsafe_allow_html=True,
        )

        agent_result = saved["agent_result"]
        show_investigation(agent_result)

        decision = agent_result.get("final_decision", "")
        accepted = decision == "ACCEPT"

        style = "decision-accept" if accepted else "decision-flag"
        label = "ACCEPTED" if accepted else "FLAGGED AS UNRELIABLE"

        st.markdown(
            '<div class="section-heading">Final AI Decision</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="{style}">
                <div class="decision-label">LEARNED AGENT DECISION</div>
                <div class="decision-value">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "This is an experimental prediction-reliability decision, "
            "not a medical diagnosis."
        )

    else:
        st.info(
            "Select a subject and sensor window, then click "
            "'Run AI Analysis' to begin."
        )

else:
    st.title("Research Evaluation")
    st.caption(
        "Frozen held-out test results from the trained MotionProof AI system."
    )

    comparison = load_results()["comparison"].copy()

    st.subheader("Policy Performance Comparison")

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
    ]

    summary = comparison[["policy", "windows"] + metrics].copy()

    summary = summary.rename(
        columns={
            "policy": "Policy",
            "windows": "Test Windows",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1 Score",
            "specificity": "Specificity",
        }
    )

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Specificity",
    ]:
        summary[column] = summary[column].map(
            lambda value: f"{value * 100:.2f}%"
        )

    st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Accuracy, Recall, and F1 Comparison")

    chart_data = comparison.set_index("policy")[
        ["accuracy", "recall", "f1"]
    ]

    st.bar_chart(
    chart_data,
    x_label="Policy",
    y_label="Score",
    stack=False,
    width="stretch",
)

    st.info(
        "The Learned Fitted Q policy achieves the highest recall "
        "(88.14%) and F1 score (0.4061) among the evaluated policies. "
        "However, it has lower accuracy and specificity, reflecting "
        "a tradeoff between detecting unreliable predictions and "
        "generating false alarms."
    )

    with st.expander("View Detailed Evaluation Results"):
        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "These are saved evaluation results, not metrics calculated "
        "from the current interactive prediction. "
        "MotionProof AI is a research prototype, not a medical device."
    )

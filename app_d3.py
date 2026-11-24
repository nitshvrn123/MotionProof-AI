"""MotionProof AI: D3 research studio.

Run with:
    streamlit run app_d3.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="MotionProof",
    page_icon="◌",
    layout="wide",
    initial_sidebar_state="auto",
)

from src.agents.investigation_explanations import explain_investigation
from ui import charts
from ui import components as ui
from ui.analysis import (
    BEHAVIORS,
    INVESTIGATIONS,
    METRICS,
    POLICY_ORDER,
    action_story,
    analysis_markdown,
    behavior_mask,
    format_metric,
    journey_headline,
    next_matching_window,
    path_label,
    previous_matching_window,
    safe_number,
    studio_headline,
)
from ui.pipeline import (
    ACC_RATE_HZ,
    PPG_RATE_HZ,
    SUBJECTS,
    analyze_window,
    load_results,
    load_subject,
    load_experiments,
    load_traces,
)
from ui.theme import PALETTE, inject_css

PAGES = ("Signal Studio", "Agent Journey", "Research Lab")
SUBJECT_LABELS = {f"S{s}": s for s in SUBJECTS}

inject_css()


# ---------------------------------------------------------------- state

def _init_state():
    st.session_state.setdefault("page", PAGES[0])
    st.session_state.setdefault("subject_label", "S13")
    st.session_state.setdefault("window", 0)
    st.session_state.setdefault("behavior", "Any behaviour")
    st.session_state.setdefault("analyses", {})
    st.session_state.setdefault("_last_subject", "S13")
    st.session_state.setdefault("_last_page", PAGES[0])

    # Optional deep link, applied once per session: ?s=14&w=120&page=journey
    if not st.session_state.get("_query_applied"):
        st.session_state["_query_applied"] = True
        params = st.query_params
        label = f"S{params.get('s', '')}"
        if label in SUBJECT_LABELS:
            st.session_state["subject_label"] = label
            st.session_state["_last_subject"] = label
        if params.get("w", "").isdigit():
            st.session_state["window"] = int(params["w"])
        pages = {p.split()[1].lower(): p for p in PAGES}
        if params.get("page", "").lower() in pages:
            st.session_state["page"] = pages[params["page"].lower()]
            st.session_state["_last_page"] = st.session_state["page"]


def _on_subject_change():
    if st.session_state.get("subject_label") is None:
        st.session_state["subject_label"] = st.session_state["_last_subject"]
        return
    st.session_state["_last_subject"] = st.session_state["subject_label"]
    st.session_state["window"] = 0


def _on_page_change():
    if st.session_state.get("page") is None:
        st.session_state["page"] = st.session_state["_last_page"]
    st.session_state["_last_page"] = st.session_state["page"]


def _shift_window(delta, maximum):
    st.session_state["window"] = min(max(st.session_state["window"] + delta, 0), maximum)


def _jump_to(window):
    if window is not None:
        st.session_state["window"] = int(window)


def _go_to(page):
    st.session_state["page"] = page
    st.session_state["_last_page"] = page


def _key(subject, window):
    return f"{subject}:{window}"


def _run_analysis(subject, window):
    with st.status("Reading the recording", expanded=False) as status:
        status.write("Heart-rate CNN, then the reliability classifier, then the learned agent.")
        record = analyze_window(subject, window)
        status.update(label="Analysis finished", state="complete")
    st.session_state["analyses"][_key(subject, window)] = record


_init_state()


# ---------------------------------------------------------------- data

results = load_results()
traces = load_traces()
subject = SUBJECT_LABELS[st.session_state["subject_label"]]
try:
    subject_data = load_subject(subject)
except FileNotFoundError:
    subject_data = None
data_available = subject_data is not None
n_windows = len(subject_data["ppg"]) if data_available else 1
st.session_state["window"] = min(st.session_state["window"], n_windows - 1)


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    ui.render(ui.side_intro())
    ui.render(ui.side_label("Participant"))
    st.segmented_control("Participant", list(SUBJECT_LABELS), key="subject_label",
                         on_change=_on_subject_change, label_visibility="collapsed")

    if not data_available:
        ui.render(ui.missing_data_side_note(subject))
    else:
        ui.render(ui.side_label("Window"))
        ui.render(ui.window_counter(st.session_state["window"], n_windows))
        st.slider("Window", 0, n_windows - 1, key="window", label_visibility="collapsed")
        left, right = st.columns(2)
        left.button("Previous", width="stretch", on_click=_shift_window,
                    args=(-1, n_windows - 1), disabled=st.session_state["window"] == 0)
        right.button("Next", width="stretch", on_click=_shift_window,
                     args=(1, n_windows - 1), disabled=st.session_state["window"] >= n_windows - 1)

        ui.render(ui.side_rule())
        ui.render(ui.side_label("Find windows where the agent…"))

        if traces is None:
            ui.render('<div class="mp-side-note">Behaviour search needs the trace file. '
                      'Run <code>python -m src.evaluation.build_agent_traces</code>.</div>')
        else:
            st.selectbox("Behaviour", list(BEHAVIORS), key="behavior", label_visibility="collapsed")
            local = traces[traces["subject"] == subject].sort_values("window_index")
            matches = behavior_mask(local["actions"], st.session_state["behavior"])
            windows = local["window_index"].tolist()
            count = sum(matches)
            current = st.session_state["window"]
            b1, b2 = st.columns(2)
            b1.button("Earlier", width="stretch", on_click=_jump_to,
                      args=(previous_matching_window(windows, matches, current),), disabled=count == 0)
            b2.button("Later", width="stretch", on_click=_jump_to,
                      args=(next_matching_window(windows, matches, current),), disabled=count == 0)
            ui.render(
                f'<div class="mp-side-note"><b>{count:,}</b> of {len(windows):,} windows in S{subject} '
                "match. Behaviours come from a replay of the frozen policy that reproduces every "
                "saved evaluation decision.</div>"
            )


# ---------------------------------------------------------------- masthead

window = st.session_state["window"]
ppg = subject_data["ppg"][window] if data_available else None
acc = subject_data["acc"][window] if data_available else None
record = st.session_state["analyses"].get(_key(subject, window))

mast_left, mast_right = st.columns([1, 1.15], vertical_alignment="center")
with mast_left:
    ui.render(ui.wordmark())
with mast_right:
    with st.container(key="nav"):
        st.segmented_control("Workspace", PAGES, key="page", on_change=_on_page_change,
                             label_visibility="collapsed")
ui.render('<div class="mp-mast-rule"></div>')
page = st.session_state["page"]


def explanations_for(result):
    findings = result.get("investigation_results", {})
    return [(a, explain_investigation(a, findings[a]))
            for a in result.get("actions", []) if a in findings]


def analyze_button(label="Analyze this recording"):
    if st.button(label, type="primary", width="stretch"):
        _run_analysis(subject, window)
        st.rerun()


# ---------------------------------------------------------------- Signal Studio

def signal_studio():
    if not data_available:
        ui.render(ui.missing_data_notice(subject))
        return

    hero_left, hero_right = st.columns([1.5, 1], gap="large", vertical_alignment="bottom")

    with hero_left:
        ui.render(ui.kicker(f"Signal Studio, participant S{subject}"))
        ui.render(ui.headline(studio_headline(record)
                              or "Eight seconds from a wrist. Can its heart rate be trusted?"))
        ui.render(ui.lede(
            "A smartwatch-style pulse sensor and accelerometer, recorded while the person "
            "went about everyday activities. Analyzing runs the trained heart-rate CNN, "
            "a reliability classifier and the learned investigation agent."
        ))

    with hero_right:
        with st.container(key="ticket"):
            ui.render(ui.ticket(subject, window, record))
            if record is None:
                analyze_button()
            else:
                st.button("Follow the agent's journey", type="primary", width="stretch",
                          on_click=_go_to, args=("Agent Journey",))

    st.write("")
    with st.container(key="paper"):
        top_l, top_r = st.columns([3, 1.2], vertical_alignment="center")
        top_l.markdown('<div class="mp-small">Drag across either panel to zoom both; '
                       "double-click to reset.</div>", unsafe_allow_html=True)
        show_magnitude = top_r.toggle("Show motion magnitude", value=False)
        st.plotly_chart(charts.signal_figure(ppg, acc, PPG_RATE_HZ, ACC_RATE_HZ, show_magnitude),
                        width="stretch", config=charts.QUIET)

    if record is None:
        st.write("")
        ui.render(ui.empty_state(
            "This window hasn't been read yet",
            "Use Analyze this recording above. The estimate, the reliability score and "
            "the agent's decision will appear here.",
        ))
        return

    ui.render(ui.h2("The reading",
                    "What the models produced from the sensors, set beside the ECG "
                    "reference that only we get to see."))
    est_col, ref_col = st.columns([1.45, 1], gap="large")
    with est_col:
        with st.container(key="panel-estimate"):
            ui.render(ui.estimate_panel(record))
    with ref_col:
        with st.container(key="panel-ledger"):
            ui.render(ui.reference_ledger(record))

    ui.render(ui.h2("The agent's call"))
    ui.render(ui.verdict_band(record["agent_result"]))
    st.write("")
    a1, a2, _ = st.columns([1, 1, 2])
    a1.button("See how it decided", width="stretch", on_click=_go_to, args=("Agent Journey",))
    a2.download_button(
        "Download summary",
        analysis_markdown(record, explanations_for(record["agent_result"])),
        file_name=f"motionproof_S{subject}_w{window}.md", mime="text/markdown", width="stretch")

    ui.render(ui.h2("Around this window",
                    "Saved CNN estimates for the neighbouring windows, with the ECG reference "
                    "for comparison. Hollow circles are windows the agent flagged."))
    with st.container(key="panel-context"):
        st.plotly_chart(charts.hr_context_figure(results["inputs"], traces, subject, window),
                        width="stretch", config=charts.QUIET)


# ---------------------------------------------------------------- Agent Journey

def evidence(action, step, total, findings, current):
    with st.container(key=f"evidence-{step}"):
        ui.render(ui.evidence_head(action, step, total))
        ui.render(f'<div class="mp-ev-text">{explain_investigation(action, findings)}</div>')

        if action == "CHECK_PPG":
            ui.render(ui.facts([
                ("Disagreement", safe_number(findings.get("cnn_spectral_disagreement_bpm"), 1, " bpm")),
                ("Autocorrelation", safe_number(findings.get("autocorrelation_strength"), 2)),
                ("Spectral concentration", safe_number(findings.get("spectral_concentration"), 2)),
            ]))
            st.plotly_chart(charts.ppg_evidence_figure(findings), width="stretch", config=charts.NO_BAR)
        elif action == "CHECK_MOTION":
            ui.render(ui.facts([
                ("Magnitude variation", safe_number(findings.get("motion_intensity"), 3)),
                ("Magnitude range", safe_number(findings.get("magnitude_range"), 3, " g")),
                ("Dominant frequency", safe_number(findings.get("dominant_motion_frequency_hz"), 2, " Hz")),
            ]))
        elif action == "CHECK_CONSISTENCY" and current["history"]:
            ui.render(ui.facts([
                ("Previous estimate", safe_number(findings.get("previous_hr_bpm"), 1, " bpm")),
                ("Recent median", safe_number(findings.get("historical_median_hr_bpm"), 1, " bpm")),
                ("Change", safe_number(findings.get("hr_jump_bpm"), 1, " bpm")),
            ]))
            st.plotly_chart(charts.consistency_evidence_figure(current["history"], current["predicted_hr"]),
                            width="stretch", config=charts.NO_BAR)

        with st.expander("All measurements"):
            st.json(findings)


def agent_journey():
    if not data_available:
        ui.render(ui.missing_data_notice(subject))
        return

    ui.render(ui.kicker(f"Agent Journey, participant S{subject}, window {window:,}"))

    if record is None:
        ui.render(ui.headline("Every decision leaves a path."))
        ui.render(ui.lede("Analyze this window to follow the learned agent from what it saw, "
                          "through the checks it chose, to its final call."))
        st.write("")
        c1, _ = st.columns([1, 1.6])
        with c1:
            analyze_button()
        return

    result = record["agent_result"]
    actions = result.get("actions", [])
    steps = [a for a in actions if a in INVESTIGATIONS]

    ui.render(ui.headline(journey_headline(actions)))
    ui.render(ui.lede(action_story(actions) + " Every step below is read from the "
                      "environment's recorded action history."))
    ui.render(ui.story_map(actions))

    ui.render(ui.chapter(1, "What it saw",
                         "The agent starts with two numbers and a budget of two checks. "
                         "It never sees the ECG reference."))
    ui.render(f'<div style="padding-left:84px">{ui.flatten(ui.saw_block(record))}</div>')
    ui.render(ui.thread(70, draw=True))

    ui.render(ui.chapter(2, "What it checked",
                         "Each check returns measurements, summarised below in plain words. "
                         "These summaries describe what was measured; they are not the "
                         "policy's reasons." if steps else "",
                         hollow=not steps))
    if not steps:
        ui.render(f'<div style="padding-left:84px">{ui.flatten(ui.blank_investigation(record))}</div>')
    else:
        _, body = st.columns([0.075, 1])
        with body:
            findings = result.get("investigation_results", {})
            cols = st.columns(len(steps), gap="medium")
            for i, (col, action) in enumerate(zip(cols, steps), start=1):
                with col:
                    evidence(action, i, len(steps), findings.get(action, {}), record)
    ui.render(ui.thread(70))

    ui.render(ui.chapter(3, "What it decided"))
    frequency = ""
    if traces is not None:
        same = int((traces["actions"] == "|".join(actions)).sum())
        frequency = (f" Across all {len(traces):,} held-out windows, the agent took this exact "
                     f"path {same:,} times ({same / len(traces):.1%}).")
    ui.render(f'<div style="padding-left:84px">{ui.flatten(ui.decision_block(record, frequency))}</div>')

    ui.render(f'<div style="padding-left:84px;margin-top:26px">{ui.flatten(ui.caption(
        "<b>About confidence.</b> The agent picks the action with the highest learned value. "
        "Those values are not calibrated probabilities, so they are not shown as confidence. "
        f"Path: {path_label(actions)}."))}</div>')


# ---------------------------------------------------------------- Research Lab

def _signed(value):
    return f"{value:.3f}".replace("-", "\u2212")


def research_lab():
    comparison = results["comparison"]
    by = comparison.set_index("policy")
    learned, fixed, none = (by.loc[p] for p in POLICY_ORDER)
    best_reward = by["average_reward"].idxmax()
    total = int(comparison["windows"].iloc[0])

    ui.render(ui.kicker("Research Lab, frozen held-out results"))
    ui.render(ui.headline(
        f"The learned agent catches {learned['recall']:.0%} of unreliable readings, "
        "and pays for it in false alarms.", wide=True))
    ui.render(ui.lede(
        f"Three policies were tested on {total:,} eight-second windows from participants "
        "S13, S14 and S15, none of whom were used for training or tuning. Every number "
        "here is read from the saved evaluation files."))

    st.write("")
    ui.render(ui.finding(
        f"{learned['recall']:.1%}", PALETTE["terracotta"],
        "It finds most of the unreliable readings",
        f"An estimate counts as unreliable when it is more than 10 bpm off. The learned agent "
        f"flags <b>{learned['recall']:.1%}</b> of those, against {fixed['recall']:.1%} for the "
        f"fixed rules and {none['recall']:.1%} with no investigation. It also has the best F1 "
        f"score, <b>{learned['f1']:.3f}</b>."))
    ui.render(ui.finding(
        f"{learned['false_positive_rate']:.1%}", PALETTE["terracotta_deep"],
        "It also flags many readings that were fine",
        f"That is its false-positive rate, the highest of the three (fixed rules "
        f"{fixed['false_positive_rate']:.1%}, no investigation {none['false_positive_rate']:.1%}). "
        f"Its accuracy, <b>{learned['accuracy']:.1%}</b>, is the lowest.", flip=True))
    ui.render(ui.finding(
        _signed(by.loc[best_reward, 'average_reward']), "#8A7062",
        f"{best_reward} earns the best reward",
        f"The reward function gives +2 for catching an unreliable reading, −1.5 for a "
        f"false alarm, −2 for a miss and −0.1 for each check. Under it, {best_reward.lower()} averages {_signed(by.loc[best_reward, 'average_reward'])} "
        f"per window, against {_signed(learned['average_reward'])} for the learned agent. "
        "Better detection did not translate into higher reward."))

    ui.render(ui.h2("The trade-off in one picture",
                    "Up means catching more unreliable readings. Right means wrongly flagging "
                    "more reliable ones. The ideal policy would sit in the top-left corner."))
    chart_col, text_col = st.columns([1.65, 1], gap="large")
    with chart_col:
        with st.container(key="paper-lab"):
            st.plotly_chart(charts.tradeoff_figure(comparison), width="stretch", config=charts.QUIET)
    with text_col:
        st.write("")
        ui.render(ui.callout("No policy wins outright. Moving up the chart has meant "
                             "moving right."))
        ui.render(ui.body(
            f"Compared with the fixed rules, the learned agent catches "
            f"<b>{(learned['recall'] - fixed['recall']) * 100:.1f} points</b> more unreliable "
            f"readings and wrongly flags <b>{(learned['false_positive_rate'] - fixed['false_positive_rate']) * 100:.1f} "
            f"points</b> more reliable ones, while using slightly fewer checks per window "
            f"({learned['average_investigations']:.2f} vs {fixed['average_investigations']:.2f})."))

    ui.render(ui.h2("Compare any measure"))
    labels = {k: v[0] for k, v in METRICS.items()}
    choice = st.segmented_control("Measure", list(labels.values()), default="Recall",
                                  key="lab_metric", label_visibility="collapsed")
    metric = next(k for k, v in labels.items() if v == (choice or "Recall"))
    higher = METRICS[metric][1]
    with st.container(key="panel-metric"):
        st.plotly_chart(charts.policy_metric_figure(comparison, metric), width="stretch",
                        config=charts.NO_BAR)
        hint = {True: "Higher is better.", False: "Lower is better.",
                None: "Average number of checks used per window."}[higher]
        ui.render(ui.caption(hint))

    per_subject = results["per_subject"]
    if per_subject is not None:
        ui.render(ui.h2("Person by person", "The same measure, split by held-out participant."))
        with st.container(key="panel-subjects"):
            st.plotly_chart(charts.per_subject_figure(per_subject, metric), width="stretch",
                            config=charts.NO_BAR)
        tops = per_subject.loc[per_subject.groupby("subject")["recall"].idxmax(), "policy"]
        if (tops == "Learned Fitted Q").all():
            ui.render(ui.caption("The learned agent has the highest recall for each of the "
                                 "three participants, not just on average."))

    ui.render(ui.h2("What the checks are worth",
                    "Two post-training experiments on the frozen agent. Nothing was retrained."))
    left, right = st.columns(2, gap="large")
    if results["budget_ablation"] is not None:
        with left:
            with st.container(key="panel-budget"):
                ui.render('<div class="mp-ev-title">Fewer checks allowed</div>')
                st.plotly_chart(charts.budget_ablation_figure(results["budget_ablation"]),
                                width="stretch", config=charts.NO_BAR)
            ui.render(ui.caption(
                "Investigations are blocked after N steps. <b>A budget of 0 is not the "
                "no-investigation baseline</b>: that baseline thresholds the failure "
                "probability at 0.5, while budget 0 forces the agent's own final choice."))
    if results["action_ablation"] is not None:
        with right:
            with st.container(key="panel-tools"):
                ui.render('<div class="mp-ev-title">One check taken away</div>')
                st.plotly_chart(charts.action_ablation_figure(results["action_ablation"]),
                                width="stretch", config=charts.NO_BAR)
            act = results["action_ablation"].set_index("configuration")
            ui.render(ui.caption(
                f"Removing the PPG check drops F1 from {act.loc['Full Agent', 'f1']:.3f} to "
                f"{act.loc['Without PPG', 'f1']:.3f}. Removing the consistency check changes "
                f"almost nothing ({act.loc['Without Consistency', 'f1']:.3f})."))

    if traces is not None:
        direct = traces["investigations_used"].eq(0).mean()
        two = traces["investigations_used"].eq(2).mean()
        ppg_share = traces["actions"].str.contains("CHECK_PPG").mean()
        cons_share = traces["actions"].str.contains("CHECK_CONSISTENCY").mean()
        ui.render(ui.h2("How the agent spends its checks"))
        ui.render(ui.body(
            f"It decides without any check in <b>{direct:.1%}</b> of windows and uses both "
            f"checks in <b>{two:.1%}</b>. The PPG check is its favourite, used in "
            f"{ppg_share:.1%} of windows; the consistency check appears in only {cons_share:.1%}."))
        st.write("")
        with st.container(key="panel-paths"):
            st.plotly_chart(charts.path_distribution_figure(traces), width="stretch",
                            config=charts.NO_BAR)
            ui.render(ui.caption("Green paths end in ACCEPT, terracotta paths in FLAG_UNRELIABLE."))

    post_d2_studies()

    st.write("")
    with st.expander("Appendix: full evaluation table"):
        table = comparison.assign(_o=comparison["policy"].map(
            {p: i for i, p in enumerate(POLICY_ORDER)})).sort_values("_o")
        display = pd.DataFrame({"Policy": table["policy"]})
        for key, (label, _, kind) in METRICS.items():
            display[label] = [format_metric(v, kind) for v in table[key]]
        display["TP / FP / TN / FN"] = [
            f"{r.true_positives:,} / {r.false_positives:,} / {r.true_negatives:,} / {r.false_negatives:,}"
            for r in table.itertuples()]
        st.dataframe(display, hide_index=True, width="stretch")
        st.caption("Source: results/metrics/final_test/test_policy_comparison.csv")


def post_d2_studies():
    exp = load_experiments()
    if all(v is None for v in exp.values()):
        return
    ui.render(ui.experiments_banner())

    cal = exp["calibration_test"]
    if cal is not None and exp["calibration_bins"] is not None:
        c = cal.set_index("method")
        ui.render(ui.study("1. Are the failure probabilities honest?",
                           "Calibration of the reliability classifier, fitted on S11–S12"))
        left, right = st.columns([1.5, 1], gap="large")
        with left:
            with st.container(key="panel-calibration"):
                st.plotly_chart(charts.calibration_figure(exp["calibration_bins"]), width="stretch",
                                config=charts.NO_BAR)
        with right:
            ui.render(ui.body(
                f"The original classifier predicts a <b>{c.loc['original', 'mean_predicted']:.1%}</b> "
                f"failure rate on test, where <b>{c.loc['original', 'positive_rate']:.1%}</b> occurs. "
                f"Platt scaling, chosen on validation, lowers the Brier score from "
                f"{c.loc['original', 'brier']:.3f} to <b>{c.loc['platt', 'brier']:.3f}</b> and ECE from "
                f"{c.loc['original', 'ece']:.3f} to <b>{c.loc['platt', 'ece']:.3f}</b>, without changing "
                "how windows are ranked."))
            ui.render(ui.caption(
                "Some overconfidence remains: failures are much more common in the validation "
                "participants than in the test participants, a shift calibration cannot foresee. "
                "Bins with fewer than 20 windows are hidden."))

    fp = exp["fp_comparison"]
    if fp is not None:
        frozen = fp[fp["family"] == "frozen_d2_baseline"].set_index("variant").loc["Learned Fitted Q"]
        veto = fp[fp["family"] == "agent_calibrated_veto"].iloc[0]
        ui.render(ui.study("2. Can false alarms be reduced?",
                           "Thresholds, a calibrated veto, cost-based rules and 18 retrained policies"))
        left, right = st.columns([1.5, 1], gap="large")
        with left:
            with st.container(key="panel-fp"):
                st.plotly_chart(charts.fp_variants_figure(fp), width="stretch", config=charts.NO_BAR)
        with right:
            ui.render(ui.callout("No variant beat the original agent on both recall and false alarms."))
            ui.render(ui.body(
                f"Each option slides along the same trade-off. The calibrated veto, the best "
                f"deterministic option, lowers the false-positive rate from {frozen['false_positive_rate']:.1%} "
                f"to <b>{veto['false_positive_rate']:.1%}</b> but recall falls from {frozen['recall']:.1%} to "
                f"<b>{veto['recall']:.1%}</b>, leaving F1 essentially unchanged "
                f"({frozen['f1']:.3f} vs {veto['f1']:.3f})."))
            ui.render(ui.caption(
                "Raising the false-alarm penalty in the reward cut false positives but cost even more "
                "recall. Retraining with the original reward varies by several points between seeds "
                "(error bars), as much as the improvement being sought."))

    rob = exp["robustness"]
    if rob is not None:
        ui.render(ui.study("3. What happens when the signal gets worse?",
                           "Synthetic perturbations of the real test recordings, full pipeline re-run"))
        metric_label = st.segmented_control("Robustness measure", ["Recall", "False-positive rate"],
                                            default="Recall", key="rob_metric", label_visibility="collapsed")
        metric = "false_positive_rate" if metric_label == "False-positive rate" else "recall"
        left, right = st.columns([1.6, 1], gap="large")
        with left:
            with st.container(key="panel-robust"):
                st.plotly_chart(charts.robustness_figure(rob, metric), width="stretch", config=charts.NO_BAR)
        with right:
            pivot = rob.pivot_table(index=["perturbation", "level"], columns="policy", values="recall")
            wins = int((pivot["Learned Fitted Q"] >= pivot[["Fixed Rule", "No Investigation"]].max(axis=1)).sum())
            ui.render(ui.body(
                f"The agent keeps the highest recall in <b>{wins} of {len(pivot)}</b> conditions. Under heavy "
                "noise it flags almost everything, which is safe but no longer selective. Quieter problems, "
                "such as a weakened pulse or slow drift, are where unreliable readings start to slip through."))
            ui.render(ui.caption("These are simulated perturbations of real recordings, not new data."))

    motion = exp["motion"]
    if motion is not None:
        learned = motion[motion["policy"] == "Learned Fitted Q"]
        ui.render(ui.study("Real movement, not simulated",
                           "Clean test windows grouped by measured accelerometer motion (not activity labels)"))
        left, right = st.columns([1.5, 1], gap="large")
        with left:
            with st.container(key="panel-motion"):
                st.plotly_chart(charts.motion_quartile_figure(motion), width="stretch", config=charts.NO_BAR)
        with right:
            ui.render(ui.body(
                f"False alarms are mostly a motion problem. In the calmest quarter of windows the agent "
                f"wrongly flags <b>{learned['false_positive_rate'].iloc[0]:.1%}</b> of reliable readings; in "
                f"the most active quarter, <b>{learned['false_positive_rate'].iloc[-1]:.1%}</b>. Failures do "
                f"become more common with motion ({learned['unreliable_rate'].iloc[0]:.1%} to "
                f"{learned['unreliable_rate'].iloc[-1]:.1%}), but flags rise much faster."))


# ---------------------------------------------------------------- router

{"Signal Studio": signal_studio, "Agent Journey": agent_journey}.get(page, research_lab)()
ui.render(ui.footer())

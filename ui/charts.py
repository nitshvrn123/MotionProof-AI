"""Plotly figures for the D3 interface, drawn in the warm earth palette."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.analysis import METRICS, POLICY_ORDER, path_label, split_actions
from ui.theme import DECISION_COLORS, PALETTE, POLICY_COLORS, TOOL_COLORS, base_layout, style_axes

ACC_COLORS = ("#B96547", "#C99A65", "#8A7062")
NO_BAR = {"displayModeBar": False}
QUIET = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def signal_figure(ppg, acc, ppg_rate, acc_rate, show_magnitude=False):
    """Pulse and motion on one shared time axis, so zoom stays in sync."""

    ppg = np.asarray(ppg, dtype=float)
    acc = np.asarray(acc, dtype=float)
    t_ppg = np.arange(len(ppg)) / ppg_rate
    t_acc = np.arange(len(acc)) / acc_rate

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        row_heights=[0.58, 0.42],
                        subplot_titles=("Pulse", "Motion"))

    fig.add_trace(go.Scatter(
        x=t_ppg, y=ppg, mode="lines", name="PPG",
        line=dict(color=PALETTE["terracotta"], width=2.2, shape="spline", smoothing=0.6),
        fill="tozeroy", fillcolor="rgba(229,184,164,.28)",
        hovertemplate="%{x:.2f} s<br>PPG %{y:.1f} a.u.<extra></extra>",
    ), row=1, col=1)

    for axis, color in enumerate(ACC_COLORS):
        fig.add_trace(go.Scatter(
            x=t_acc, y=acc[:, axis], mode="lines", name=f"{'XYZ'[axis]} axis",
            line=dict(color=color, width=1.6),
            hovertemplate="%{x:.2f} s<br>" + "XYZ"[axis] + " %{y:.3f} g<extra></extra>",
        ), row=2, col=1)

    if show_magnitude:
        fig.add_trace(go.Scatter(
            x=t_acc, y=np.linalg.norm(acc, axis=1), mode="lines", name="Magnitude",
            line=dict(color=PALETTE["chocolate"], width=1.4, dash="dot"),
            hovertemplate="%{x:.2f} s<br>|ACC| %{y:.3f} g<extra></extra>",
        ), row=2, col=1)

    fig.update_layout(**base_layout(
        height=560, hovermode="x unified", margin=dict(l=58, r=10, t=46, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1,
                    font=dict(color=PALETTE["brown"])),
    ))
    fig.update_annotations(font=dict(family="Fraunces, Georgia, serif", size=19,
                                     color=PALETTE["chocolate"]), xanchor="left", x=0)
    fig.update_yaxes(title_text="a.u.", row=1, col=1)
    fig.update_yaxes(title_text="g", row=2, col=1)
    fig.update_xaxes(title_text="Seconds into the window", row=2, col=1)
    return style_axes(fig)


def hr_context_figure(inputs, traces, subject, window, span=40):
    """Saved CNN predictions vs ECG reference around the selected window."""

    rows = inputs[inputs["subject"] == subject].sort_values("window_index")
    lo, hi = max(0, window - span), window + span
    rows = rows[(rows["window_index"] >= lo) & (rows["window_index"] <= hi)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rows["window_index"], y=rows["reference_hr"], mode="lines",
        name="ECG reference (evaluation only)",
        line=dict(color=PALETTE["caramel"], width=2, dash="dash"),
        hovertemplate="Window %{x}<br>Reference %{y:.1f} bpm<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=rows["window_index"], y=rows["cnn_predicted_hr"], mode="lines",
        name="CNN estimate", line=dict(color=PALETTE["chocolate"], width=2.2),
        hovertemplate="Window %{x}<br>CNN %{y:.1f} bpm<extra></extra>",
    ))

    if traces is not None:
        local = traces[traces["subject"] == subject].merge(
            rows[["window_index", "cnn_predicted_hr"]], on="window_index")
        flagged = local[local["final_decision"] == "FLAG_UNRELIABLE"]
        fig.add_trace(go.Scatter(
            x=flagged["window_index"], y=flagged["cnn_predicted_hr"], mode="markers",
            name="Flagged by the agent",
            marker=dict(color=PALETTE["cream"], size=8,
                        line=dict(color=PALETTE["terracotta"], width=2)),
            hovertemplate="Window %{x}<br>Flagged<extra></extra>",
        ))

    fig.add_vrect(x0=window - 0.5, x1=window + 0.5, fillcolor=PALETTE["peach"],
                  opacity=0.45, line_width=0)
    fig.update_layout(**base_layout(height=310, hovermode="x unified",
                                    margin=dict(l=54, r=10, t=30, b=44)))
    fig.update_xaxes(title_text="Window")
    fig.update_yaxes(title_text="bpm")
    return style_axes(fig)


def ppg_evidence_figure(findings):
    labels, values, colors = [], [], []

    for key, label, color in (
        ("cnn_predicted_hr_bpm", "CNN", PALETTE["chocolate"]),
        ("spectral_hr_bpm", "Spectral", TOOL_COLORS["CHECK_PPG"]),
    ):
        value = findings.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            labels.append(label)
            values.append(float(value))
            colors.append(color)

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors, width=0.5,
        text=[f"{v:.1f} bpm" for v in values], textposition="outside",
        textfont=dict(color=PALETTE["chocolate"]),
        hovertemplate="%{y}: %{x:.1f} bpm<extra></extra>",
    ))
    fig.update_layout(**base_layout(height=150, margin=dict(l=70, r=20, t=8, b=30),
                                    showlegend=False))
    fig.update_xaxes(range=[0, max(values + [1]) * 1.4], title_text="")
    fig.update_yaxes(showline=False)
    return style_axes(fig)


def consistency_evidence_figure(history, current_hr):
    history = [float(h) for h in history]
    x = list(range(-len(history), 0))
    y = history + [float(current_hr)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=history, mode="lines+markers", name="Earlier estimates",
        line=dict(color=TOOL_COLORS["CHECK_CONSISTENCY"], width=2),
        marker=dict(size=7), hovertemplate="%{x:+d}: %{y:.1f} bpm<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[y[-1]], mode="markers", name="This window",
        marker=dict(size=13, color=PALETTE["chocolate"]),
        hovertemplate="This window: %{y:.1f} bpm<extra></extra>",
    ))
    pad = max(3.0, (max(y) - min(y)) * 0.18)
    fig.update_layout(**base_layout(
        height=210, margin=dict(l=46, r=10, t=10, b=74),
        legend=dict(orientation="h", yanchor="top", y=-0.42, xanchor="left", x=0)))
    fig.update_xaxes(title_text="Windows before this one", dtick=1)
    fig.update_yaxes(title_text="bpm", range=[min(y) - pad, max(y) + pad])
    return style_axes(fig)


def _ordered(frame):
    order = {name: i for i, name in enumerate(POLICY_ORDER)}
    return frame.assign(_o=frame["policy"].map(order).fillna(99)).sort_values("_o")


def _fmt(value, kind):
    return f"{value * 100:.1f}%" if kind == "pct" else f"{value:.3f}"


def policy_metric_figure(comparison, metric):
    """Horizontal lollipop: one stem per policy."""

    label, _, kind = METRICS[metric]
    frame = _ordered(comparison).iloc[::-1]
    values = frame[metric].astype(float).tolist()
    colors = [POLICY_COLORS.get(p, PALETTE["caramel"]) for p in frame["policy"]]
    base = 0.0 if metric != "average_reward" else min(values) * 1.25

    fig = go.Figure()
    for policy, value, color in zip(frame["policy"], values, colors):
        fig.add_trace(go.Scatter(
            x=[base, value], y=[policy, policy], mode="lines",
            line=dict(color=color, width=3), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=values, y=frame["policy"], mode="markers+text",
        marker=dict(size=22, color=colors, line=dict(color=PALETTE["cream"], width=3)),
        text=[_fmt(v, kind) for v in values], textposition="middle right",
        textfont=dict(family="Fraunces, Georgia, serif", size=17, color=PALETTE["chocolate"]),
        hovertemplate="%{y}<br>" + label + " %{text}<extra></extra>", showlegend=False))

    span = max(values) - base
    fig.update_layout(**base_layout(height=280, margin=dict(l=130, r=30, t=10, b=40)))
    fig.update_xaxes(range=[base, max(values) + span * 0.28 + 1e-9],
                     tickformat=".0%" if kind == "pct" else None, title_text=label)
    fig.update_yaxes(showline=False, showgrid=False, tickfont=dict(size=14, color=PALETTE["chocolate"]))
    return style_axes(fig)


def tradeoff_figure(comparison):
    frame = _ordered(comparison)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=PALETTE["beige"], dash="dot"),
                             hoverinfo="skip", showlegend=False))
    for _, row in frame.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["false_positive_rate"]], y=[row["recall"]], mode="markers+text",
            text=[row["policy"]], textposition="middle right",
            textfont=dict(family="Fraunces, Georgia, serif", size=15, color=PALETTE["chocolate"]),
            marker=dict(size=26, color=POLICY_COLORS.get(row["policy"]),
                        line=dict(color=PALETTE["cream"], width=3)),
            hovertemplate=(f"{row['policy']}<br>Catches %{{y:.1%}} of unreliable readings"
                           "<br>Wrongly flags %{x:.1%} of reliable ones<extra></extra>"),
            showlegend=False))
    fig.add_annotation(x=0.47, y=0.53, text="Equal trade-off line", showarrow=False,
                       font=dict(size=11, color=PALETTE["taupe"]), textangle=-40)
    fig.update_layout(**base_layout(height=440, margin=dict(l=60, r=20, t=16, b=54)))
    fig.update_xaxes(title_text="False-positive rate: reliable readings wrongly flagged",
                     tickformat=".0%", range=[0.25, 0.47])
    fig.update_yaxes(title_text="Recall: unreliable readings caught",
                     tickformat=".0%", range=[0.5, 0.95])
    return style_axes(fig)


def reward_figure(comparison):
    return policy_metric_figure(comparison, "average_reward")


def per_subject_figure(per_subject, metric):
    label, _, kind = METRICS[metric]
    fig = go.Figure()
    for policy in POLICY_ORDER:
        rows = per_subject[per_subject["policy"] == policy].sort_values("subject")
        fig.add_trace(go.Bar(
            x=[f"S{s}" for s in rows["subject"]], y=rows[metric], name=policy,
            marker_color=POLICY_COLORS[policy], marker_line_width=0,
            marker_cornerradius=6,
            hovertemplate=policy + "<br>%{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(**base_layout(height=330, barmode="group", bargap=0.34, bargroupgap=0.08))
    fig.update_yaxes(title_text=label, tickformat=".0%" if kind == "pct" else None)
    return style_axes(fig)


def budget_ablation_figure(budget):
    frame = budget.sort_values("budget")
    fig = go.Figure()
    for metric, color, name, dash in (
        ("recall", PALETTE["caramel"], "Recall", "solid"),
        ("f1", PALETTE["chocolate"], "F1 score", "solid"),
        ("false_positive_rate", PALETTE["terracotta"], "False-positive rate", "dot"),
    ):
        fig.add_trace(go.Scatter(
            x=frame["budget"], y=frame[metric], mode="lines+markers", name=name,
            line=dict(color=color, width=2.6, dash=dash),
            marker=dict(size=11, line=dict(color=PALETTE["cream"], width=2)),
            hovertemplate=name + ", budget %{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(**base_layout(height=330, hovermode="x unified",
                                    legend=dict(orientation="h", yanchor="top", y=-0.25, x=0),
                                    margin=dict(l=48, r=12, t=14, b=96)))
    fig.update_xaxes(title_text="Checks allowed per window", dtick=1)
    fig.update_yaxes(range=[0, 1])
    return style_axes(fig)


def action_ablation_figure(action):
    fig = go.Figure()
    for metric, color, name in (
        ("f1", PALETTE["chocolate"], "F1 score"),
        ("false_positive_rate", PALETTE["terracotta"], "False-positive rate"),
    ):
        fig.add_trace(go.Bar(
            x=[c.replace(" ", "<br>", 1) for c in action["configuration"]], y=action[metric],
            name=name, marker_color=color, marker_line_width=0, marker_cornerradius=6,
            hovertemplate="%{x}<br>" + name + " %{y:.3f}<extra></extra>"))
    fig.update_layout(**base_layout(height=330, barmode="group", bargap=0.38,
                                    legend=dict(orientation="h", yanchor="top", y=-0.3, x=0),
                                    margin=dict(l=48, r=12, t=14, b=100)))
    fig.update_yaxes(range=[0, 0.55])
    fig.update_xaxes(tickangle=0)
    return style_axes(fig)


def path_distribution_figure(traces, top=10):
    counts = traces["actions"].value_counts().head(top)[::-1]
    total = len(traces)
    colors = [DECISION_COLORS.get(split_actions(a)[-1], PALETTE["caramel"]) for a in counts.index]

    fig = go.Figure(go.Bar(
        x=counts.values, y=[path_label(a) for a in counts.index], orientation="h",
        marker_color=colors, marker_line_width=0, marker_cornerradius=5, width=0.62,
        text=[f"{c / total:.1%}" for c in counts.values], textposition="outside",
        textfont=dict(color=PALETTE["chocolate"]),
        hovertemplate="%{y}<br>%{x:,} windows<extra></extra>"))
    fig.update_layout(**base_layout(height=430, showlegend=False,
                                    margin=dict(l=300, r=30, t=10, b=44)))
    fig.update_xaxes(title_text="Held-out windows", range=[0, counts.max() * 1.15])
    fig.update_yaxes(showline=False, tickfont=dict(color=PALETTE["chocolate"], size=13))
    return style_axes(fig)


EXPERIMENT_COLORS = {"original": "#A8968A", "platt": "#B96547", "isotonic": "#C99A65"}


def calibration_figure(bins, methods=("original", "platt"), min_count=20):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(color=PALETTE["beige"], dash="dot")))
    labels = {"original": "Original classifier", "platt": "Platt-calibrated", "isotonic": "Isotonic"}
    for method in methods:
        part = bins[(bins["method"] == method) & (bins["count"] >= min_count)]
        fig.add_trace(go.Scatter(
            x=part["mean_predicted"], y=part["observed_rate"], mode="lines+markers", name=labels[method],
            line=dict(color=EXPERIMENT_COLORS[method], width=2.4),
            marker=dict(size=9, line=dict(color=PALETTE["cream"], width=2)),
            customdata=part["count"],
            hovertemplate="Predicted %{x:.0%}<br>Observed %{y:.0%}<br>%{customdata:,} windows<extra></extra>"))
    fig.update_layout(**base_layout(height=360, margin=dict(l=56, r=16, t=16, b=50)))
    fig.update_xaxes(title_text="Predicted failure probability", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title_text="Observed failure rate", tickformat=".0%", range=[0, 1])
    return style_axes(fig)


def fp_variants_figure(comparison):
    fig = go.Figure()
    styles = {
        "frozen_d2_baseline": ("circle", 20),
        "agent_calibrated_veto": ("diamond", 16),
        "probability_threshold": ("square", 14),
        "bayes_cost_rule": ("triangle-up", 15),
        "reward_retrained_policy": ("hexagon", 16),
    }
    names = {
        "agent_calibrated_veto": "Agent + calibrated veto",
        "probability_threshold": "Calibrated threshold",
        "bayes_cost_rule": "Cost-based rule",
        "reward_retrained_policy": "Retrained, original reward (3 seeds)",
    }
    for _, row in comparison.iterrows():
        symbol, size = styles.get(row["family"], ("circle", 14))
        frozen = row["family"] == "frozen_d2_baseline"
        label = row["variant"] if frozen else names.get(row["family"], row["family"])
        color = POLICY_COLORS.get(row["variant"], PALETTE["caramel"]) if frozen else PALETTE["peach"]
        error_x = error_y = None
        if pd.notna(row.get("false_positive_rate_seed_sd")):
            error_x = dict(type="data", array=[row["false_positive_rate_seed_sd"]], color=PALETTE["taupe"])
            error_y = dict(type="data", array=[row["recall_seed_sd"]], color=PALETTE["taupe"])
        fig.add_trace(go.Scatter(
            x=[row["false_positive_rate"]], y=[row["recall"]], mode="markers", name=label,
            error_x=error_x, error_y=error_y,
            marker=dict(symbol=symbol, size=size, color=color,
                        line=dict(color=PALETTE["chocolate"] if not frozen else PALETTE["cream"], width=1.5 if not frozen else 3)),
            hovertemplate=f"{label}<br>Recall %{{y:.1%}}<br>False-positive rate %{{x:.1%}}<extra></extra>"))
    fig.update_layout(**base_layout(height=420, margin=dict(l=60, r=16, t=16, b=120),
                                    legend=dict(orientation="h", yanchor="top", y=-0.2, x=0)))
    fig.update_xaxes(title_text="False-positive rate (test)", tickformat=".0%")
    fig.update_yaxes(title_text="Recall (test)", tickformat=".0%")
    return style_axes(fig)


def robustness_figure(summary, metric="recall"):
    learned = summary[summary["policy"] == "Learned Fitted Q"].reset_index(drop=True)
    labels = ["Clean" if p == "clean" else f"{p.replace('_', ' ').capitalize()} {l:g}"
              for p, l in zip(learned["perturbation"], learned["level"])]
    fig = go.Figure()
    for policy, color in (("No Investigation", POLICY_COLORS["No Investigation"]),
                          ("Fixed Rule", POLICY_COLORS["Fixed Rule"]),
                          ("Learned Fitted Q", POLICY_COLORS["Learned Fitted Q"])):
        part = summary[summary["policy"] == policy].reset_index(drop=True)
        fig.add_trace(go.Scatter(
            x=part[metric], y=labels, mode="markers", name=policy,
            marker=dict(size=11 if policy == "Learned Fitted Q" else 9, color=color,
                        line=dict(color=PALETTE["cream"], width=1.5)),
            hovertemplate=policy + "<br>%{y}<br>%{x:.1%}<extra></extra>"))
    title = {"recall": "Recall", "false_positive_rate": "False-positive rate"}[metric]
    fig.update_layout(**base_layout(height=560, margin=dict(l=230, r=16, t=30, b=46)))
    fig.update_xaxes(title_text=title, tickformat=".0%", range=[0, 1.02])
    fig.update_yaxes(autorange="reversed", showline=False, tickfont=dict(color=PALETTE["chocolate"], size=12))
    return style_axes(fig)


def motion_quartile_figure(motion):
    learned = motion[motion["policy"] == "Learned Fitted Q"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=learned["motion_quartile"], y=learned["unreliable_rate"], name="Actual failure rate",
                         marker_color=PALETTE["caramel"], marker_cornerradius=6,
                         hovertemplate="%{x}<br>Failure rate %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Bar(x=learned["motion_quartile"], y=learned["false_positive_rate"],
                         name="Agent false-positive rate", marker_color=PALETTE["terracotta"],
                         marker_cornerradius=6,
                         hovertemplate="%{x}<br>False-positive rate %{y:.1%}<extra></extra>"))
    fig.update_layout(**base_layout(height=330, barmode="group", bargap=0.35,
                                    margin=dict(l=54, r=12, t=30, b=46)))
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return style_axes(fig)

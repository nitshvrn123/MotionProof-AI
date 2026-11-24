"""Pure helpers for the D3 interface. No Streamlit imports, so they are testable."""

import math

INVESTIGATIONS = ("CHECK_PPG", "CHECK_MOTION", "CHECK_CONSISTENCY")
TERMINALS = ("ACCEPT", "FLAG_UNRELIABLE")

ACTION_LABELS = {
    "CHECK_PPG": "PPG check",
    "CHECK_MOTION": "Motion check",
    "CHECK_CONSISTENCY": "Consistency check",
    "ACCEPT": "Accept",
    "FLAG_UNRELIABLE": "Flag unreliable",
}

ACTION_TITLES = {
    "CHECK_PPG": "PPG signal check",
    "CHECK_MOTION": "Motion check",
    "CHECK_CONSISTENCY": "Temporal consistency check",
}

ACTION_ICONS = {
    "CHECK_PPG": "♡",
    "CHECK_MOTION": "↯",
    "CHECK_CONSISTENCY": "≈",
}

ACTION_SUMMARIES = {
    "CHECK_PPG": "Spectral heart-rate estimate and periodicity of the PPG window.",
    "CHECK_MOTION": "Acceleration-magnitude variation, range and dominant frequency.",
    "CHECK_CONSISTENCY": "Comparison with the model's recent heart-rate predictions.",
}

# Behaviour filters for recording discovery. Each maps to a predicate on an
# action list taken from the verified trace cache.
BEHAVIORS = {
    "Any behaviour": lambda a: True,
    "Immediate ACCEPT": lambda a: a == ["ACCEPT"],
    "Immediate FLAG_UNRELIABLE": lambda a: a == ["FLAG_UNRELIABLE"],
    "Uses CHECK_PPG": lambda a: "CHECK_PPG" in a,
    "Uses CHECK_MOTION": lambda a: "CHECK_MOTION" in a,
    "Uses CHECK_CONSISTENCY": lambda a: "CHECK_CONSISTENCY" in a,
    "Two investigations": lambda a: sum(x in INVESTIGATIONS for x in a) == 2,
    "Investigated, then ACCEPT": lambda a: len(a) > 1 and a[-1] == "ACCEPT",
    "Investigated, then FLAG": lambda a: len(a) > 1 and a[-1] == "FLAG_UNRELIABLE",
}

METRICS = {
    "recall": ("Recall", True, "pct"),
    "precision": ("Precision", True, "pct"),
    "f1": ("F1 score", True, "dec"),
    "accuracy": ("Accuracy", True, "pct"),
    "false_positive_rate": ("False-positive rate", False, "pct"),
    "average_investigations": ("Avg. investigations", None, "dec2"),
    "average_reward": ("Average reward", True, "dec"),
}

POLICY_ORDER = ("Learned Fitted Q", "Fixed Rule", "No Investigation")

ACTION_PHRASES = {
    "CHECK_PPG": "checked the pulse signal",
    "CHECK_MOTION": "checked the motion data",
    "CHECK_CONSISTENCY": "compared it with recent predictions",
}

DECISION_PHRASES = {
    "ACCEPT": "accepted the estimate",
    "FLAG_UNRELIABLE": "flagged the estimate as potentially unreliable",
}

DECISION_SHORT = {"ACCEPT": "accepted it", "FLAG_UNRELIABLE": "flagged it"}


def action_story(actions, subject="The agent"):
    """Describe a recorded action sequence in one sentence.

    Only restates which actions happened and in what order; it never
    attributes a reason to the policy.
    """

    if isinstance(actions, str):
        actions = split_actions(actions)

    if not actions:
        return ""

    decision = DECISION_PHRASES.get(actions[-1], actions[-1])
    checks = [ACTION_PHRASES[a] for a in actions[:-1] if a in ACTION_PHRASES]

    if not checks:
        return f"{subject} {decision} without running any checks."

    return f"{subject} {', then '.join(checks)}, and {decision}."


def journey_headline(actions):
    """Short factual headline for the journey page."""

    if isinstance(actions, str):
        actions = split_actions(actions)

    if not actions:
        return ""

    checks = sum(a in INVESTIGATIONS for a in actions)
    ending = "accepted" if actions[-1] == "ACCEPT" else "flagged"

    if checks == 0:
        return f"No checks. Straight to {ending}."

    count = {1: "One check", 2: "Two checks"}.get(checks, f"{checks} checks")
    return f"{count}, then {ending}."


def studio_headline(record):
    if record is None:
        return None

    decision = record["agent_result"].get("final_decision", "")
    return (
        f"The model reads {record['predicted_hr']:.1f} bpm. "
        f"The agent {DECISION_SHORT.get(decision, decision)}."
    )


def split_actions(actions):
    """Turn a pipe-joined trace string into a list of actions."""

    if not isinstance(actions, str) or not actions:
        return []

    return actions.split("|")


def behavior_mask(action_series, behavior):
    """Boolean list: which traces match a named behaviour."""

    predicate = BEHAVIORS[behavior]
    return [bool(predicate(split_actions(a))) for a in action_series]


def next_matching_window(windows, matches, current):
    """Next matching window after current, wrapping around. None if none."""

    candidates = [w for w, m in zip(windows, matches) if m]

    if not candidates:
        return None

    later = [w for w in candidates if w > current]
    return min(later) if later else min(candidates)


def previous_matching_window(windows, matches, current):
    candidates = [w for w, m in zip(windows, matches) if m]

    if not candidates:
        return None

    earlier = [w for w in candidates if w < current]
    return max(earlier) if earlier else max(candidates)


def path_label(actions):
    """Readable label for an action sequence."""

    if isinstance(actions, str):
        actions = split_actions(actions)

    if not actions:
        return "No actions"

    if len(actions) == 1:
        return f"Direct {ACTION_LABELS.get(actions[0], actions[0]).lower()}"

    return " → ".join(ACTION_LABELS.get(a, a) for a in actions)


def format_metric(value, kind):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"

    if kind == "pct":
        return f"{value * 100:.1f}%"

    if kind == "dec2":
        return f"{value:.2f}"

    return f"{value:.3f}"


def safe_number(value, decimals=1, suffix=""):
    if isinstance(value, bool) or value is None:
        return "—"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"

    if not math.isfinite(value):
        return "—"

    return f"{value:.{decimals}f}{suffix}"


def analysis_markdown(record, explanations):
    """Plain-language export of one analysis. No medical conclusions."""

    result = record["agent_result"]
    actions = result.get("actions", [])
    error = abs(record["predicted_hr"] - record["reference_hr"])

    lines = [
        "# MotionProof AI — analysis summary",
        "",
        f"Recording: participant S{record['subject']}, "
        f"window {record['window']} (8-second window, PPG-DaLiA held-out test set)",
        "",
        "## Model estimates (inference)",
        f"- CNN predicted heart rate: {record['predicted_hr']:.1f} BPM",
        "- Reliability classifier estimated failure probability "
        f"(P[|error| > 10 BPM]): {record['probability'] * 100:.1f}%",
        "",
        "## Learned agent",
        f"- Action sequence: {' → '.join(actions)}",
        f"- Investigations used: {result.get('investigations_used', 0)} / 2",
        f"- Final decision: {result.get('final_decision', '')}",
        "",
    ]

    if explanations:
        lines.append("## Investigation measurements")
        for action, text in explanations:
            lines.append(f"- {ACTION_TITLES.get(action, action)}: {text}")
        lines.append("")

    lines += [
        "## Evaluation only (not available to the models or agent)",
        f"- Reference heart rate (ECG-derived): {record['reference_hr']:.1f} BPM",
        f"- Absolute prediction error: {error:.1f} BPM",
        "",
        "---",
        "Research prototype. Not a medical device and not a diagnosis. "
        "Measurement descriptions are factual summaries and are not "
        "necessarily the reason the policy chose an action.",
    ]

    return "\n".join(lines)

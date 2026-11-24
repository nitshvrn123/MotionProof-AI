"""HTML building blocks for the D3 interface.

Markup is flattened before rendering so Markdown never treats indented HTML
as a code block.
"""

from html import escape

import streamlit as st

from ui.analysis import ACTION_LABELS, ACTION_TITLES, INVESTIGATIONS, action_story
from ui.theme import PALETTE, TOOL_COLORS


def flatten(markup):
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def render(markup, target=st):
    target.markdown(flatten(markup), unsafe_allow_html=True)


PULSE_PATH = "M2 20 H14 L18 10 L23 30 L28 4 L33 26 L37 20 H54"


def wordmark():
    return f"""
    <div class="mp-wordmark">
      <svg width="40" height="40" viewBox="0 0 56 40" aria-hidden="true">
        <rect x="0" y="0" width="56" height="40" rx="20" fill="{PALETTE['chocolate']}"/>
        <path d="{PULSE_PATH}" transform="translate(1 0)" stroke="{PALETTE['peach']}"
              stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <div>
        <div class="mp-wordmark-name">MotionProof</div>
        <div class="mp-wordmark-tag">Should this heart-rate reading be trusted?</div>
      </div>
    </div>
    """


def kicker(text):
    return f'<div class="mp-kicker">{escape(text)}</div>'


def headline(text, wide=False):
    css = "mp-headline wide" if wide else "mp-headline"
    return f'<div class="{css}" role="heading" aria-level="1">{escape(text)}</div>'


def lede(text):
    return f'<div class="mp-lede">{escape(text)}</div>'


def h2(title, sub=""):
    sub_html = f'<div class="mp-h2-sub">{escape(sub)}</div>' if sub else ""
    return f'<div class="mp-h2" role="heading" aria-level="2">{escape(title)}</div>{sub_html}'


def body(html_text):
    return f'<div class="mp-body">{html_text}</div>'


def caption(html_text):
    return f'<div class="mp-caption">{html_text}</div>'


# ---------------------------------------------------------------- sidebar

def side_intro():
    return """
    <div class="mp-side-title">Recordings</div>
    <div class="mp-side-sub">Eight-second windows from three people the models never saw during training.</div>
    """


def side_label(text):
    return f'<div class="mp-side-label">{escape(text)}</div>'


def window_counter(window, total):
    return (
        f'<div class="mp-window-number">{window:,}</div>'
        f'<div class="mp-window-of">window, of {total - 1:,} in this recording</div>'
    )


def side_rule():
    return '<div class="mp-side-rule"></div>'


# ---------------------------------------------------------------- studio

def ticket(subject, window, record):
    state = "Analyzed" if record else "Not analyzed yet"
    pending = "" if record else f"""
    <div class="mp-ticket-pending">
      <svg width="54" height="34" viewBox="0 0 56 40" aria-hidden="true">
        <path d="{PULSE_PATH}" stroke="{PALETTE['caramel']}" stroke-width="2"
              fill="none" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <span>Run the analysis to get the CNN's heart-rate estimate, its reliability
      score and the agent's decision.</span>
    </div>
    """

    return f"""
    <div class="mp-ticket-top">
      <div class="mp-ticket-title">Participant S{subject}</div>
      <div class="mp-ticket-state">{state}</div>
    </div>
    <div class="mp-ticket-grid">
      <div><span>Window</span><b>{window:,}</b></div>
      <div><span>Length</span><b>8 seconds</b></div>
      <div><span>Pulse sensor</span><b>PPG, 64 Hz</b></div>
      <div><span>Motion sensor</span><b>3-axis, 32 Hz</b></div>
    </div>
    {pending}
    """


def gauge(probability):
    p = max(0.0, min(1.0, float(probability)))
    return f"""
    <svg viewBox="0 0 200 128" width="100%" style="max-width:250px;display:block;margin:0 auto"
         role="img" aria-label="Estimated failure probability {p * 100:.1f} percent">
      <path d="M20 104 A80 80 0 0 1 180 104" stroke="{PALETTE['beige']}" stroke-width="14"
            fill="none" stroke-linecap="round" pathLength="100"/>
      <path d="M20 104 A80 80 0 0 1 180 104" stroke="{PALETTE['terracotta']}" stroke-width="14"
            fill="none" stroke-linecap="round" pathLength="100"
            stroke-dasharray="{max(p * 100, 0.6):.2f} 100"/>
      <text x="100" y="92" text-anchor="middle" font-family="Fraunces, Georgia, serif"
            font-size="38" font-weight="300" fill="{PALETTE['chocolate']}">{p * 100:.1f}%</text>
      <text x="20" y="124" text-anchor="middle" font-size="10" fill="{PALETTE['taupe']}">0</text>
      <text x="180" y="124" text-anchor="middle" font-size="10" fill="{PALETTE['taupe']}">100</text>
    </svg>
    """


def estimate_panel(record):
    return f"""
    <div class="mp-estimate">
      <div>
        <div class="mp-label">Heart rate estimated by the CNN</div>
        <div class="mp-big">{record['predicted_hr']:.1f}<small>bpm</small></div>
        <div class="mp-small" style="margin-top:14px;max-width:34ch">
          Predicted from the pulse waveform alone. This is what the agent is asked to trust or flag.</div>
      </div>
      <div>
        {gauge(record['probability'])}
        <div class="mp-gauge-caption">Estimated failure probability
          <em>Classifier's estimate that the error exceeds 10 bpm. It is not the agent's confidence.</em></div>
      </div>
    </div>
    """


def reference_ledger(record, threshold=10.0):
    error = abs(record["predicted_hr"] - record["reference_hr"])
    scale = max(threshold * 2, error * 1.25)
    tick = threshold / scale * 100
    dot = min(error / scale * 100, 100)
    verdict = "above" if error > threshold else "within"

    return f"""
    <div class="mp-ledger-title">Checked against the ECG</div>
    <div class="mp-small">Evaluation only. The models and the agent never see these values.</div>
    <div style="margin-top:10px">
      <div class="mp-ledger-row">Reference heart rate<span class="lead"></span>
        <b>{record['reference_hr']:.1f}<small>bpm</small></b></div>
      <div class="mp-ledger-row">Absolute error<span class="lead"></span>
        <b>{error:.1f}<small>bpm</small></b></div>
    </div>
    <div class="mp-track" style="--t:{tick:.1f}%">
      <div class="rail"></div>
      <div class="tick" style="left:{tick:.1f}%"><span>10 bpm</span></div>
      <div class="dot" style="left:{dot:.1f}%"></div>
    </div>
    <div class="mp-small">The error is {verdict} the 10 bpm line that defines an unreliable
      prediction in this project.</div>
    """


def budget_dots(used, budget=2):
    dots = "".join(f'<i class="{"used" if i < used else ""}"></i>' for i in range(budget))
    return f'<span class="mp-budget">{dots}&nbsp;{used} of {budget} checks used</span>'


def path_chips(actions):
    chips = "".join(
        f'<span class="mp-chip">{escape(ACTION_LABELS.get(a, a))}</span>' for a in actions
    )
    return chips


def verdict_band(result):
    decision = result.get("final_decision", "")
    actions = result.get("actions", [])
    used = int(result.get("investigations_used", 0))
    accepted = decision == "ACCEPT"
    word = "Accepted" if accepted else "Flagged"
    note = (
        "Accepting is a judgement about the model's prediction, not a statement "
        "that the reading is medically safe."
        if accepted
        else "This policy is tuned towards catching unreliable readings, so some "
        "flags land on predictions that were in fact accurate."
    )

    return f"""
    <div class="mp-verdict {'accept' if accepted else 'flag'}">
      <div class="mp-verdict-word">{word}</div>
      <div>
        <div class="mp-verdict-text">{escape(action_story(actions))}
          <span>{escape(note)}</span></div>
        <div class="mp-chips">{path_chips(actions)}{budget_dots(used)}</div>
      </div>
    </div>
    """


def empty_state(title, text):
    return f"""
    <div class="mp-empty">
      <svg width="86" height="62" viewBox="0 0 56 40" aria-hidden="true">
        <path d="{PULSE_PATH}" stroke="{PALETTE['peach']}" stroke-width="1.8"
              fill="none" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <div>
        <div class="mp-empty-title">{escape(title)}</div>
        <div class="mp-body">{escape(text)}</div>
      </div>
    </div>
    """


def missing_data_notice(subject):
    return f"""
    <div class="mp-empty" style="margin-top:46px">
      <svg width="86" height="62" viewBox="0 0 56 40" aria-hidden="true">
        <path d="{PULSE_PATH}" stroke="{PALETTE['peach']}" stroke-width="1.8"
              fill="none" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <div>
        <div class="mp-empty-title">The recording for participant S{subject} isn't on this computer</div>
        <div class="mp-body">Signal Studio and Agent Journey need the processed PPG-DaLiA windows,
        which are not stored in the repository. Add <code>data/processed/test/S{subject}.npz</code>
        (see "Dataset and participant splits" in the README) and reload. The Research Lab works
        without them.</div>
      </div>
    </div>
    """


def missing_data_side_note(subject):
    return (f'<div class="mp-side-note">No local recording found for S{subject}. '
            "Window browsing and behaviour search need <code>data/processed/test/</code>.</div>")


# ---------------------------------------------------------------- journey

def story_map(actions):
    checks = [a for a in actions if a in INVESTIGATIONS]
    middle = f"{len(checks)} check{'s' if len(checks) != 1 else ''}" if checks else "no checks"
    decision = "accepted" if actions and actions[-1] == "ACCEPT" else "flagged"
    return f"""
    <div class="mp-story-map">
      <span><b>Observation</b></span><span class="seg"></span>
      <span><b>Investigation</b>, {middle}</span><span class="seg"></span>
      <span><b>Decision</b>, {decision}</span>
    </div>
    """


def chapter(number, title, text="", hollow=False):
    num_css = "mp-chapter-num hollow" if hollow else "mp-chapter-num"
    text_html = f'<div class="mp-chapter-text">{text}</div>' if text else ""
    return f"""
    <div class="mp-chapter">
      <div class="{num_css}">{number}</div>
      <div>
        <div class="mp-chapter-title">{escape(title)}</div>
        {text_html}
      </div>
    </div>
    """


def thread(height=48, draw=False):
    css = "mp-thread draw" if draw else "mp-thread"
    return (
        f'<svg class="{css}" width="4" height="{height}" viewBox="0 0 4 {height}" aria-hidden="true">'
        f'<path d="M2 0 V{height}"/></svg>'
    )


def tokens(used, budget=2):
    items = []
    for i in range(budget):
        if i < used:
            items.append(f'<span class="mp-token spent">{i + 1}</span>')
        else:
            items.append('<span class="mp-token unused">·</span>')
    return f'<div class="mp-tokens">{"".join(items)}</div>'


def saw_block(record):
    return f"""
    <div class="mp-saw">
      <div class="mp-saw-item"><span>CNN estimate</span>
        <b>{record['predicted_hr']:.1f}<small>bpm</small></b></div>
      <div class="mp-saw-item"><span>Estimated failure probability</span>
        <b>{record['probability'] * 100:.1f}<small>%</small></b></div>
      <div class="mp-saw-item"><span>Checks it may spend</span>{tokens(0)}</div>
    </div>
    """


def blank_investigation(record):
    return f"""
    <div class="mp-blank">
      {tokens(0)}
      <div>
        <div class="mp-blank-title">No checks were run</div>
        <div class="mp-body">Both checks went unspent. The agent's highest-valued action
        at the first step was a final decision, so it decided from the observation alone:
        {record['predicted_hr']:.1f} bpm with an estimated failure probability of
        {record['probability'] * 100:.1f}%.</div>
      </div>
    </div>
    """


def evidence_head(action, step, total):
    color = TOOL_COLORS.get(action, PALETTE["caramel"])
    return f"""
    <div class="mp-ev-head">
      <div class="mp-ev-title">{escape(ACTION_TITLES.get(action, action))}</div>
      <div class="mp-ev-order">Check {step} of {total}</div>
    </div>
    <div class="mp-ev-rule" style="background:{color}"></div>
    """


def facts(items):
    cells = "".join(
        f"<div><b>{escape(value)}</b><span>{escape(label)}</span></div>" for label, value in items
    )
    return f'<div class="mp-facts">{cells}</div>'


def decision_block(record, frequency_html=""):
    result = record["agent_result"]
    decision = result.get("final_decision", "")
    accepted = decision == "ACCEPT"
    word = "Accepted" if accepted else "Flagged as unreliable"
    used = int(result.get("investigations_used", 0))
    meta = f"Final action <b>{escape(decision)}</b>, {used} of 2 checks used."

    return f"""
    <div class="mp-decision {'accept' if accepted else 'flag'}">
      <div class="mp-decision-word">{word}</div>
      <div class="mp-decision-text">{escape(action_story(result.get('actions', [])))}
        {frequency_html}</div>
      <div class="mp-decision-meta">{meta}</div>
    </div>
    """


# ---------------------------------------------------------------- research

def finding(number_text, color, title, text, flip=False):
    return f"""
    <div class="mp-finding{' flip' if flip else ''}">
      <div class="mp-finding-num" style="color:{color}">{escape(number_text)}</div>
      <div>
        <div class="mp-finding-title">{escape(title)}</div>
        <div class="mp-body">{text}</div>
      </div>
    </div>
    """


def experiments_banner():
    return """
    <div class="mp-exp">
      <div class="mp-exp-tag">Experimental, not part of the frozen D2 evaluation</div>
      <div class="mp-exp-title">Since D2: three follow-up studies</div>
      <div class="mp-body" style="margin-top:14px">Every choice below was made on validation participants
      S11 and S12, then reported once on S13 to S15. The frozen results above are unchanged.
      Full tables are in docs/EXPERIMENTS.md.</div>
    </div>
    """


def study(title, sub):
    return f'<div class="mp-study">{escape(title)}</div><div class="mp-study-sub">{escape(sub)}</div>'


def callout(text):
    return f'<div class="mp-callout">{escape(text)}</div>'


def footer():
    return """
    <div class="mp-footer">
      <div><b>MotionProof</b><br>EEE 6778 Applied Machine Learning II, University of Florida</div>
      <div style="text-align:right">A research prototype, not a medical device.<br>
      Data: PPG-DaLiA, held-out participants S13, S14 and S15.</div>
    </div>
    """

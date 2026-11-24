"""Visual system for MotionProof D3: warm earth palette, editorial type, chart styling."""

import streamlit as st

PALETTE = {
    "oatmeal": "#F7F1E8",
    "cream": "#FFF9F1",
    "sand": "#EDE0D1",
    "terracotta": "#B96547",
    "terracotta_deep": "#9C4F35",
    "caramel": "#C99A65",
    "peach": "#E5B8A4",
    "peach_soft": "#F6E3D9",
    "chocolate": "#3B2B25",
    "brown": "#66544A",
    "taupe": "#948277",
    "beige": "#E4D5C5",
    "grid": "#EDE2D4",
    "olive": "#6E8455",
    "olive_soft": "#E7EBDB",
}

POLICY_COLORS = {
    "Learned Fitted Q": "#B96547",
    "Fixed Rule": "#C99A65",
    "No Investigation": "#A8968A",
}

TOOL_COLORS = {
    "CHECK_PPG": "#B96547",
    "CHECK_MOTION": "#C99A65",
    "CHECK_CONSISTENCY": "#D79C86",
}

DECISION_COLORS = {"ACCEPT": "#6E8455", "FLAG_UNRELIABLE": "#B96547"}

FONT_BODY = "'Figtree', 'Helvetica Neue', system-ui, sans-serif"
FONT_DISPLAY = "'Fraunces', 'Iowan Old Style', Georgia, serif"

_CSS = """
<style>
@font-face { font-family: 'Fraunces'; font-style: normal; font-display: swap; font-weight: 100 900;
  src: url('app/static/fonts/fraunces-latin-full-normal.woff2') format('woff2-variations'),
       url('app/static/fonts/fraunces-latin-full-normal.woff2') format('woff2'); }
@font-face { font-family: 'Figtree'; font-weight: 400; font-display: swap; src: url('app/static/fonts/figtree-latin-400-normal.woff2') format('woff2'); }
@font-face { font-family: 'Figtree'; font-weight: 500; font-display: swap; src: url('app/static/fonts/figtree-latin-500-normal.woff2') format('woff2'); }
@font-face { font-family: 'Figtree'; font-weight: 600; font-display: swap; src: url('app/static/fonts/figtree-latin-600-normal.woff2') format('woff2'); }
@font-face { font-family: 'Figtree'; font-weight: 700; font-display: swap; src: url('app/static/fonts/figtree-latin-700-normal.woff2') format('woff2'); }

:root { color-scheme: light; }

html, body, .stApp, [data-testid="stSidebar"], button, input, textarea, select {
  font-family: %(body)s;
}
.stApp {
  background-color: %(oatmeal)s;
  background-image:
    radial-gradient(1100px 520px at 78%% -12%%, rgba(229,184,164,.28), transparent 62%%),
    url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0.35 0 0 0 0 0.24 0 0 0 0 0.17 0 0 0 .05 0'/></filter><rect width='100%%' height='100%%' filter='url(%%23n)'/></svg>");
  color: %(brown)s;
}
[data-testid="stHeader"], [data-testid="stHeader"] [data-testid="stToolbar"], [data-testid="stHeader"] div { background: transparent; pointer-events: none !important; }
[data-testid="stHeader"] button, [data-testid="stHeader"] a, [data-testid="stMainMenu"], [data-testid="stMainMenu"] * { pointer-events: auto !important; }
[data-testid="stAppDeployButton"], [data-testid="stDecoration"] { display: none; }
.block-container { max-width: 1240px; padding: 2.6rem 2.4rem 4rem; }
p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] { color: %(brown)s; }
h1, h2, h3, h4 { font-family: %(display)s; color: %(chocolate)s !important; }
a { color: %(terracotta)s; }
*:focus-visible { outline: 2px solid %(terracotta)s !important; outline-offset: 2px; }

/* ---------- sidebar: the recording explorer ---------- */
[data-testid="stSidebar"] { background: %(sand)s; border-right: 1px solid %(beige)s; }
[data-testid="stSidebarUserContent"] { padding: 1.6rem 1.4rem 2rem; }
.mp-side-title { font-family: %(display)s; font-size: 26px; font-weight: 400; color: %(chocolate)s; letter-spacing: -.01em; margin: 0 0 2px; }
.mp-side-sub { font-size: 13px; color: %(taupe)s; margin-bottom: 18px; line-height: 1.5; }
.mp-side-label { font-size: 13px; font-weight: 600; color: %(chocolate)s; margin: 20px 0 8px; }
.mp-window-number { font-family: %(display)s; font-size: 56px; font-weight: 300; line-height: .95; color: %(chocolate)s; font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
.mp-window-of { font-size: 13px; color: %(taupe)s; margin: 6px 0 4px; }
.mp-side-note { font-size: 12.5px; color: %(brown)s; line-height: 1.55; margin-top: 8px; }
.mp-side-note b { color: %(chocolate)s; }
[data-testid="stSidebar"] div.stButton > button { padding: 0 10px; min-height: 40px; }
[data-testid="stSidebar"] div.stButton > button p { font-size: 13.5px; white-space: nowrap; }
.mp-side-rule { height: 1px; background: #DCCAB6; margin: 26px 0 4px; }

/* ---------- widgets ---------- */
[data-testid="stSlider"] [role="group"],
[data-testid="stCheckbox"] label > div:first-child { filter: hue-rotate(17deg) saturate(.52) brightness(.8); }
[data-testid="stSliderThumbValue"] { display: none; }
[data-testid="stSliderTickBar"] p { color: %(taupe)s; font-size: 12px; }

[data-testid="stButtonGroup"] button {
  background: transparent !important; border: 1px solid #D9C5B0 !important;
  color: %(brown)s !important; font-weight: 600; border-radius: 999px !important; margin-right: 6px;
}
[data-testid="stButtonGroup"] button p { font-size: 14px; color: inherit !important; }
[data-testid="stButtonGroup"] button[data-selected="true"] {
  background: %(chocolate)s !important; border-color: %(chocolate)s !important; color: %(cream)s !important;
}
[data-testid="stButtonGroup"] button:hover { border-color: %(terracotta)s !important; }

/* Top navigation reads as editorial links, not tabs */
.st-key-nav [data-testid="stButtonGroup"] { display: flex; justify-content: flex-end; }
.st-key-nav [data-testid="stButtonGroup"] button {
  border: 0 !important; border-radius: 0 !important; padding: 6px 2px !important; margin: 0 0 0 26px;
  background: transparent !important; color: %(taupe)s !important; border-bottom: 2px solid transparent !important;
}
.st-key-nav [data-testid="stButtonGroup"] button p { font-size: 15.5px; font-weight: 500; }
.st-key-nav [data-testid="stButtonGroup"] button[data-selected="true"] {
  background: transparent !important; color: %(chocolate)s !important; border-bottom: 2px solid %(terracotta)s !important;
}

div.stButton > button, div.stDownloadButton > button {
  border-radius: 999px; border: 1px solid #D9C5B0; background: %(cream)s; color: %(chocolate)s;
  font-weight: 600; padding: 0 20px; min-height: 44px;
  transition: background-color .15s ease, border-color .15s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover { border-color: %(terracotta)s; color: %(terracotta_deep)s; background: #FFFCF7; }
div.stButton > button:disabled { opacity: .45; }
div.stButton > button[kind="primary"] {
  background: %(terracotta)s; border: 0; color: #FFF9F1; min-height: 54px; font-size: 15.5px;
  box-shadow: 0 10px 24px -12px rgba(156,79,53,.7);
}
div.stButton > button[kind="primary"]:hover { background: %(terracotta_deep)s; color: #FFF9F1; }
div.stButton > button[kind="primary"] p { color: #FFF9F1; }
[data-baseweb="select"] > div { background: %(cream)s; border-radius: 14px; border-color: #D9C5B0; }
[data-testid="stExpander"] details { background: transparent; border: 1px solid %(beige)s; border-radius: 16px; }
[data-testid="stExpander"] summary p { color: %(brown)s; font-weight: 500; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
[data-testid="stStatusWidget"], [data-testid="stStatus"] { border-radius: 16px; }

/* ---------- surfaces (keyed Streamlit containers) ---------- */
.st-key-paper, .st-key-paper-lab {
  background: %(cream)s; border: 1px solid %(beige)s; border-radius: 34px 34px 34px 12px;
  padding: 26px 26px 14px; box-shadow: 0 30px 60px -40px rgba(91,60,40,.35);
}
[class*="st-key-panel"] {
  background: %(cream)s; border: 1px solid %(beige)s; border-radius: 28px; padding: 24px 26px;
}
[class*="st-key-quiet"] { background: transparent; border-top: 1px solid %(beige)s; padding-top: 18px; }
[class*="st-key-evidence"] {
  background: %(cream)s; border: 1px solid %(beige)s; border-radius: 26px 26px 26px 10px; padding: 22px 24px 12px;
}
.st-key-ticket {
  background: %(cream)s; border: 1px solid %(beige)s; border-radius: 30px; padding: 24px 24px 20px;
  box-shadow: 0 24px 50px -36px rgba(91,60,40,.45);
}

/* ---------- masthead ---------- */
.mp-wordmark { display:flex; align-items:center; gap: 12px; padding-top: 4px; }
.mp-wordmark svg { flex: none; }
.mp-wordmark-name { font-family: %(display)s; font-size: 25px; font-weight: 500; color: %(chocolate)s; letter-spacing: -.01em; }
.mp-wordmark-tag { font-size: 13px; color: %(taupe)s; }
.mp-mast-rule { height: 1px; background: %(beige)s; margin: 6px 0 0; }

/* ---------- editorial type ---------- */
.mp-kicker { font-size: 14px; color: %(terracotta)s; font-weight: 600; margin: 46px 0 14px; }
.mp-headline {
  font-family: %(display)s; font-weight: 300; color: %(chocolate)s;
  font-size: clamp(40px, 5.4vw, 74px); line-height: 1.06; letter-spacing: -.028em; margin: 0;
  font-variation-settings: "SOFT" 100, "opsz" 144; max-width: 15ch;
}
.mp-headline.wide { max-width: 19ch; }
.mp-lede { font-size: 17.5px; line-height: 1.7; color: %(brown)s; max-width: 58ch; margin: 34px 0 0; }
.mp-h2 { font-family: %(display)s; font-weight: 400; font-size: 34px; letter-spacing: -.02em; color: %(chocolate)s; margin: 64px 0 6px; line-height: 1.1; }
.mp-h2-sub { color: %(taupe)s; font-size: 15px; margin-bottom: 20px; max-width: 64ch; line-height: 1.6; }
.mp-body { font-size: 15.5px; line-height: 1.75; color: %(brown)s; max-width: 62ch; }
.mp-body b { color: %(chocolate)s; font-weight: 600; }
.mp-small { font-size: 13px; color: %(taupe)s; line-height: 1.55; }

/* ---------- ticket (recording identity in the hero) ---------- */
.mp-ticket-top { display:flex; justify-content: space-between; align-items: baseline; }
.mp-ticket-title { font-family: %(display)s; font-size: 22px; color: %(chocolate)s; }
.mp-ticket-state { font-size: 13px; color: %(taupe)s; }
.mp-ticket-grid { display:grid; grid-template-columns: 1fr 1fr; gap: 14px 18px; margin: 20px 0 18px; padding: 16px 0; border-top: 1px dashed #DCC9B4; border-bottom: 1px dashed #DCC9B4; }
.mp-ticket-grid span { display:block; font-size: 12.5px; color: %(taupe)s; }
.mp-ticket-grid b { font-family: %(display)s; font-weight: 400; font-size: 21px; color: %(chocolate)s; }
.mp-ticket-pending { display:flex; align-items:center; gap: 12px; color: %(brown)s; font-size: 14px; line-height: 1.5; margin-bottom: 14px; }

/* ---------- reading composition ---------- */
.mp-estimate { display:grid; grid-template-columns: 1.15fr 1fr; gap: 20px; align-items: center; }
.mp-label { font-size: 13.5px; color: %(taupe)s; margin-bottom: 4px; }
.mp-big { font-family: %(display)s; font-weight: 300; font-size: 88px; line-height: .95; color: %(chocolate)s; letter-spacing: -.035em; font-variant-numeric: tabular-nums; }
.mp-big small { font-family: %(body)s; font-size: 18px; color: %(taupe)s; letter-spacing: 0; margin-left: 6px; font-weight: 500; }
.mp-gauge-caption { text-align: center; font-size: 13px; color: %(brown)s; line-height: 1.5; margin-top: -6px; }
.mp-gauge-caption em { font-style: normal; color: %(taupe)s; display:block; font-size: 12px; }
.mp-ledger-title { font-family: %(display)s; font-size: 21px; color: %(chocolate)s; margin-bottom: 4px; }
.mp-ledger-row { display:flex; align-items: baseline; gap: 10px; padding: 12px 0; font-size: 14.5px; color: %(brown)s; }
.mp-ledger-row .lead { flex: 1; border-bottom: 1px dotted #CDB9A3; transform: translateY(-4px); }
.mp-ledger-row b { font-family: %(display)s; font-weight: 400; font-size: 24px; color: %(chocolate)s; font-variant-numeric: tabular-nums; }
.mp-ledger-row b small { font-family: %(body)s; font-size: 13px; color: %(taupe)s; margin-left: 3px; }
.mp-track { position: relative; height: 34px; margin: 14px 0 4px; }
.mp-track .rail { position:absolute; top: 15px; left: 0; right: 0; height: 4px; border-radius: 9px; background: linear-gradient(90deg, #E7EBDB 0%%, #E7EBDB var(--t), %(peach_soft)s var(--t), %(peach_soft)s 100%%); }
.mp-track .tick { position:absolute; top: 6px; width: 1px; height: 22px; background: %(taupe)s; }
.mp-track .tick span { position:absolute; top: -16px; left: 50%%; transform: translateX(-50%%); white-space: nowrap; font-size: 11.5px; color: %(taupe)s; }
.mp-track .dot { position:absolute; top: 9px; width: 16px; height: 16px; border-radius: 50%%; background: %(chocolate)s; border: 3px solid %(cream)s; box-shadow: 0 0 0 1px %(chocolate)s; transform: translateX(-50%%); }

/* ---------- verdict band ---------- */
.mp-verdict { border-radius: 30px; padding: 30px 34px; display:grid; grid-template-columns: auto 1fr; gap: 28px; align-items: center; }
.mp-verdict.accept { background: %(olive_soft)s; }
.mp-verdict.flag { background: %(peach_soft)s; }
.mp-verdict-word { font-family: %(display)s; font-weight: 400; font-size: 46px; letter-spacing: -.02em; line-height: 1; }
.mp-verdict.accept .mp-verdict-word { color: #55693F; }
.mp-verdict.flag .mp-verdict-word { color: %(terracotta_deep)s; }
.mp-verdict-text { font-size: 16px; color: %(chocolate)s; line-height: 1.6; }
.mp-verdict-text span { display:block; font-size: 13.5px; color: %(brown)s; margin-top: 6px; }
.mp-chips { display:flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; align-items: center; }
.mp-chip { font-size: 13px; padding: 6px 13px; border-radius: 999px; background: rgba(255,249,241,.85); border: 1px solid rgba(59,43,37,.12); color: %(chocolate)s; font-weight: 500; }
.mp-budget { display:inline-flex; gap: 5px; align-items:center; margin-left: 6px; font-size: 13px; color: %(brown)s; }
.mp-budget i { width: 11px; height: 11px; border-radius: 50%%; border: 1.5px solid %(chocolate)s; display:inline-block; }
.mp-budget i.used { background: %(chocolate)s; }

/* ---------- journey story ---------- */
.mp-story-map { display:flex; align-items:center; gap: 14px; margin: 30px 0 8px; color: %(taupe)s; font-size: 14px; flex-wrap: wrap; }
.mp-story-map b { color: %(chocolate)s; font-weight: 600; }
.mp-story-map .seg { flex: 1 1 40px; min-width: 30px; height: 1px; background: #D5C1AB; }
.mp-chapter { display:grid; grid-template-columns: 74px 1fr; gap: 10px; position: relative; margin-top: 30px; }
.mp-chapter-num {
  width: 54px; height: 54px; border-radius: 50%%; display:grid; place-items:center;
  font-family: %(display)s; font-size: 24px; color: %(cream)s; background: %(chocolate)s; position: relative; z-index: 1;
}
.mp-chapter-num.hollow { background: %(oatmeal)s; color: %(taupe)s; border: 1.5px dashed %(taupe)s; }
.mp-chapter-title { font-family: %(display)s; font-weight: 400; font-size: 38px; letter-spacing: -.02em; color: %(chocolate)s; line-height: 1.05; margin-top: 4px; }
.mp-chapter-text { font-size: 16px; color: %(brown)s; line-height: 1.7; max-width: 62ch; margin-top: 10px; }
.mp-thread { display:block; margin: 0 0 0 26px; overflow: visible; }
.mp-thread path { stroke: %(caramel)s; stroke-width: 2; fill: none; stroke-dasharray: 4 6; }
.mp-thread.draw path { stroke-dasharray: 600; stroke-dashoffset: 600; animation: mpDraw 1.1s ease-out .2s forwards; }
@keyframes mpDraw { to { stroke-dashoffset: 0; stroke-dasharray: 4 6; } }
@media (prefers-reduced-motion: reduce) { .mp-thread.draw path { animation: none; stroke-dashoffset: 0; stroke-dasharray: 4 6; } }
.mp-saw { display:flex; flex-wrap: wrap; gap: 40px; margin-top: 22px; align-items: flex-end; }
.mp-saw-item span { display:block; font-size: 13px; color: %(taupe)s; margin-bottom: 4px; }
.mp-saw-item b { font-family: %(display)s; font-weight: 300; font-size: 48px; color: %(chocolate)s; letter-spacing: -.02em; line-height: 1; }
.mp-saw-item b small { font-family: %(body)s; font-size: 15px; color: %(taupe)s; font-weight: 500; margin-left: 4px; }
.mp-tokens { display:flex; gap: 10px; align-items:center; }
.mp-token { width: 40px; height: 40px; border-radius: 50%%; border: 1.5px solid %(chocolate)s; display:grid; place-items:center; font-size: 13px; color: %(chocolate)s; }
.mp-token.spent { background: %(chocolate)s; color: %(cream)s; }
.mp-token.unused { border-style: dashed; color: %(taupe)s; border-color: %(taupe)s; }
.mp-blank {
  margin-top: 22px; border: 1.5px dashed #D5C1AB; border-radius: 28px; padding: 30px 32px;
  display:grid; grid-template-columns: auto 1fr; gap: 26px; align-items: center; background: rgba(255,249,241,.5);
}
.mp-blank-title { font-family: %(display)s; font-size: 24px; color: %(chocolate)s; margin-bottom: 4px; }
.mp-ev-head { display:flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.mp-ev-title { font-family: %(display)s; font-size: 24px; color: %(chocolate)s; letter-spacing: -.01em; }
.mp-ev-order { font-size: 13px; color: %(taupe)s; white-space: nowrap; }
.mp-ev-rule { height: 3px; width: 42px; border-radius: 3px; margin: 10px 0 14px; }
.mp-ev-text { font-size: 15px; color: %(chocolate)s; line-height: 1.7; }
.mp-facts { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 8px 18px; margin: 18px 0 6px; }
.mp-facts div { border-top: 1px solid %(beige)s; padding-top: 10px; }
.mp-facts b { display:block; font-family: %(display)s; font-weight: 400; font-size: 22px; color: %(chocolate)s; font-variant-numeric: tabular-nums; }
.mp-facts span { font-size: 12.5px; color: %(taupe)s; }
.mp-decision { margin-top: 22px; border-radius: 32px; padding: 34px 36px; }
.mp-decision.accept { background: %(olive_soft)s; }
.mp-decision.flag { background: %(peach_soft)s; }
.mp-decision-word { font-family: %(display)s; font-weight: 300; font-size: clamp(44px, 6vw, 80px); letter-spacing: -.03em; line-height: 1; }
.mp-decision.accept .mp-decision-word { color: #55693F; }
.mp-decision.flag .mp-decision-word { color: %(terracotta_deep)s; }
.mp-decision-text { font-size: 16px; line-height: 1.7; color: %(chocolate)s; margin-top: 16px; max-width: 64ch; }
.mp-decision-meta { margin-top: 16px; font-size: 13.5px; color: %(brown)s; }

/* ---------- research ---------- */
.mp-finding { display:grid; grid-template-columns: minmax(220px, 0.9fr) 1.4fr; gap: 36px; align-items: center; padding: 34px 0; border-top: 1px solid %(beige)s; }
.mp-finding.flip { grid-template-columns: 1.4fr minmax(220px, 0.9fr); }
.mp-finding.flip .mp-finding-num { order: 2; text-align: right; }
.mp-finding-num { font-family: %(display)s; font-weight: 300; font-size: clamp(64px, 8vw, 116px); line-height: .9; letter-spacing: -.04em; font-variant-numeric: tabular-nums; }
.mp-finding-title { font-family: %(display)s; font-size: 26px; color: %(chocolate)s; margin-bottom: 8px; letter-spacing: -.01em; }
.mp-callout { border-left: 3px solid %(terracotta)s; padding: 4px 0 4px 20px; margin: 18px 0; font-family: %(display)s; font-size: 22px; line-height: 1.45; color: %(chocolate)s; font-weight: 300; max-width: 40ch; }
.mp-caption { font-size: 13.5px; color: %(brown)s; line-height: 1.65; margin-top: 6px; max-width: 70ch; }
.mp-caption b { color: %(chocolate)s; }

/* ---------- post-D2 experiments ---------- */
.mp-exp { margin-top: 90px; border: 1.5px dashed %(caramel)s; border-radius: 30px; padding: 26px 30px; background: rgba(201,154,101,.07); }
.mp-exp-tag { display:inline-block; font-size: 13px; font-weight: 600; color: #8A5E2E; background: rgba(201,154,101,.22); padding: 5px 12px; border-radius: 999px; margin-bottom: 14px; }
.mp-exp-title { font-family: %(display)s; font-weight: 300; font-size: clamp(34px, 4vw, 52px); color: %(chocolate)s; letter-spacing: -.025em; line-height: 1.05; }
.mp-study { font-family: %(display)s; font-size: 26px; color: %(chocolate)s; margin: 54px 0 4px; letter-spacing: -.01em; }
.mp-study-sub { font-size: 13.5px; color: %(taupe)s; margin-bottom: 16px; }

/* ---------- empty + footer ---------- */
.mp-empty { text-align: left; padding: 40px 36px; border: 1.5px dashed #D5C1AB; border-radius: 30px; display:grid; grid-template-columns: auto 1fr; gap: 26px; align-items:center; }
.mp-empty-title { font-family: %(display)s; font-size: 26px; color: %(chocolate)s; margin-bottom: 4px; }
.mp-footer { margin-top: 80px; padding-top: 22px; border-top: 1px solid %(beige)s; display:flex; justify-content: space-between; gap: 18px; flex-wrap: wrap; font-size: 13px; color: %(taupe)s; }
.mp-footer b { font-family: %(display)s; font-weight: 400; color: %(chocolate)s; font-size: 16px; }

@media (max-width: 900px) {
  .block-container { padding: 1rem 1.1rem 3rem; }
  .mp-estimate, .mp-verdict, .mp-finding, .mp-finding.flip, .mp-blank, .mp-empty { grid-template-columns: 1fr; }
  .mp-finding.flip .mp-finding-num { order: 0; text-align: left; }
  .mp-big { font-size: 64px; }
  .mp-facts { grid-template-columns: 1fr 1fr; }
  .st-key-nav [data-testid="stButtonGroup"] { justify-content: flex-start; }
  .st-key-nav [data-testid="stButtonGroup"] button { margin: 0 20px 0 0; }
}
</style>
"""


def inject_css():
    tokens = dict(PALETTE, body=FONT_BODY, display=FONT_DISPLAY)
    st.markdown(_CSS % tokens, unsafe_allow_html=True)


def base_layout(height=320, **overrides):
    """Shared Plotly layout: transparent paper, warm type, quiet legend."""

    layout = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Figtree, Helvetica Neue, sans-serif", size=12.5,
                  color=PALETTE["taupe"]),
        margin=dict(l=54, r=16, t=40, b=44),
        hoverlabel=dict(bgcolor=PALETTE["cream"], bordercolor=PALETTE["beige"],
                        font=dict(color=PALETTE["chocolate"], family="Figtree, sans-serif")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=PALETTE["brown"])),
        title=dict(font=dict(family="Fraunces, Georgia, serif", size=17,
                             color=PALETTE["chocolate"]),
                   x=0, xanchor="left", y=0.98),
    )
    layout.update(overrides)
    return layout


def style_axes(fig):
    axis = dict(gridcolor=PALETTE["grid"], griddash="dot", zeroline=False,
                linecolor=PALETTE["beige"], tickcolor=PALETTE["beige"], showline=True,
                title_font=dict(size=12.5, color=PALETTE["taupe"]),
                tickfont=dict(color=PALETTE["taupe"]))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig

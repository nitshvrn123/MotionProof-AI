# Exploratory usability feedback (D4)

This is an informal, exploratory usability check of the MotionProof AI Streamlit interface with 3–5 volunteers. It is not a clinical evaluation, not a statistically representative study, and does not collect health data. Confirm with the course instructor that informal classmate feedback is acceptable before running sessions.

## What is collected

- Task success (yes / partly / no) and approximate time per task, recorded by the facilitator.
- The 10-item System Usability Scale (SUS), answered on a 1–5 scale.
- Four short questions about clarity and trust (1–5 scale) and free-text comments.

No names, e-mail addresses, photos, screen or audio recordings, or physiological data are collected. Each volunteer is identified only as P1, P2, … in `responses_template.csv`.

## Before the session

1. Launch the app: `streamlit run app_d3.py` (processed recordings in `data/processed/`).
2. Read the participant a one-line description: "MotionProof AI estimates heart rate from a wrist sensor and decides whether each estimate can be trusted. It is a research prototype, not a medical device."
3. Tell them they can stop at any time and that the interface, not them, is being evaluated.

## Tasks (about 10 minutes)

| # | Task | Success means |
|---|---|---|
| T1 | Open participant S15 and find a window where the agent flagged the estimate as unreliable. | A flagged window is on screen |
| T2 | Explain in your own words why the agent flagged it. | Mentions a check result (e.g., pulse frequency disagreement or strong motion) |
| T3 | Find a window where the agent accepted the estimate without running any checks. | Uses the behaviour filter or navigation to reach one |
| T4 | Open the Research Lab and say which policy catches the most failures and what it costs. | Names the learned agent and mentions more false alarms |

## After the tasks

Hand over `sus_questionnaire.md`, then the four clarity and trust questions in the same file. Enter answers in a copy of `responses_template.csv`.

## Scoring

```
python -m src.experiments.score_usability docs/usability/responses.csv
```

The script computes each participant's SUS score (0–100), the mean and range, task success rates, and the mean of each clarity and trust question. With 3–5 volunteers, report individual scores and the range rather than claiming a population average.

"""Tests for the D3 interface helpers and the verified agent-trace cache."""

from pathlib import Path

import pandas as pd
import pytest

from ui.analysis import (
    analysis_markdown,
    behavior_mask,
    format_metric,
    next_matching_window,
    path_label,
    previous_matching_window,
    safe_number,
    split_actions,
)
from ui.analysis import action_story, journey_headline, studio_headline
from ui.components import (
    blank_investigation,
    flatten,
    reference_ledger,
    story_map,
    verdict_band,
)

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "results" / "demo" / "learned_agent_traces.csv"
DECISIONS = ROOT / "results" / "metrics" / "final_test" / "test_policy_decisions.csv"


def make_record(actions, findings=None, history=None):
    used = sum(a.startswith("CHECK_") for a in actions)
    return {
        "subject": 13,
        "window": 7,
        "predicted_hr": 80.8,
        "probability": 0.213,
        "history": history or [],
        "reference_hr": 70.0,
        "agent_result": {
            "final_decision": actions[-1],
            "actions": actions,
            "investigations_used": used,
            "remaining_budget": 2 - used,
            "investigation_results": findings or {},
        },
    }


def test_split_actions_handles_empty_and_pipe_strings():
    assert split_actions("CHECK_PPG|ACCEPT") == ["CHECK_PPG", "ACCEPT"]
    assert split_actions("") == []
    assert split_actions(None) == []


def test_behavior_mask_matches_expected_paths():
    series = ["ACCEPT", "FLAG_UNRELIABLE", "CHECK_PPG|CHECK_MOTION|ACCEPT",
              "CHECK_CONSISTENCY|FLAG_UNRELIABLE"]
    assert behavior_mask(series, "Immediate ACCEPT") == [True, False, False, False]
    assert behavior_mask(series, "Two investigations") == [False, False, True, False]
    assert behavior_mask(series, "Uses CHECK_CONSISTENCY") == [False, False, False, True]
    assert behavior_mask(series, "Investigated, then FLAG") == [False, False, False, True]


def test_matching_window_navigation_wraps():
    windows = [0, 1, 2, 3, 4]
    matches = [False, True, False, True, False]
    assert next_matching_window(windows, matches, 1) == 3
    assert next_matching_window(windows, matches, 3) == 1
    assert previous_matching_window(windows, matches, 3) == 1
    assert previous_matching_window(windows, matches, 1) == 3
    assert next_matching_window(windows, [False] * 5, 0) is None


def test_labels_and_number_formatting():
    assert path_label(["ACCEPT"]) == "Direct accept"
    assert path_label("CHECK_PPG|FLAG_UNRELIABLE") == "PPG check → Flag unreliable"
    assert format_metric(0.88137, "pct") == "88.1%"
    assert format_metric(float("nan"), "pct") == "—"
    assert safe_number(float("inf")) == "—"
    assert safe_number(True) == "—"
    assert safe_number(9.64, 1, " BPM") == "9.6 BPM"


def test_flatten_removes_indentation_that_markdown_treats_as_code():
    flat = flatten("<div>\n    <span>a</span>\n</div>")
    assert "\n" not in flat
    assert not flat.startswith(" ")


def test_action_story_restates_recorded_order_only():
    assert action_story(["FLAG_UNRELIABLE"]) == (
        "The agent flagged the estimate as potentially unreliable without running any checks."
    )
    story = action_story(["CHECK_MOTION", "CHECK_PPG", "ACCEPT"])
    assert story.index("motion") < story.index("pulse")
    assert story.endswith("accepted the estimate.")
    assert "because" not in story


def test_headlines_reflect_actual_checks():
    assert journey_headline(["ACCEPT"]) == "No checks. Straight to accepted."
    assert journey_headline("CHECK_PPG|CHECK_CONSISTENCY|FLAG_UNRELIABLE") == "Two checks, then flagged."
    assert studio_headline(None) is None
    assert studio_headline(make_record(["ACCEPT"])) == "The model reads 80.8 bpm. The agent accepted it."


def test_direct_decision_shows_no_fabricated_checks():
    record = make_record(["FLAG_UNRELIABLE"])
    assert "no checks" in story_map(record["agent_result"]["actions"])
    assert "No checks were run" in blank_investigation(record)
    assert "0 of 2 checks used" in verdict_band(record["agent_result"])


def test_verdict_does_not_imply_medical_safety():
    markup = verdict_band(make_record(["CHECK_PPG", "ACCEPT"])["agent_result"])
    assert "not a statement that the reading is medically safe" in markup
    assert "1 of 2 checks used" in markup


def test_reference_ledger_is_marked_evaluation_only():
    markup = reference_ledger(make_record(["ACCEPT"]))
    assert "Evaluation only" in markup
    assert "10.8" in markup
    assert "above the 10 bpm line" in markup


def test_markdown_export_separates_evaluation_values():
    text = analysis_markdown(make_record(["CHECK_PPG", "ACCEPT"]),
                             [("CHECK_PPG", "Measured something.")])
    model_part, eval_part = text.split("## Evaluation only")
    assert "70.0" not in model_part
    assert "Reference heart rate" in eval_part
    assert "Absolute prediction error: 10.8 BPM" in eval_part
    assert "Not a medical device" in text


@pytest.mark.skipif(not TRACES.exists(), reason="trace cache not built")
def test_trace_cache_reproduces_frozen_learned_decisions():
    traces = pd.read_csv(TRACES)
    decisions = pd.read_csv(DECISIONS)
    learned = decisions[decisions["policy"] == "Learned Fitted Q"]
    merged = traces.merge(learned, on=["subject", "window_index"],
                          suffixes=("", "_frozen"), validate="one_to_one")

    assert len(merged) == len(learned) == 13007
    flagged = (merged["final_decision"] == "FLAG_UNRELIABLE").astype(int)
    assert (flagged == merged["predicted_unreliable"]).all()
    assert (merged["investigations_used"] == merged["investigations_used_frozen"]).all()
    assert merged["investigations_used"].between(0, 2).all()


def test_signal_chart_panels_share_one_time_axis():
    import numpy as np
    from ui.charts import signal_figure
    fig = signal_figure(np.zeros(512), np.zeros((256, 3)), 64, 32)
    axes = {fig.layout.xaxis.matches, fig.layout.xaxis2.matches}
    assert axes & {"x", "x2"}, "PPG and ACC panels must zoom together"


EXPERIMENTS = ROOT / "results" / "experiments"


@pytest.mark.skipif(not (EXPERIMENTS / "phase3_robustness/synthetic_summary.csv").exists(),
                    reason="experiments not run")
def test_experiment_outputs_have_the_columns_the_ui_reads():
    need = {
        "phase1_calibration/test_calibration_metrics.csv": {"method", "mean_predicted", "positive_rate", "brier", "ece"},
        "phase1_calibration/test_reliability_bins.csv": {"method", "count", "mean_predicted", "observed_rate"},
        "phase2_fp_reduction/test_comparison.csv": {"family", "variant", "recall", "false_positive_rate", "f1",
                                                    "recall_seed_sd", "false_positive_rate_seed_sd"},
        "phase3_robustness/synthetic_summary.csv": {"policy", "perturbation", "level", "recall", "false_positive_rate"},
        "phase3_robustness/natural_motion_subgroups.csv": {"policy", "motion_quartile", "unreliable_rate",
                                                           "false_positive_rate"},
    }
    for name, columns in need.items():
        assert columns <= set(pd.read_csv(EXPERIMENTS / name).columns), name
    fp = pd.read_csv(EXPERIMENTS / "phase2_fp_reduction/test_comparison.csv")
    assert {"Learned Fitted Q", "Fixed Rule", "No Investigation"} <= set(fp["variant"])
    assert (fp["family"] == "agent_calibrated_veto").any()

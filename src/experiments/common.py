"""Shared plumbing for post-D2 experiments: configs, manifests, metrics, splits."""

import hashlib
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "experiments"
RESULTS_DIR = ROOT / "results" / "experiments"
EXPERIMENTAL_CHECKPOINTS = ROOT / "checkpoints" / "experimental"

TRAIN_SUBJECTS = tuple(range(1, 11))
VALIDATION_SUBJECTS = (11, 12)
TEST_SUBJECTS = (13, 14, 15)

FINAL_TEST = ROOT / "results" / "metrics" / "final_test"
VALIDATION_INPUTS = ROOT / "results" / "metrics" / "validation_policy_inputs_nonlinear.csv"
TEST_INPUTS = FINAL_TEST / "test_policy_inputs_nonlinear.csv"


def load_config(name):
    with open(CONFIG_DIR / f"{name}.json") as handle:
        return json.load(handle)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def output_dir(name):
    path = RESULTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions():
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "pandas", "scipy", "sklearn", "torch", "joblib"):
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = None
    return versions


def write_manifest(out_dir, config, inputs, outputs, notes=None):
    """Record what was run, on which inputs, producing which outputs."""

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "inputs": {str(Path(p).relative_to(ROOT)): sha256(p) for p in inputs},
        "outputs": sorted(str(Path(p).relative_to(ROOT)) for p in outputs),
        "versions": package_versions(),
        "notes": notes or [],
    }
    with open(Path(out_dir) / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2, default=float)
    return manifest


def assert_no_test_subjects(frame, column="subject"):
    leaked = set(frame[column].unique()) & set(TEST_SUBJECTS)
    if leaked:
        raise RuntimeError(f"Held-out test subjects used for fitting/selection: {sorted(leaked)}")


def classification_metrics(target, flagged, investigations=None, rewards=None):
    """Same definitions as the frozen D2 evaluation (positive = unreliable)."""

    target = np.asarray(target, dtype=int)
    flagged = np.asarray(flagged, dtype=int)
    tp = int(((flagged == 1) & (target == 1)).sum())
    fp = int(((flagged == 1) & (target == 0)).sum())
    tn = int(((flagged == 0) & (target == 0)).sum())
    fn = int(((flagged == 0) & (target == 1)).sum())
    n = len(target)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "windows": n,
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "average_investigations": float(np.mean(investigations)) if investigations is not None else 0.0,
        "average_reward": float(np.mean(rewards)) if rewards is not None else float("nan"),
        "flag_rate": flagged.mean(),
    }

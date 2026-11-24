"""Synthetic, reproducible signal perturbations for robustness testing.

Every function takes (ppg, acc, level, rng) for a batch of windows and
returns new (ppg, acc) arrays. Inputs are never modified in place, so the
clean recordings stay untouched. These are synthetic perturbations of real
held-out recordings; they are not new participants or new recordings.

Shapes: ppg (n, 512) sampled at 64 Hz, acc (n, 256, 3) sampled at 32 Hz.
"""

import numpy as np

PPG_RATE = 64
ACC_RATE = 32


def _window_std(ppg):
    std = np.std(ppg.astype(np.float64), axis=1, keepdims=True)
    return np.where(std < 1e-9, 1.0, std)


def _time(n_samples, rate):
    return np.arange(n_samples) / rate


def gaussian_noise(ppg, acc, level, rng):
    """Additive white noise with std = level x the window's PPG std."""

    noise = rng.standard_normal(ppg.shape) * _window_std(ppg) * level
    return ppg + noise, acc.copy()


def accelerometer_coupled_artifact(ppg, acc, level, rng):
    """Inject the window's own accelerometer magnitude into the PPG.

    Mimics motion artefacts that track the recorded movement. The artefact
    is the mean-removed |ACC| resampled to 64 Hz, scaled to level x PPG std.
    """

    magnitude = np.linalg.norm(acc.astype(np.float64), axis=2)
    magnitude -= magnitude.mean(axis=1, keepdims=True)
    t_acc = _time(acc.shape[1], ACC_RATE)
    t_ppg = _time(ppg.shape[1], PPG_RATE)
    resampled = np.stack([np.interp(t_ppg, t_acc, row) for row in magnitude])
    unit = resampled / _window_std(resampled)
    return ppg + level * _window_std(ppg) * unit, acc.copy()


def periodic_motion_artifact(ppg, acc, level, rng, frequency_hz=2.0):
    """Sinusoidal artefact at a step-like cadence (default 2 Hz = 120/min)."""

    phase = rng.uniform(0, 2 * np.pi, size=(ppg.shape[0], 1))
    t = _time(ppg.shape[1], PPG_RATE)[None, :]
    artifact = np.sin(2 * np.pi * frequency_hz * t + phase)
    return ppg + level * _window_std(ppg) * np.sqrt(2) * artifact, acc.copy()


def baseline_wander(ppg, acc, level, rng, frequency_hz=0.2):
    """Slow drift (e.g. sensor pressure change), amplitude level x PPG std."""

    phase = rng.uniform(0, 2 * np.pi, size=(ppg.shape[0], 1))
    t = _time(ppg.shape[1], PPG_RATE)[None, :]
    return ppg + level * _window_std(ppg) * np.sin(2 * np.pi * frequency_hz * t + phase), acc.copy()


def amplitude_attenuation(ppg, acc, level, rng):
    """Weak pulse signal: scale the mean-removed PPG by gain = level."""

    mean = ppg.mean(axis=1, keepdims=True)
    return mean + (ppg - mean) * level, acc.copy()


def signal_dropout(ppg, acc, level, rng):
    """Contact loss: replace a random contiguous `level`-second segment with the window mean."""

    out = ppg.astype(np.float64).copy()
    length = int(round(level * PPG_RATE))
    starts = rng.integers(0, ppg.shape[1] - length + 1, size=ppg.shape[0])
    means = out.mean(axis=1)
    for i, start in enumerate(starts):
        out[i, start:start + length] = means[i]
    return out, acc.copy()


def clipping(ppg, acc, level, rng):
    """Saturation: clip each window to its [level, 1 - level] quantile range."""

    lo = np.quantile(ppg, level, axis=1, keepdims=True)
    hi = np.quantile(ppg, 1 - level, axis=1, keepdims=True)
    return np.clip(ppg, lo, hi), acc.copy()


PERTURBATIONS = {
    "gaussian_noise": gaussian_noise,
    "accelerometer_coupled_artifact": accelerometer_coupled_artifact,
    "periodic_motion_artifact": periodic_motion_artifact,
    "baseline_wander": baseline_wander,
    "amplitude_attenuation": amplitude_attenuation,
    "signal_dropout": signal_dropout,
    "clipping": clipping,
}


def apply(name, ppg, acc, level, seed):
    """Apply a named perturbation with a fixed seed. Returns float32 copies."""

    if name == "clean":
        return ppg.astype(np.float32).copy(), acc.astype(np.float32).copy()
    rng = np.random.default_rng(seed)
    new_ppg, new_acc = PERTURBATIONS[name](ppg.astype(np.float64), acc.astype(np.float64), level, rng)
    new_ppg = np.asarray(new_ppg, dtype=np.float32)
    new_acc = np.asarray(new_acc, dtype=np.float32)
    if not (np.isfinite(new_ppg).all() and np.isfinite(new_acc).all()):
        raise RuntimeError(f"{name} produced nonfinite samples")
    return new_ppg, new_acc


def motion_intensity(acc):
    """Std of accelerometer magnitude per window (natural motion level, in g)."""

    return np.linalg.norm(np.asarray(acc, dtype=np.float64), axis=2).std(axis=1)

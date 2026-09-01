"""Prétraitement audio pour la vérification du locuteur : mesure d'énergie et
découpe des portions non-voisées.

ECAPA produit un embedding beaucoup plus stable si on ne lui donne que la parole :
les silences et le bruit de fond entre les mots diluent le vecteur et font chuter
la cohérence d'enrôlement comme les scores runtime.
"""

from __future__ import annotations

import numpy as np

_INT16_FULL_SCALE = 32768.0


def to_float(pcm: bytes) -> np.ndarray:
    """PCM 16-bit mono little-endian -> float32 dans [-1, 1]."""
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / _INT16_FULL_SCALE


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


def frame_rms(samples: np.ndarray, sample_rate: int, frame_ms: float = 30.0) -> np.ndarray:
    """RMS par trame (fenêtre glissante non chevauchante)."""
    step = max(1, int(sample_rate * frame_ms / 1000))
    n = samples.size // step
    if n == 0:
        return np.array([rms(samples)], dtype=np.float32)
    trimmed = samples[: n * step].reshape(n, step)
    return np.sqrt(np.mean(trimmed.astype(np.float32) ** 2, axis=1))


def keep_voiced(
    samples: np.ndarray,
    sample_rate: int,
    *,
    threshold: float,
    frame_ms: float = 30.0,
    pad_ms: float = 120.0,
) -> np.ndarray:
    """Ne garde que les trames dont le RMS dépasse `threshold`, avec une marge
    `pad_ms` de part et d'autre de chaque région voisée. Renvoie un tableau vide
    si rien ne dépasse le seuil."""
    if samples.size == 0:
        return samples

    step = max(1, int(sample_rate * frame_ms / 1000))
    energies = frame_rms(samples, sample_rate, frame_ms)
    voiced = energies > threshold
    if not voiced.any():
        return np.array([], dtype=np.float32)

    pad_frames = max(1, int(pad_ms / frame_ms))
    mask = voiced.copy()
    (idx,) = np.where(voiced)
    for i in idx:
        lo = max(0, i - pad_frames)
        hi = min(len(mask), i + pad_frames + 1)
        mask[lo:hi] = True

    sample_mask = np.repeat(mask, step)
    if sample_mask.size < samples.size:
        sample_mask = np.concatenate(
            [sample_mask, np.full(samples.size - sample_mask.size, mask[-1])]
        )
    return samples[: sample_mask.size][sample_mask[: samples.size]]

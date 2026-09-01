"""LOT 1.5 — logique de gating du SpeakerVerificationGate (sans modèle réel).

Vérifie que, selon le score renvoyé par le profil, le gate :
- accepté  : relâche le segment complet (start -> preroll -> audio -> stop) + événement RTVI
- rejeté   : ne laisse RIEN passer sauf l'événement RTVI (le STT ne voit jamais la voix tierce)
- douteux  : idem rejeté côté frames, avec status=uncertain ; aussi le cas d'un
  segment trop court avec un score bas (jamais rejeté sec)

Le test avec le vrai modèle ECAPA (discrimination de deux voix) est manuel :
`tests/manual_speaker_ecapa.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipecat.frames.frames import (
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.tests.utils import run_test

from front.speaker.config import SpeakerConfig
from front.speaker.verification import SpeakerVerificationGate

_CONFIG = SpeakerConfig(
    profile_path="/dev/null",
    accept_threshold=0.45,
    reject_threshold=0.30,
    enroll_phrases=3,
    enroll_seconds=4.0,
    min_confident_seconds=0.5,
)


class _FakeEmbedder:
    def embed_float(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        return np.zeros(192, dtype=np.float32)


class _FakeProfile:
    n_phrases = 3

    def __init__(self, score: float) -> None:
        self._score = score

    def similarity(self, embedding: np.ndarray) -> float:
        return self._score

    def self_consistency(self) -> float:
        return 1.0


def _gate(score: float) -> SpeakerVerificationGate:
    gate = SpeakerVerificationGate(_FakeEmbedder(), _CONFIG)
    gate._profile = _FakeProfile(score)
    return gate


def _segment(seconds: float = 1.0) -> list:
    n = int(16000 * seconds)
    audio = (np.random.default_rng(0).normal(0, 3000, n)).astype(np.int16).tobytes()
    return [
        VADUserStartedSpeakingFrame(),
        InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
        VADUserStoppedSpeakingFrame(),
    ]


async def _run(score: float, seconds: float = 1.0):
    down, _ = await run_test(
        _gate(score), frames_to_send=_segment(seconds), expected_down_frames=None
    )
    return list(down)


@pytest.mark.asyncio
async def test_accepted_releases_full_segment():
    down = await _run(0.80)
    kinds = [type(f).__name__ for f in down]
    assert "VADUserStartedSpeakingFrame" in kinds
    assert "InputAudioRawFrame" in kinds
    assert "VADUserStoppedSpeakingFrame" in kinds
    msg = next(f for f in down if isinstance(f, RTVIServerMessageFrame))
    assert msg.data["status"] == "accepted"


@pytest.mark.asyncio
async def test_rejected_blocks_everything_but_event():
    down = await _run(0.10)
    assert [type(f).__name__ for f in down] == ["RTVIServerMessageFrame"]
    assert down[0].data["status"] == "rejected"
    assert down[0].data["type"] == "speaker_verification"


@pytest.mark.asyncio
async def test_uncertain_emits_signal_without_forwarding_segment():
    down = await _run(0.38)
    assert [type(f).__name__ for f in down] == ["RTVIServerMessageFrame"]
    assert down[0].data["status"] == "uncertain"


@pytest.mark.asyncio
async def test_short_low_score_is_uncertain_not_rejected():
    # 0.3s de voix, score très bas : pas fiable -> douteux, jamais rejeté sec
    down = await _run(0.05, seconds=0.35)
    assert [type(f).__name__ for f in down] == ["RTVIServerMessageFrame"]
    assert down[0].data["status"] == "uncertain"
    assert down[0].data["short_segment"] is True


def test_keep_voiced_trims_silence():
    from front.speaker import audio

    sr = 16000
    rng = np.random.default_rng(0)
    silence = np.zeros(sr, dtype=np.float32)
    speech = rng.normal(0, 0.2, sr).astype(np.float32)
    trimmed = audio.keep_voiced(
        np.concatenate([silence, speech, silence]), sr, threshold=0.02
    )
    assert sr * 0.8 < trimmed.size < sr * 1.6


def test_from_embeddings_drops_single_outlier():
    from front.speaker.profile import SpeakerProfile

    base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tilted = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    tilted /= np.linalg.norm(tilted)
    outlier = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    profile = SpeakerProfile.from_embeddings(
        [base, base, tilted, tilted, outlier], sample_rate=16000
    )
    assert profile.n_phrases == 4
    assert profile.centroid @ base > 0.98

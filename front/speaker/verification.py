"""SpeakerVerificationGate — entre le VAD et le STT dans la cascade d'écoute.

Bufferise le segment de parole délimité par le VAD, calcule son embedding ECAPA
(CPU) au `VADUserStoppedSpeakingFrame`, le compare au profil enrôlé, et **gate**
le segment :

- **accepté** (score >= accept_threshold) : le segment complet (start -> audio ->
  stop) est relâché vers le STT, qui le transcrit normalement.
- **rejeté** (score < reject_threshold) : le segment est jeté. Le STT ne reçoit
  ni audio ni frame de fin -> `run_stt` (GPU) n'est jamais appelé.
- **douteux** (entre les deux) : événement RTVI `speaker_verification`
  status=uncertain + log WARNING. Le segment n'est pas transmis. La confirmation
  vocale qui consomme ce signal est construite au LOT 2 (décision utilisateur
  2026-09-01) — ni rejet muet, ni acceptation automatique.

Le start/stop VAD et l'audio du segment ne sont **jamais** poussés vers l'aval
tant que la décision n'est pas prise : le STT ne voit donc jamais de voix tierce
et son buffer interne ne se pollue pas.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

from front.speaker.config import SpeakerConfig
from front.speaker.embedding import SpeakerEmbedder
from front.speaker.profile import SpeakerProfile

_MIN_SEGMENT_SECONDS = 0.3
_PREROLL_SECONDS = 0.32  # audio conservé avant le déclenchement VAD (latence de détection)


class SpeakerVerificationGate(FrameProcessor):
    """Filtre la cascade d'écoute : seule la voix de l'utilisateur enrôlé atteint le STT."""

    def __init__(
        self, embedder: SpeakerEmbedder, config: SpeakerConfig, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._embedder = embedder
        self._config = config
        self._profile: SpeakerProfile | None = None

        self._bytes_per_second = config.sample_rate * 2  # 16-bit mono
        self._preroll_max = int(self._bytes_per_second * _PREROLL_SECONDS)
        self._preroll = bytearray()

        self._capturing = False
        self._segment: list[AudioRawFrame] = []
        self._started_frame: VADUserStartedSpeakingFrame | None = None

    def load(self) -> None:
        """Warm-start : charge le profil enrôlé (échoue fort s'il est absent)."""
        self._profile = SpeakerProfile.load(self._config.profile_path)
        logger.info(
            "SpeakerVerificationGate: profil chargé "
            f"({self._config.profile_path}, {self._profile.n_phrases} phrases, "
            f"cohérence interne {self._profile.self_consistency():.3f}) — "
            f"seuils accept={self._config.accept_threshold} reject={self._config.reject_threshold}"
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # Audio descendant : bufferisé pendant la capture, sinon laissé passer (+ pré-roll).
        if isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._capturing:
                self._segment.append(frame)
            else:
                self._preroll += frame.audio
                if len(self._preroll) > self._preroll_max:
                    del self._preroll[: -self._preroll_max]
                await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._begin_capture(frame)
            return  # retenu jusqu'à la décision

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._capturing:
                await self._finish_capture(frame, direction)
            else:
                await self.push_frame(frame, direction)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            self._reset()

        await self.push_frame(frame, direction)

    def _begin_capture(self, frame: VADUserStartedSpeakingFrame) -> None:
        self._capturing = True
        self._started_frame = frame
        self._segment = []
        if self._preroll:
            self._segment.append(
                InputAudioRawFrame(
                    audio=bytes(self._preroll),
                    sample_rate=self._config.sample_rate,
                    num_channels=1,
                )
            )
        self._preroll.clear()

    async def _finish_capture(
        self, stop_frame: VADUserStoppedSpeakingFrame, direction: FrameDirection
    ) -> None:
        frames = self._segment
        started = self._started_frame
        self._capturing = False
        self._segment = []
        self._started_frame = None

        pcm = b"".join(bytes(f.audio) for f in frames)
        duration = len(pcm) / self._bytes_per_second
        if duration < _MIN_SEGMENT_SECONDS:
            logger.debug(
                f"SpeakerVerificationGate: segment {duration:.2f}s < "
                f"{_MIN_SEGMENT_SECONDS}s — ignoré (pas de vérification)"
            )
            return

        assert self._profile is not None, "gate non chargé (load())"
        embedding = await asyncio.to_thread(
            self._embedder.embed_pcm16, pcm, self._config.sample_rate
        )
        score = self._profile.similarity(embedding)

        if score >= self._config.accept_threshold:
            logger.info(
                f"SpeakerVerificationGate: ACCEPTÉ score={score:.3f} "
                f"({duration:.1f}s) — segment relâché vers le STT"
            )
            await self._emit("accepted", score)
            if started is not None:
                await self.push_frame(started, direction)
            for audio_frame in frames:
                await self.push_frame(audio_frame, direction)
            await self.push_frame(stop_frame, direction)
            return

        if score < self._config.reject_threshold:
            logger.info(
                f"SpeakerVerificationGate: REJETÉ score={score:.3f} "
                f"({duration:.1f}s) — voix non reconnue, STT non déclenché"
            )
            await self._emit("rejected", score)
            return

        logger.warning(
            f"SpeakerVerificationGate: DOUTEUX score={score:.3f} "
            f"(∈ [{self._config.reject_threshold}, {self._config.accept_threshold}[) — "
            f"confirmation requise (matérialisation vocale : LOT 2), segment non transmis"
        )
        await self._emit("uncertain", score)

    async def _emit(self, status: str, score: float) -> None:
        await self.push_frame(
            RTVIServerMessageFrame(
                data={
                    "type": "speaker_verification",
                    "status": status,
                    "score": round(float(score), 4),
                    "accept_threshold": self._config.accept_threshold,
                    "reject_threshold": self._config.reject_threshold,
                }
            ),
            FrameDirection.DOWNSTREAM,
        )

    def _reset(self) -> None:
        self._capturing = False
        self._segment = []
        self._started_frame = None
        self._preroll.clear()

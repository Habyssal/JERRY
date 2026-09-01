"""SpeakerVerificationGate — entre le VAD et le STT dans la cascade d'écoute.

Bufferise le segment de parole délimité par le VAD, calcule son embedding ECAPA
(CPU) au `VADUserStoppedSpeakingFrame`, le compare au profil enrôlé, et **gate**
le segment :

- **accepté** (score >= accept_threshold) : le segment complet (start -> audio ->
  stop) est relâché vers le STT, qui le transcrit normalement.
- **rejeté** (score < reject_threshold, segment assez long) : le segment est jeté.
  Le STT ne reçoit ni audio ni frame de fin -> `run_stt` (GPU) jamais appelé.
- **douteux** : entre les deux seuils, OU segment trop court pour une décision
  fiable (`min_confident_seconds`) avec un score bas. Événement RTVI
  `speaker_verification` status=uncertain + log WARNING, segment non transmis. La
  confirmation vocale qui consomme ce signal est construite au LOT 2 (décision
  utilisateur 2026-09-01) — ni rejet muet, ni acceptation automatique.

Détails de calcul du score (importants pour la fiabilité) :
- le **pré-roll** (audio d'avant le déclenchement VAD) est rejoué au STT pour le
  contexte d'attaque, mais **exclu** du calcul de l'embedding (il dilue le score).
- les portions non-voisées du segment sont découpées, avec un seuil d'énergie
  adapté au **bruit de fond mesuré en continu**.
- un segment court (< `min_confident_seconds` de voix nette) ne peut pas être
  rejeté sec : ECAPA n'est pas fiable en dessous de ~1-2 s.

Le STT ne reçoit **que** les segments explicitement relâchés : en dehors de ça,
aucun frame audio n'est poussé vers l'aval (sinon son buffer glissant interne
recouvre le pré-roll rejoué et il transcrit deux fois le début de phrase).
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

from front.speaker import audio as _audio
from front.speaker.config import SpeakerConfig
from front.speaker.embedding import SpeakerEmbedder
from front.speaker.profile import SpeakerProfile

_MIN_SEGMENT_SECONDS = 0.3
_PREROLL_SECONDS = 0.32  # audio conservé avant le déclenchement VAD (latence de détection)
_MIN_VOICED_FOR_TRIM = 0.4  # en dessous, on garde le segment brut plutôt que la découpe


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
        self._noise_rms = 0.02  # plancher de bruit, adapté en continu hors parole

        self._capturing = False
        self._segment: list[AudioRawFrame] = []
        self._preroll_frame: InputAudioRawFrame | None = None
        self._started_frame: VADUserStartedSpeakingFrame | None = None

    def load(self) -> None:
        """Warm-start : charge le profil enrôlé (échoue fort s'il est absent)."""
        self._profile = SpeakerProfile.load(self._config.profile_path)
        logger.info(
            "SpeakerVerificationGate: profil chargé "
            f"({self._config.profile_path}, {self._profile.n_phrases} phrases, "
            f"cohérence {self._profile.self_consistency():.3f}) — seuils "
            f"accept={self._config.accept_threshold} reject={self._config.reject_threshold} "
            f"min_confident={self._config.min_confident_seconds}s"
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._capturing:
                self._segment.append(frame)
            else:
                self._preroll += frame.audio
                if len(self._preroll) > self._preroll_max:
                    del self._preroll[: -self._preroll_max]
                self._update_noise(frame.audio)
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

    def _update_noise(self, audio: bytes) -> None:
        r = _audio.rms(_audio.to_float(audio))
        if r <= 0.0:
            return
        if r < self._noise_rms:
            self._noise_rms = r  # descente instantanée vers le plancher
        else:
            self._noise_rms += (r - self._noise_rms) * 0.0005  # remontée très lente
        self._noise_rms = min(max(self._noise_rms, 0.001), 0.05)

    def _begin_capture(self, frame: VADUserStartedSpeakingFrame) -> None:
        self._capturing = True
        self._started_frame = frame
        self._segment = []
        self._preroll_frame = (
            InputAudioRawFrame(
                audio=bytes(self._preroll),
                sample_rate=self._config.sample_rate,
                num_channels=1,
            )
            if self._preroll
            else None
        )
        self._preroll.clear()

    async def _finish_capture(
        self, stop_frame: VADUserStoppedSpeakingFrame, direction: FrameDirection
    ) -> None:
        preroll_frame = self._preroll_frame
        frames = self._segment
        started = self._started_frame
        self._capturing = False
        self._segment = []
        self._preroll_frame = None
        self._started_frame = None

        seg_pcm = b"".join(bytes(f.audio) for f in frames)
        if len(seg_pcm) / self._bytes_per_second < _MIN_SEGMENT_SECONDS:
            logger.debug("SpeakerVerificationGate: segment trop court, ignoré")
            return

        assert self._profile is not None, "gate non chargé (load())"
        samples = _audio.to_float(seg_pcm)
        threshold = max(0.012, self._noise_rms * 2.0)
        voiced = _audio.keep_voiced(samples, self._config.sample_rate, threshold=threshold)
        voiced_s = voiced.size / self._config.sample_rate
        scored = voiced if voiced_s >= _MIN_VOICED_FOR_TRIM else samples

        embedding = await asyncio.to_thread(
            self._embedder.embed_float, scored, self._config.sample_rate
        )
        score = self._profile.similarity(embedding)
        short = voiced_s < self._config.min_confident_seconds

        if score >= self._config.accept_threshold:
            status = "accepted"
        elif score < self._config.reject_threshold and not short:
            status = "rejected"
        else:
            status = "uncertain"

        tag = " [court]" if short else ""
        detail = f"score={score:.3f} voix={voiced_s:.1f}s{tag} bruit={self._noise_rms:.3f}"
        if status == "accepted":
            logger.info(f"SpeakerVerificationGate: ACCEPTÉ {detail} — relâché vers le STT")
        elif status == "rejected":
            logger.info(f"SpeakerVerificationGate: REJETÉ {detail} — STT non déclenché")
        else:
            logger.warning(
                f"SpeakerVerificationGate: DOUTEUX {detail} — confirmation requise "
                f"(matérialisation vocale : LOT 2), segment non transmis"
            )

        await self._emit(status, score, voiced_s, short)

        if status == "accepted":
            if started is not None:
                await self.push_frame(started, direction)
            if preroll_frame is not None:
                await self.push_frame(preroll_frame, direction)
            for audio_frame in frames:
                await self.push_frame(audio_frame, direction)
            await self.push_frame(stop_frame, direction)

    async def _emit(self, status: str, score: float, voiced_seconds: float, short: bool) -> None:
        await self.push_frame(
            RTVIServerMessageFrame(
                data={
                    "type": "speaker_verification",
                    "status": status,
                    "score": round(float(score), 4),
                    "voiced_seconds": round(float(voiced_seconds), 2),
                    "short_segment": bool(short),
                    "accept_threshold": self._config.accept_threshold,
                    "reject_threshold": self._config.reject_threshold,
                }
            ),
            FrameDirection.DOWNSTREAM,
        )

    def _reset(self) -> None:
        self._capturing = False
        self._segment = []
        self._preroll_frame = None
        self._started_frame = None
        self._preroll.clear()

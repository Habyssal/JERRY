"""Réponse triviale (écho) pour le LOT 1 : pas de LLM, juste TranscriptionFrame -> TextFrame.

EchoResponder joue le rôle que tiendrait normalement un LLM du point de vue de
KokoroTTSService : celui-ci n'aggrège et ne synthétise le texte en attente qu'à
réception d'une frame de fin de tour (LLMFullResponseEndFrame/EndFrame), donc on
émet la paire LLMFullResponseStartFrame/EndFrame autour du texte à vocaliser.
Avant ça, on bascule la langue/voix TTS sur celle détectée par le STT (FR/EN).

TTFA (time-to-first-audio) mesuré entre deux processors car TTSAudioRawFrame
n'existe qu'en aval du service TTS : EchoResponder (avant TTS) horodate la
transcription, TTFALogger (après TTS) calcule l'écart à la première frame audio.
"""

import time

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from front.services.tts_kokoro import KokoroTTSServiceFrEn


class EchoResponder(FrameProcessor):
    """Convertit une transcription utilisateur en texte à vocaliser (écho), sans LLM."""

    def __init__(self, tts: KokoroTTSServiceFrEn, **kwargs):
        super().__init__(**kwargs)
        self._tts = tts
        self.transcription_at: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self.transcription_at = time.monotonic()
            logger.info(f"EchoResponder: écho de « {frame.text} » ({frame.language})")
            if frame.language is not None:
                self._tts.set_language(frame.language)
            await self.push_frame(LLMFullResponseStartFrame(), direction)
            await self.push_frame(TextFrame(frame.text), direction)
            await self.push_frame(LLMFullResponseEndFrame(), direction)
            return

        await self.push_frame(frame, direction)


class TTFALogger(FrameProcessor):
    """Placé après le TTS : logue le TTFA à la première frame audio de chaque réponse."""

    def __init__(self, echo: EchoResponder, **kwargs):
        super().__init__(**kwargs)
        self._echo = echo

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSAudioRawFrame) and self._echo.transcription_at is not None:
            ttfa_ms = (time.monotonic() - self._echo.transcription_at) * 1000
            logger.info(f"TTFALogger: TTFA = {ttfa_ms:.0f} ms")
            self._echo.transcription_at = None

        await self.push_frame(frame, direction)

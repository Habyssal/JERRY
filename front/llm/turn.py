"""Adaptateur de tour de parole : WakeWordGate → LLM front (LOT 2a).

`WakeWordGate` (LOT 1.5) fait déjà l'agrégation de tour : il émet **une seule**
`TranscriptionFrame` par tour adressé via « JOSS ». On la convertit ici en tour
utilisateur pour le couple d'agrégateurs `LLMContext` :
`LLMMessagesAppendFrame(messages=[{role:"user", ...}], run_llm=True)` — ce qui
déclenche la complétion sans dépendre de la détection de fin de tour VAD de
l'agrégateur (le tour est déjà clos par `WakeWordGate`).

LOT 2b (à venir) : la compaction / normalisation des disfluences
(« raconte-moi… non plutôt une blague ») se branchera dans `_prepare_text()`.

La langue TTS est mutée ici, à partir de la langue détectée côté STT (comme le
faisait `EchoResponder` au LOT 1) — le prompt système demande au LLM de répondre
dans cette même langue.
"""

from __future__ import annotations

import time

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMMessagesAppendFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from front.services.tts_kokoro import KokoroTTSServiceFrEn


class LLMTurnAdapter(FrameProcessor):
    """Transforme le tour agrégé par WakeWordGate en tour utilisateur LLM."""

    def __init__(self, tts: KokoroTTSServiceFrEn, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tts = tts
        self.turn_at: float | None = None  # horodatage du dernier tour (TTFA)

    def _prepare_text(self, text: str) -> str:
        """Point d'insertion de la compaction LOT 2b. Pour l'instant : trim."""
        return text.strip()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            text = self._prepare_text(frame.text)
            if not text:
                return
            if frame.language is not None:
                self._tts.set_language(frame.language)
            self.turn_at = time.monotonic()
            logger.info(f"LLMTurnAdapter: tour utilisateur → LLM : « {text} »")
            await self.push_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": text}], run_llm=True
                ),
                direction,
            )
            return

        await self.push_frame(frame, direction)


class TTFALogger(FrameProcessor):
    """Placé après le TTS : logue le TTFA (tour utilisateur → 1ʳᵉ frame audio)."""

    def __init__(self, adapter: LLMTurnAdapter, **kwargs) -> None:
        super().__init__(**kwargs)
        self._adapter = adapter

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSAudioRawFrame) and self._adapter.turn_at is not None:
            ttfa_ms = (time.monotonic() - self._adapter.turn_at) * 1000
            logger.info(f"TTFALogger: TTFA = {ttfa_ms:.0f} ms")
            self._adapter.turn_at = None

        await self.push_frame(frame, direction)

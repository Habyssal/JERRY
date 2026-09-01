"""Service STT local pour Parakeet TDT (NeMo), branché sur SegmentedSTTService."""

import asyncio
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger
from lingua import Language as LinguaLanguage

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from front.langfilter import build_detector, detect_fr_or_en

DEFAULT_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"

_LINGUA_TO_PIPECAT = {
    LinguaLanguage.FRENCH: Language.FR,
    LinguaLanguage.ENGLISH: Language.EN,
}


class ParakeetSTTService(SegmentedSTTService):
    """STT local basé sur NeMo ASR (Parakeet TDT), chargé une fois au démarrage (warm-start)."""

    def __init__(self, *, model_name: str = DEFAULT_MODEL_NAME, **kwargs):
        super().__init__(
            sample_rate=16000,
            settings=STTSettings(model=model_name, language=None),
            **kwargs,
        )
        self._model_name = model_name
        self._model = None

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        return language.value if language else None

    def load(self):
        """Charge le modèle Parakeet et le détecteur de langue (warm-start)."""
        from nemo.collections.asr.models import ASRModel

        logger.info(f"ParakeetSTTService: chargement du modèle {self._model_name}...")
        self._model = ASRModel.from_pretrained(model_name=self._model_name)
        self._model.eval()
        build_detector()
        logger.info("ParakeetSTTService: modèle et détecteur de langue chargés.")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if self._model is None:
            yield ErrorFrame("Parakeet model not loaded")
            return

        await self.start_processing_metrics()

        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        results = await asyncio.to_thread(self._model.transcribe, [audio_float])

        await self.stop_processing_metrics()

        text = results[0].text if hasattr(results[0], "text") else str(results[0])
        text = text.strip()
        if not text:
            return

        lingua_language = detect_fr_or_en(text)
        if lingua_language is None:
            logger.info(f"ParakeetSTTService: rejeté (ni FR ni EN) : [{text}]")
            return

        language = _LINGUA_TO_PIPECAT[lingua_language]
        logger.debug(f"ParakeetSTTService: transcription ({language.value}): [{text}]")
        yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), language)

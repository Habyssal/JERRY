"""Cascade d'écoute : VAD -> STT -> mot de réveil « JOSS » -> LLM front -> TTS.

LOT 1   : boucle audio nue (écho).
LOT 1.5 : le front ne réagit qu'après le mot de réveil (`front/wakeword.py`).
LOT 2a  : LLM front (Ollama, Ministral 3 3B — bascule actée, xLAM tool-calling
          cassé via Ollama) branché après le mot de réveil, avec outils de
          démonstration pour valider le tool-calling. La conversation (historique
          + relances) est gérée par `front/llm/conversation.py` (`FrontConversation`),
          pas par `LLMContextAggregatorPair` : l'autorité de tour est
          `WakeWordGate`, pas le VAD. L'écho (`front/echo.py`) est remplacé.

La vérification du locuteur (`front/speaker/`) reste **débranchée** — futur
module passif d'identification des voix récurrentes (cf. Doc/Backlog.md).
"""

import asyncio
import os

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

from front.barge_in import BargeInController
from front.llm.context import build_context
from front.llm.conversation import FrontConversation, TTFALogger
from front.llm.probe_tools import register_probe_tools
from front.llm.service import build_llm_service
from front.services.stt_parakeet import ParakeetSTTService
from front.services.tts_kokoro import KokoroTTSServiceFrEn
from front.wakeword import DEFAULT_WAKE_WORD, WakeWordGate


def build_pipeline() -> tuple[Pipeline, ParakeetSTTService]:
    """Assemble le pipeline LOT 1 + 1.5 + 2a. Kokoro (TTS) charge son modèle dès
    sa construction ; Parakeet (STT) est chargé via stt.load() (warm-start
    explicite) ; le LLM est chargé côté Ollama à la première complétion."""
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        )
    )

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(sample_rate=16000))
    barge_in = BargeInController()
    stt = ParakeetSTTService()
    wake_gate = WakeWordGate(
        wake_word=os.environ.get("JERRY_WAKE_WORD", DEFAULT_WAKE_WORD).lower(),
        command_timeout_s=float(os.environ.get("JERRY_WAKE_TIMEOUT_S", "8")),
        aggregation_silence_s=float(os.environ.get("JERRY_WAKE_AGG_SILENCE_S", "1.2")),
    )
    tts = KokoroTTSServiceFrEn(
        settings=KokoroTTSServiceFrEn.Settings(voice="ff_siwis", language=Language.FR)
    )

    llm = build_llm_service()
    tools = register_probe_tools(llm)
    context = build_context(tools=tools)

    conversation = FrontConversation(context, tts)
    ttfa_logger = TTFALogger(conversation)

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            barge_in,
            stt,
            wake_gate,
            llm,
            conversation,
            tts,
            ttfa_logger,
            transport.output(),
        ]
    )
    return pipeline, stt


async def run():
    """Warm-start les modèles puis démarre la cascade d'écoute."""
    pipeline, stt = build_pipeline()

    logger.info("Warm-start: chargement du modèle Parakeet (STT)...")
    stt.load()

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            enable_metrics=True,
        ),
        observers=[MetricsLogObserver(), TranscriptionLogObserver()],
        idle_timeout_secs=None,  # écoute permanente : pas d'auto-arrêt après inactivité
    )

    runner = WorkerRunner()
    await runner.add_workers(worker)

    logger.info(
        "front ready — cascade d'écoute (LOT 1 + 1.5 + 2a) en écoute, "
        "mot de réveil « JOSS », LLM front branché. Ctrl+C pour arrêter."
    )
    await runner.run()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()

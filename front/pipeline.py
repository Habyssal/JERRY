"""LOT 1 + 1.5 — cascade d'écoute : VAD -> STT -> mot de réveil « JOSS » -> écho -> TTS.

LOT 1.5 : le front ne réagit qu'après le mot de réveil (`front/wakeword.py`).
La vérification du locuteur (`front/speaker/`) est **débranchée** — elle deviendra
un module passif d'identification des voix récurrentes (cf. Doc/Backlog.md).
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
from front.echo import EchoResponder, TTFALogger
from front.services.stt_parakeet import ParakeetSTTService
from front.services.tts_kokoro import KokoroTTSServiceFrEn
from front.wakeword import DEFAULT_WAKE_WORD, WakeWordGate


def build_pipeline() -> tuple[Pipeline, ParakeetSTTService]:
    """Assemble le pipeline LOT 1 + 1.5. Kokoro (TTS) charge son modèle dès sa
    construction ; Parakeet (STT) est chargé séparément via stt.load() pour un
    warm-start explicite."""
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
    )
    tts = KokoroTTSServiceFrEn(settings=KokoroTTSServiceFrEn.Settings(voice="ff_siwis", language=Language.FR))
    echo = EchoResponder(tts)
    ttfa_logger = TTFALogger(echo)

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            barge_in,
            stt,
            wake_gate,
            echo,
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
        "front ready — cascade d'écoute (LOT 1 + 1.5) en écoute, "
        "mot de réveil « JOSS ». Ctrl+C pour arrêter."
    )
    await runner.run()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()

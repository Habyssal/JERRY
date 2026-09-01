"""LOT 1 + 1.5 — cascade d'écoute : VAD -> speaker verification -> STT -> écho -> TTS."""

import asyncio

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
from front.speaker.config import SpeakerConfig
from front.speaker.embedding import SpeakerEmbedder
from front.speaker.verification import SpeakerVerificationGate


def build_pipeline() -> tuple[Pipeline, ParakeetSTTService, SpeakerVerificationGate, SpeakerEmbedder]:
    """Assemble le pipeline LOT 1 + 1.5. Kokoro (TTS) charge son modèle dès sa
    construction ; Parakeet (STT) et l'embedder ECAPA sont chargés séparément
    via load() pour un warm-start explicite."""
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        )
    )

    speaker_config = SpeakerConfig.from_env()

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(sample_rate=16000))
    barge_in = BargeInController()
    embedder = SpeakerEmbedder(device="cpu")
    speaker_gate = SpeakerVerificationGate(embedder, speaker_config)
    stt = ParakeetSTTService()
    tts = KokoroTTSServiceFrEn(settings=KokoroTTSServiceFrEn.Settings(voice="ff_siwis", language=Language.FR))
    echo = EchoResponder(tts)
    ttfa_logger = TTFALogger(echo)

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            barge_in,
            speaker_gate,
            stt,
            echo,
            tts,
            ttfa_logger,
            transport.output(),
        ]
    )
    return pipeline, stt, speaker_gate, embedder


async def run():
    """Warm-start les modèles puis démarre la cascade d'écoute."""
    pipeline, stt, speaker_gate, embedder = build_pipeline()

    speaker_gate.load()  # lit le profil enrôlé — échoue vite s'il manque, avant tout téléchargement

    logger.info("Warm-start: chargement de l'embedder locuteur ECAPA-TDNN (CPU)...")
    embedder.load()

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

    logger.info("front ready — cascade d'écoute (LOT 1 + 1.5) en écoute. Ctrl+C pour arrêter.")
    await runner.run()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()

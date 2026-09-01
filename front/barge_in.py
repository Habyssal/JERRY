"""Barge-in minimal, sans LLM.

Dans pipecat 1.3.0, l'interruption automatique est normalement portée par le
LLMUserAggregator (context aggregator), absent ici puisque le LOT 1 n'a pas de
LLM. On la reconstruit à la main : si l'utilisateur recommence à parler
(VADUserStartedSpeakingFrame) pendant que le bot parle (entre
BotStartedSpeakingFrame et BotStoppedSpeakingFrame, qui remontent en amont
depuis le transport de sortie), on émet une InterruptionFrame en aval —
TTSService et le transport de sortie l'interceptent tous les deux pour couper
la synthèse et la lecture en cours.
"""

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class BargeInController(FrameProcessor):
    """Placé après le VAD, avant le STT : coupe le bot si l'utilisateur reprend la parole."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bot_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, VADUserStartedSpeakingFrame) and self._bot_speaking:
            logger.info("BargeInController: interruption — l'utilisateur reprend la parole")
            await self.push_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

        await self.push_frame(frame, direction)

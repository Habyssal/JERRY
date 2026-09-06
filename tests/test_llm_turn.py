"""LOT 2a — LLMTurnAdapter : le tour agrégé par WakeWordGate devient un tour LLM.

Tests hors-ligne (pas d'Ollama). Le flux complet avec le vrai LLM est couvert
par `tests/test_llm_pipeline.py` (skippé si Ollama absent).
"""

from __future__ import annotations

import pytest

from pipecat.frames.frames import LLMMessagesAppendFrame, TextFrame, TranscriptionFrame
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language

from front.llm.turn import LLMTurnAdapter


class _FakeTTS:
    def __init__(self) -> None:
        self.language = None

    def set_language(self, language) -> None:
        self.language = language


def _tx(text: str, language=Language.FR) -> TranscriptionFrame:
    return TranscriptionFrame(text, "", "2026-09-06T00:00:00Z", language)


async def _run(frames):
    tts = _FakeTTS()
    down, _ = await run_test(
        LLMTurnAdapter(tts), frames_to_send=frames, expected_down_frames=None
    )
    return tts, down


@pytest.mark.asyncio
async def test_transcription_becomes_user_turn_with_run_llm():
    tts, down = await _run([_tx("quelle heure est-il ?")])
    appends = [f for f in down if isinstance(f, LLMMessagesAppendFrame)]
    assert len(appends) == 1
    assert appends[0].run_llm is True
    assert appends[0].messages == [{"role": "user", "content": "quelle heure est-il ?"}]
    # aucune TranscriptionFrame ne doit passer en aval (consommée ici)
    assert not any(isinstance(f, TranscriptionFrame) for f in down)


@pytest.mark.asyncio
async def test_tts_language_is_muted_from_transcription():
    tts, _ = await _run([_tx("what time is it?", Language.EN)])
    assert tts.language == Language.EN


@pytest.mark.asyncio
async def test_non_transcription_frames_pass_through():
    _, down = await _run([TextFrame("coucou")])
    assert any(isinstance(f, TextFrame) and f.text == "coucou" for f in down)


@pytest.mark.asyncio
async def test_blank_transcription_is_dropped():
    _, down = await _run([_tx("   ")])
    assert not any(isinstance(f, LLMMessagesAppendFrame) for f in down)

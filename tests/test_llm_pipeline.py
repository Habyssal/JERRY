"""LOT 2a — flux complet mot de réveil → LLM front (Ollama) → réponse.

Hits Ollama en local (Ministral 3 3B). Skippé si l'endpoint est injoignable,
pour que `pytest -q` reste vert sur une machine sans Ollama.

Valide le critère du LOT 2 :
- une question directe reçoit une réponse (message assistant non vide) ;
- une demande couverte par un outil déclenche un appel de fonction
  (message `role="tool"` injecté) — tool-calling validé via Ollama.
"""

from __future__ import annotations

import socket
import time

import pytest

from pipecat.frames.frames import ErrorFrame, Frame, LLMTextFrame, TranscriptionFrame
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language

from front.llm.context import build_context
from front.llm.context_guard import ContextAlternationGuard
from front.llm.probe_tools import register_probe_tools
from front.llm.service import DEFAULT_BASE_URL, build_llm_service
from front.llm.turn import LLMTurnAdapter


def _ollama_up() -> bool:
    host = DEFAULT_BASE_URL.split("//")[1].split("/")[0]
    name, _, port = host.partition(":")
    try:
        with socket.create_connection((name, int(port or 11434)), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _ollama_up(), reason="Ollama injoignable")


class _FakeTTS:
    def set_language(self, language) -> None:
        pass


class _FirstTokenClock(FrameProcessor):
    """Horodate la 1ʳᵉ frame de texte LLM (proxy du temps de réponse perçu)."""

    def __init__(self) -> None:
        super().__init__()
        self.first_text_at: float | None = None
        self._start = time.monotonic()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame) and self.first_text_at is None:
            self.first_text_at = time.monotonic() - self._start
        await self.push_frame(frame, direction)


def _tx(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text, "", "2026-09-06T00:00:00Z", Language.FR)


async def _run(frames):
    llm = build_llm_service()
    tools = register_probe_tools(llm)
    context = build_context(tools=tools)
    aggregators = LLMContextAggregatorPair(context)
    clock = _FirstTokenClock()
    pipeline = Pipeline(
        [
            LLMTurnAdapter(_FakeTTS()),
            aggregators.user(),
            ContextAlternationGuard(),
            llm,
            clock,
            aggregators.assistant(),
        ]
    )
    down, _ = await run_test(pipeline, frames_to_send=frames, expected_down_frames=None)
    messages = context.get_messages()
    assistant = [m for m in messages if m.get("role") == "assistant"]
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    errors = [f for f in down if isinstance(f, ErrorFrame)]
    last_reply = assistant[-1].get("content", "") if assistant else ""
    return last_reply, tool_msgs, errors, clock.first_text_at


async def _ask(text: str):
    reply, tool_msgs, errors, ttft = await _run([_tx(text), SleepFrame(12.0)])
    assert not errors, f"ErrorFrame(s): {[f.error for f in errors]}"
    return reply, tool_msgs, ttft


@pytest.mark.asyncio
async def test_direct_question_gets_a_spoken_reply():
    reply, _, ttft = await _ask("dis-moi bonjour en une phrase")
    print(f"\n[LLM] réponse={reply!r}  1er token à {ttft and ttft * 1000:.0f}ms")
    assert reply and reply.strip()


@pytest.mark.asyncio
async def test_time_question_triggers_tool_call():
    reply, tool_msgs, ttft = await _ask("quelle heure est-il ?")
    print(f"\n[LLM] réponse={reply!r}  outils={len(tool_msgs)}  1er token à {ttft and ttft * 1000:.0f}ms")
    assert tool_msgs, "aucun appel d'outil (message role=tool absent)"
    assert reply and reply.strip()


@pytest.mark.asyncio
async def test_rapid_turns_do_not_break_the_context():
    # reproduit le bug live : l'utilisateur enchaîne plusieurs tours (barge-in)
    # sans laisser JOSS répondre, dont un tour avec appel d'outil interrompu.
    # Sans ContextAlternationGuard : cascade de 500 (roles must alternate).
    reply, _, errors, _ = await _run(
        [
            _tx("quelle heure est-il ?"),
            SleepFrame(0.3),
            _tx("mets un minuteur de cinq minutes"),
            SleepFrame(0.3),
            _tx("dis-moi bonjour"),
            SleepFrame(0.3),
            _tx("raconte une blague"),
            SleepFrame(15.0),
        ]
    )
    print(f"\n[LLM] réponse finale={reply!r}  erreurs={[f.error for f in errors]}")
    assert not errors, "le contexte a été corrompu (500 roles must alternate)"
    assert reply and reply.strip()

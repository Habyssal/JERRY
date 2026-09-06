"""LOT 1.5 — WakeWordGate : filtre les transcriptions selon le mot de réveil « JOSS »."""

from __future__ import annotations

import pytest

from pipecat.frames.frames import TranscriptionFrame
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language

from front.wakeword import WakeWordGate


def _tx(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text, "", "2026-09-01T00:00:00Z", Language.FR)


async def _run(texts, **kwargs):
    down, _ = await run_test(
        WakeWordGate(**kwargs),
        frames_to_send=[_tx(t) for t in texts],
        expected_down_frames=None,
    )
    tx = [f.text for f in down if isinstance(f, TranscriptionFrame)]
    events = [f.data for f in down if isinstance(f, RTVIServerMessageFrame)]
    return tx, events


@pytest.mark.asyncio
async def test_wake_word_prefix_forwards_command_only():
    tx, events = await _run(["JOSS, quelle heure est-il ?"])
    assert tx == ["quelle heure est-il ?"]
    assert events[-1]["status"] == "command"


@pytest.mark.asyncio
async def test_no_wake_word_is_ignored():
    tx, events = await _run(["bonjour tout le monde, ça va ?"])
    assert tx == []
    assert events[-1]["status"] == "ignored"


@pytest.mark.asyncio
async def test_wake_word_alone_then_next_utterance_is_command():
    tx, events = await _run(["Joss.", "allume la lumière du salon"])
    assert tx == ["allume la lumière du salon"]
    assert [e["status"] for e in events] == ["awake", "command"]


@pytest.mark.asyncio
async def test_fuzzy_variant_is_accepted():
    tx, _ = await _run(["Josse raconte-moi une blague"])
    assert tx == ["raconte-moi une blague"]


@pytest.mark.asyncio
async def test_command_window_expires():
    times = [0.0, 100.0]  # 2e transcription bien après le timeout
    clock = lambda: times.pop(0) if len(times) > 1 else times[0]  # noqa: E731
    tx, events = await _run(
        ["JOSS", "allume la lumière"], command_timeout_s=8.0, time_fn=clock
    )
    assert tx == []  # la 2e n'est plus dans la fenêtre -> exige un nouveau réveil
    assert [e["status"] for e in events] == ["awake", "ignored"]


@pytest.mark.asyncio
async def test_command_continues_across_pause():
    # « JOSS raconte » <pause, le VAD coupe> « une histoire courte »
    tx, events = await _run(["JOSS raconte", "une histoire courte"])
    assert tx == ["raconte", "une histoire courte"]
    assert [e["status"] for e in events] == ["command", "command"]


@pytest.mark.asyncio
async def test_each_chunk_extends_the_window():
    times = [0.0, 5.0, 10.0]  # timeout 8s : sans prolongation, le chunk à t=10 tomberait
    clock = lambda: times.pop(0) if len(times) > 1 else times[0]  # noqa: E731
    tx, _ = await _run(
        ["JOSS début", "milieu", "fin"], command_timeout_s=8.0, time_fn=clock
    )
    assert tx == ["début", "milieu", "fin"]

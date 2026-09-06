"""LOT 1.5 — WakeWordGate : réveil « JOSS » + agrégation d'un tour en un seul message."""

from __future__ import annotations

import pytest

from pipecat.frames.frames import TranscriptionFrame
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language

from front.wakeword import WakeWordGate

# Délais courts pour des tests rapides ; on utilise SleepFrame pour le temps réel.
_AGG = 0.06
_TIMEOUT = 0.4


def _tx(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text, "", "2026-09-01T00:00:00Z", Language.FR)


async def _run(items):
    frames = [SleepFrame(x) if isinstance(x, float) else _tx(x) for x in items]
    down, _ = await run_test(
        WakeWordGate(aggregation_silence_s=_AGG, command_timeout_s=_TIMEOUT),
        frames_to_send=frames,
        expected_down_frames=None,
    )
    messages = [f.text for f in down if isinstance(f, TranscriptionFrame)]
    statuses = [f.data["status"] for f in down if isinstance(f, RTVIServerMessageFrame)]
    return messages, statuses


@pytest.mark.asyncio
async def test_no_wake_word_is_ignored():
    messages, statuses = await _run(["il fait beau aujourd'hui"])
    assert messages == []
    assert statuses == ["ignored"]


@pytest.mark.asyncio
async def test_single_utterance_becomes_one_message():
    messages, _ = await _run(["JOSS, quelle heure est-il ?"])
    assert messages == ["quelle heure est-il ?"]


@pytest.mark.asyncio
async def test_chunks_are_aggregated_into_one_message():
    # l'humain parle en 3 bouts (VAD coupe sur les pauses) -> UN seul message
    messages, statuses = await _run(
        ["JOSS raconte-moi", "une histoire", "assez courte s'il te plaît"]
    )
    assert messages == ["raconte-moi une histoire assez courte s'il te plaît"]
    assert statuses.count("command") == 1
    assert "listening" in statuses


@pytest.mark.asyncio
async def test_bare_wake_then_command():
    messages, statuses = await _run(["Joss.", "allume la lumière du salon"])
    assert messages == ["allume la lumière du salon"]
    assert statuses[0] == "awake"


@pytest.mark.asyncio
async def test_pause_longer_than_aggregation_still_same_turn_within_window():
    # pause de 0.15s > agrégation (0.06) mais < fenêtre : le 1er bout part,
    # le 2e est pris comme enchaînement (fenêtre de suivi), pas ignoré
    messages, _ = await _run(["JOSS lance la musique", 0.15, "monte le volume"])
    assert messages == ["lance la musique", "monte le volume"]


@pytest.mark.asyncio
async def test_goes_back_to_sleep_after_window():
    messages, statuses = await _run(["JOSS dis bonjour", 0.6, "il fait beau"])
    assert messages == ["dis bonjour"]  # "il fait beau" arrive après le sommeil
    assert statuses[-1] == "ignored"
    assert "asleep" in statuses


@pytest.mark.asyncio
async def test_fuzzy_variant_is_accepted():
    messages, _ = await _run(["Josh raconte une blague"])
    assert messages == ["raconte une blague"]

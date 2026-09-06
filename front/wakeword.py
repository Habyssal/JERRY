"""Mot de réveil « JOSS » — LOT 1.5.

Pas de vérification du locuteur : le front ne réagit qu'après avoir entendu le
mot de réveil, comme un assistant classique (Alexa…). L'identification des voix
récurrentes viendra plus tard, comme **module passif** alimenté par les
interactions réelles (cf. `Doc/Backlog.md`), pas par un enrôlement.

Implémentation LOT 1.5 : détection sur la **transcription STT** (réutilise
Parakeet, zéro dépendance). Remplaçable plus tard par un modèle wake-word dédié
(openWakeWord) placé en gate *avant* le STT — cf. `Doc/plans/Plan-JERRY.md`.

Placé après le STT, avant l'écho. Deux mécanismes :

1. **Réveil** : endormi + transcription sans « JOSS » → ignorée. « JOSS » (seul
   ou suivi de texte) → passe **éveillé** pour `command_timeout_s`.

2. **Agrégation de tour** : un humain parle en plusieurs bouts (le VAD coupe sur
   les pauses). Tant qu'on est éveillé, chaque bout est **accumulé** (et
   repousse la fenêtre) ; on n'émet **un seul message** vers l'aval qu'après
   `aggregation_silence_s` sans nouveau bout (fin de tour). Une pause au milieu
   de la phrase ou un barge-in ne coupe donc pas le message en morceaux.

Après émission d'un message, une fenêtre de suivi (`command_timeout_s`) reste
ouverte : un enchaînement / une correction est encore pris sans redire « JOSS ».
Événements RTVI `wake_word` : `ignored` / `awake` / `listening` / `command` / `asleep`.
"""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata

from loguru import logger

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

DEFAULT_WAKE_WORD = "joss"
# Variantes de transcription courantes du mot « JOSS » (FR/EN, Parakeet).
_EXTRA_VARIANTS = {"joss", "josse", "joce", "jos", "josh", "yoss", "jose", "gauss", "jaws"}


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _first_token(text: str) -> tuple[str, int]:
    """Premier mot normalisé (minuscule, sans accent) + index de fin dans `text`."""
    match = re.match(r"\s*[«»\"'(-]*\s*([^\W\d_]+)", text, re.UNICODE)
    if not match:
        return "", 0
    return _strip_accents(match.group(1).lower()), match.end()


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class WakeWordGate(FrameProcessor):
    """Filtre + agrège les transcriptions : un tour adressé via « JOSS » = un message."""

    def __init__(
        self,
        *,
        wake_word: str = DEFAULT_WAKE_WORD,
        command_timeout_s: float = 8.0,
        aggregation_silence_s: float = 1.2,
        max_edit_distance: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._wake_word = _strip_accents(wake_word.lower())
        self._variants = _EXTRA_VARIANTS | {self._wake_word}
        self._command_timeout_s = command_timeout_s
        self._aggregation_silence_s = aggregation_silence_s
        self._max_edit_distance = max_edit_distance

        self._deadline = 0.0  # time.monotonic() au-delà duquel on est endormi
        self._buffer: list[str] = []
        self._timer_handle: asyncio.TimerHandle | None = None
        self._timer_task: asyncio.Task | None = None
        self._last_user_id = ""
        self._last_language: Language | None = Language.FR

    def _is_wake_token(self, token: str) -> bool:
        if not token:
            return False
        if token in self._variants:
            return True
        return _levenshtein(token, self._wake_word) <= self._max_edit_distance

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._cancel_timer()
            await self._flush_buffer()
            self._deadline = 0.0
            await self.push_frame(frame, direction)
            return

        if not isinstance(frame, TranscriptionFrame) or not frame.text.strip():
            await self.push_frame(frame, direction)
            return

        text = frame.text.strip()
        self._last_user_id = frame.user_id
        self._last_language = frame.language

        awake = time.monotonic() < self._deadline
        token, end = _first_token(text)
        has_wake = self._is_wake_token(token)

        if not awake and not has_wake:
            if self._deadline != 0.0:  # on vient juste de dépasser la fenêtre
                self._deadline = 0.0
                await self._emit("asleep", "")
            logger.debug(f"WakeWordGate: ignoré (endormi, pas de « {self._wake_word} ») — « {text} »")
            await self._emit("ignored", text)
            return

        command = text[end:].lstrip(" ,.:;!?-–—»«\"'").strip() if has_wake else text

        await self._cancel_timer()
        self._deadline = time.monotonic() + self._command_timeout_s

        if command:
            self._buffer.append(command)
            logger.info(f"WakeWordGate: bout de tour — « {command} »")
            await self._emit("listening", command)
            self._arm_timer(self._aggregation_silence_s)
        else:
            logger.info(f"WakeWordGate: réveil (« {self._wake_word} ») — j'écoute")
            await self._emit("awake", "")
            self._arm_timer(self._command_timeout_s)

    def _arm_timer(self, delay: float) -> None:
        """(Ré)arme le déclencheur de fin de tour. `call_later` s'annule
        proprement sans coroutine — pas d'orphelin sur un ré-armement rapide."""
        self._disarm_timer_handle()
        self._timer_handle = asyncio.get_running_loop().call_later(delay, self._on_timer)

    def _disarm_timer_handle(self) -> None:
        if self._timer_handle is not None:
            self._timer_handle.cancel()
            self._timer_handle = None

    def _on_timer(self) -> None:
        self._timer_handle = None
        self._timer_task = asyncio.ensure_future(self._end_of_turn())

    async def _end_of_turn(self) -> None:
        if self._buffer:
            # Fin de tour : on émet le message agrégé. `_deadline` (posé au
            # dernier bout) laisse une fenêtre de suivi pour un enchaînement.
            await self._flush_buffer()
        else:
            # « JOSS » resté sans commande : retour au sommeil.
            self._deadline = 0.0
            logger.info("WakeWordGate: retour au sommeil")
            await self._emit("asleep", "")
        self._timer_task = None

    async def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        message = " ".join(self._buffer).strip()
        self._buffer = []
        logger.info(f"WakeWordGate: message complet — « {message} »")
        await self._emit("command", message)
        await self.push_frame(
            TranscriptionFrame(
                message, self._last_user_id, time_now_iso8601(), self._last_language
            ),
            FrameDirection.DOWNSTREAM,
        )

    async def _cancel_timer(self) -> None:
        self._disarm_timer_handle()
        if self._timer_task is not None:
            task, self._timer_task = self._timer_task, None
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def cleanup(self) -> None:
        await self._cancel_timer()
        await super().cleanup()

    async def _emit(self, status: str, text: str) -> None:
        await self.push_frame(
            RTVIServerMessageFrame(
                data={"type": "wake_word", "status": status, "wake_word": self._wake_word, "text": text}
            ),
            FrameDirection.DOWNSTREAM,
        )

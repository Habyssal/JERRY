"""Mot de réveil « JOSS » — LOT 1.5.

Pas de vérification du locuteur : le front ne réagit qu'après avoir entendu le
mot de réveil, comme un assistant classique (Alexa…). L'identification des voix
récurrentes viendra plus tard, comme **module passif** alimenté par les
interactions réelles (cf. `Doc/Backlog.md`), pas par un enrôlement.

Implémentation LOT 1.5 : détection sur la **transcription STT** (réutilise
Parakeet, zéro dépendance). Remplaçable plus tard par un modèle wake-word dédié
(openWakeWord) placé en gate *avant* le STT — cf. `Doc/plans/Plan-JERRY.md`.

Placé après le STT, avant l'écho :
- **endormi** + transcription sans « JOSS » → ignorée (pas d'écho).
- **endormi** + « JOSS <commande> » → la commande est transmise à l'aval.
- **endormi** + « JOSS » seul → passe en écoute, la transcription suivante
  (dans `command_timeout_s`) est traitée comme la commande.
- chaque changement d'état émet un événement RTVI `wake_word`.
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Callable

from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
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
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class WakeWordGate(FrameProcessor):
    """Filtre les transcriptions : seules celles adressées via « JOSS » passent à l'aval."""

    def __init__(
        self,
        *,
        wake_word: str = DEFAULT_WAKE_WORD,
        command_timeout_s: float = 8.0,
        max_edit_distance: int = 1,
        time_fn: Callable[[], float] = time.monotonic,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._wake_word = _strip_accents(wake_word.lower())
        self._variants = _EXTRA_VARIANTS | {self._wake_word}
        self._command_timeout_s = command_timeout_s
        self._max_edit_distance = max_edit_distance
        self._time_fn = time_fn  # pipecat réassigne self._clock, ne pas utiliser ce nom
        self._awaiting_command_until = 0.0

    def _is_wake_token(self, token: str) -> bool:
        if not token:
            return False
        if token in self._variants:
            return True
        return _levenshtein(token, self._wake_word) <= self._max_edit_distance

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if not isinstance(frame, TranscriptionFrame) or not frame.text.strip():
            await self.push_frame(frame, direction)
            return

        text = frame.text.strip()
        now = self._time_fn()

        if now < self._awaiting_command_until:
            self._awaiting_command_until = 0.0
            logger.info(f"WakeWordGate: commande (après réveil) — « {text} »")
            await self._emit("command", text)
            await self.push_frame(frame, direction)
            return

        token, end = _first_token(text)
        if self._is_wake_token(token):
            rest = text[end:].lstrip(" ,.:;!?-–—»\"'").strip()
            if rest:
                logger.info(f"WakeWordGate: réveil + commande — « {rest} »")
                await self._emit("command", rest)
                await self.push_frame(
                    TranscriptionFrame(rest, frame.user_id, time_now_iso8601(), frame.language),
                    direction,
                )
            else:
                self._awaiting_command_until = now + self._command_timeout_s
                logger.info(
                    f"WakeWordGate: réveil (« {self._wake_word} ») — écoute de la commande "
                    f"({self._command_timeout_s:.0f}s)"
                )
                await self._emit("awake", "")
            return

        logger.debug(f"WakeWordGate: ignoré (pas de mot de réveil) — « {text} »")
        await self._emit("ignored", text)

    async def _emit(self, status: str, text: str) -> None:
        await self.push_frame(
            RTVIServerMessageFrame(
                data={"type": "wake_word", "status": status, "wake_word": self._wake_word, "text": text}
            ),
            FrameDirection.DOWNSTREAM,
        )

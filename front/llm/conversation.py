"""Boucle de conversation du front (LOT 2a — pivot d'archi).

Remplace `LLMContextAggregatorPair` + `LLMMessagesAppendFrame` + le
`ContextAlternationGuard`. Raison : l'agrégateur pipecat pilote les tours au
**VAD** et refuse de relancer le LLM tant qu'il croit l'utilisateur ou le bot en
train de parler — ce qui **abandonnait silencieusement la 2ᵉ complétion après un
appel d'outil** (le résultat d'outil n'était jamais vocalisé). Ici l'autorité de
tour est `WakeWordGate` ; ce processeur relance le LLM **sans condition de
parole**.

Position dans le pipeline : **juste après le LLM**
`… → wake_gate → llm → conversation → tts → …`

- `TranscriptionFrame` (tour agrégé par `WakeWordGate`, qui a traversé le LLM
  sans être touché) : on interrompt tout tour en cours (`broadcast_interruption`),
  on mute la langue TTS, on ajoute le message `user` à l'historique et on pousse
  un `LLMContextFrame` **en amont** vers le LLM.
- `LLMTextFrame` (réponse assistant, streamée) : accumulée pour l'historique et
  **transmise** au TTS.
- `FunctionCallInProgressFrame` / `FunctionCallResultFrame` : on écrit dans
  l'historique le message `assistant` (avec `tool_calls`) puis le message `tool`
  (résultat), et on **relance** le LLM (`LLMContextFrame` en amont) — c'est
  cette relance que l'agrégateur pipecat sautait.
- `LLMFullResponseEndFrame` : on fige le message assistant accumulé.

L'historique est plafonné (`max_history`), en gardant le message système et une
tête de conversation valide (jamais couper une paire `assistant(tool_calls)` /
`tool`).
"""

from __future__ import annotations

import json
import time

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from front.services.tts_kokoro import KokoroTTSServiceFrEn

_DEFAULT_MAX_HISTORY = 24


def _result_to_str(result) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(result)


class FrontConversation(FrameProcessor):
    """Gère l'historique LLM et les relances, avec les tours pilotés par le mot de réveil."""

    def __init__(
        self,
        context: LLMContext,
        tts: KokoroTTSServiceFrEn,
        *,
        max_history: int = _DEFAULT_MAX_HISTORY,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._context = context
        self._tts = tts
        self._max_history = max_history

        self.turn_at: float | None = None  # horodatage du dernier tour (TTFA)
        self._assistant_buf: list[str] = []
        self._pending_tool_calls: dict[str, dict] = {}  # tool_call_id -> {name, arguments}

    # ------------------------------------------------------------------ history

    def _add_message(self, message: dict) -> None:
        self._context.add_message(message)
        self._trim_history()

    def _trim_history(self) -> None:
        messages = self._context.get_messages()
        system = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        if len(rest) <= self._max_history:
            return
        rest = rest[-self._max_history :]
        # La tête ne doit pas commencer par un 'tool' ni un 'assistant' orphelin.
        while rest and rest[0].get("role") in ("tool", "assistant"):
            rest.pop(0)
        self._context.set_messages(system + rest)

    def _drop_incomplete_tool_exchange(self) -> None:
        """Retire une paire `assistant(tool_calls)` / `tool` en fin d'historique
        dont la réponse assistant n'a jamais été produite (tour interrompu).
        Sinon le prochain message `user` casse l'alternance stricte du template
        Ministral (`tool, user` → 500)."""
        messages = self._context.get_messages()
        changed = False
        while messages and (
            messages[-1].get("role") == "tool"
            or (messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"))
        ):
            messages.pop()
            changed = True
        if changed:
            self._context.set_messages(messages)

    def _append_user_text(self, text: str) -> None:
        """Ajoute un tour utilisateur en garantissant l'alternance : fusionne
        avec le `user` précédent si le LLM n'a pas répondu entre-temps."""
        self._drop_incomplete_tool_exchange()
        messages = self._context.get_messages()
        if messages and messages[-1].get("role") == "user":
            prev = messages[-1].get("content") or ""
            messages[-1] = {"role": "user", "content": f"{prev}\n{text}".strip()}
            self._context.set_messages(messages)
        else:
            self._context.add_message({"role": "user", "content": text})
        self._trim_history()

    # ------------------------------------------------------------------ frames

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            await self._on_user_turn(frame)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            self._assistant_buf = []

        elif isinstance(frame, LLMTextFrame):
            if getattr(frame, "append_to_context", True):
                self._assistant_buf.append(frame.text)

        elif isinstance(frame, LLMFullResponseEndFrame):
            self._flush_assistant_message()

        elif isinstance(frame, FunctionCallInProgressFrame):
            self._pending_tool_calls[frame.tool_call_id] = {
                "name": frame.function_name,
                "arguments": frame.arguments or {},
            }

        elif isinstance(frame, FunctionCallResultFrame):
            await self._on_tool_result(frame, direction)

        await self.push_frame(frame, direction)

    # ------------------------------------------------------------------ turns

    async def _on_user_turn(self, frame: TranscriptionFrame) -> None:
        text = frame.text.strip()
        await self.broadcast_interruption()
        self._assistant_buf = []
        self._pending_tool_calls.clear()

        if frame.language is not None:
            self._tts.set_language(frame.language)

        self.turn_at = time.monotonic()
        self._append_user_text(text)
        logger.info(f"FrontConversation: tour utilisateur → LLM : « {text} »")
        await self.push_frame(LLMContextFrame(self._context), FrameDirection.UPSTREAM)

    async def _on_tool_result(self, frame: FunctionCallResultFrame, direction: FrameDirection) -> None:
        call = self._pending_tool_calls.pop(frame.tool_call_id, None)
        name = frame.function_name or (call or {}).get("name", "")
        arguments = frame.arguments if frame.arguments is not None else (call or {}).get("arguments", {})

        self._add_message(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": frame.tool_call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                    }
                ],
            }
        )
        self._add_message(
            {
                "role": "tool",
                "tool_call_id": frame.tool_call_id,
                "content": _result_to_str(frame.result),
            }
        )
        logger.info(f"FrontConversation: résultat outil « {name} » → relance LLM")
        await self.push_frame(LLMContextFrame(self._context), FrameDirection.UPSTREAM)

    def _flush_assistant_message(self) -> None:
        text = "".join(self._assistant_buf).strip()
        self._assistant_buf = []
        if text:
            self._add_message({"role": "assistant", "content": text})
            logger.info(f"FrontConversation: réponse assistant — « {text} »")


class TTFALogger(FrameProcessor):
    """Placé après le TTS : logue le TTFA (tour utilisateur → 1ʳᵉ frame audio)."""

    def __init__(self, conversation: FrontConversation, **kwargs) -> None:
        super().__init__(**kwargs)
        self._conversation = conversation

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSAudioRawFrame) and self._conversation.turn_at is not None:
            ttfa_ms = (time.monotonic() - self._conversation.turn_at) * 1000
            logger.info(f"TTFALogger: TTFA = {ttfa_ms:.0f} ms")
            self._conversation.turn_at = None

        await self.push_frame(frame, direction)

"""Garde d'alternance du contexte LLM (LOT 2a — correctif).

Le template de chat de Ministral (comme les vrais modèles Mistral) **exige une
alternance stricte** `user` / `assistant` après le message système, avec la seule
exception des appels d'outils (`assistant` avec `tool_calls` → `tool`). Toute
séquence `user, user` ou `tool, user` (sans `assistant` entre) fait renvoyer une
**500** par llama-server (`Jinja Exception: roles must alternate`).

Or la boucle vocale produit naturellement ces séquences :
- l'utilisateur enchaîne deux tours avant que JOSS ait répondu (barge-in, pause) ;
- une réponse en cours est **interrompue** (l'utilisateur reparle) → l'agrégateur
  assistant n'écrit jamais le message assistant → le contexte finit sur `user`
  ou `tool`, et le tour suivant ajoute un second `user` → contexte cassé **en
  cascade** (toutes les complétions suivantes échouent).

Ce processeur, placé **juste avant le LLM**, réécrit la liste de messages pour
garantir l'alternance, sans jamais inventer de contenu :
- messages `user` consécutifs → **fusionnés** (jointure par retour ligne) ;
- messages `assistant` texte consécutifs → fusionnés ;
- `tool` suivi d'un `user` (l'outil n'a jamais été verbalisé) → on **retire**
  la paire `assistant(tool_calls) + tool` orpheline et on rattache le `user` ;
- `assistant(tool_calls)` en fin de liste sans résultat → retiré.

La réécriture est appliquée **sur le contexte partagé** (`set_messages`) : une
fois nettoyé, il le reste pour les tours suivants (auto-réparation).
"""

from __future__ import annotations

from loguru import logger

from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def _text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def _merge_user(a: dict, b: dict) -> dict:
    parts = [t for t in (_text(a), _text(b)) if t]
    return {"role": "user", "content": "\n".join(parts)}


def _merge_assistant(a: dict, b: dict) -> dict:
    parts = [t for t in (_text(a), _text(b)) if t]
    return {"role": "assistant", "content": "\n".join(parts)}


def sanitize_messages(messages: list[dict]) -> list[dict]:
    """Renvoie une copie des messages garantissant l'alternance des rôles."""
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role")

        if not out or role == "system":
            out.append(msg)
            continue

        prev = out[-1].get("role")

        if role == "user" and prev == "user":
            out[-1] = _merge_user(out[-1], msg)
            continue

        if (
            role == "assistant"
            and prev == "assistant"
            and not msg.get("tool_calls")
            and not out[-1].get("tool_calls")
        ):
            out[-1] = _merge_assistant(out[-1], msg)
            continue

        if role == "user" and prev == "tool":
            # L'appel d'outil n'a jamais été verbalisé (réponse interrompue).
            # On retire la paire assistant(tool_calls) + tool orpheline.
            while out and out[-1].get("role") in ("tool", "assistant"):
                out.pop()
            if out and out[-1].get("role") == "user":
                out[-1] = _merge_user(out[-1], msg)
            else:
                out.append(msg)
            continue

        out.append(msg)

    # assistant(tool_calls) en fin de liste sans résultat d'outil -> retiré
    while out and out[-1].get("role") == "assistant" and out[-1].get("tool_calls"):
        out.pop()

    return out


class ContextAlternationGuard(FrameProcessor):
    """Nettoie le contexte LLM (alternance des rôles) avant chaque complétion."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            before = frame.context.get_messages()
            after = sanitize_messages(before)
            if len(after) != len(before):
                logger.info(
                    f"ContextAlternationGuard: contexte réécrit {len(before)} → {len(after)} messages"
                )
                frame.context.set_messages(after)

        await self.push_frame(frame, direction)

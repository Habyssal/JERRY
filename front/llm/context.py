"""Contexte LLM front + prompt système « JOSS » (LOT 2a).

Utilise le `LLMContext` universel de pipecat 1.3.0 (pas l'ancien
`OpenAILLMContext`). Le couple d'agrégateurs (`LLMContextAggregatorPair`) tient
l'historique à jour : tour utilisateur poussé via `LLMMessagesAppendFrame`
(cf. `turn.LLMTurnAdapter`), réponse assistant réinjectée automatiquement.
"""

from __future__ import annotations

import os

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext

# Réponse vocale : courte, sans mise en forme, dans la langue de l'utilisateur.
DEFAULT_SYSTEM_PROMPT = (
    "Tu es JOSS, un assistant personnel qui répond à la voix. "
    "Réponds toujours dans la langue de la dernière phrase de l'utilisateur "
    "(français ou anglais uniquement). "
    "Sois bref et direct : une à trois phrases, ton parlé, pas de listes, "
    "pas de Markdown, pas d'emojis, pas de balises. "
    "Si une demande dépasse tes outils disponibles, dis-le simplement. "
    "N'invente jamais le résultat d'un outil : appelle l'outil."
)


def build_context(tools: ToolsSchema | None = None) -> LLMContext:
    """Contexte initial : un seul message système. `tools` = schéma des outils
    exposés au LLM (None → aucun outil)."""
    system_prompt = os.environ.get("JERRY_LLM_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    messages = [{"role": "system", "content": system_prompt}]
    if tools is not None:
        return LLMContext(messages=messages, tools=tools)
    return LLMContext(messages=messages)

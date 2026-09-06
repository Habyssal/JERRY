"""Sonde manuelle — tool-calling xLAM / Ministral via Ollama (LOT 2a).

Pas un test pytest : à lancer à la main sur `sid` (Ollama local).
    uv run python tests/manual_llm_toolcall.py

Valide le critère du LOT 2 : « tool-calling xLAM validé via Ollama (bascule
Ministral si défaillant) ». Frappe directement l'API OpenAI-compat d'Ollama avec
le schéma d'outils de `front/llm/probe_tools.py`.
"""

from __future__ import annotations

import asyncio
import json

from openai import AsyncOpenAI

from front.llm.context import DEFAULT_SYSTEM_PROMPT
from front.llm.service import MINISTRAL_MODEL, XLAM_MODEL, DEFAULT_BASE_URL

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "obtenir_date_heure",
            "description": "Donne la date et l'heure actuelles.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "regler_minuteur",
            "description": "Règle un minuteur d'une durée donnée en secondes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duree_secondes": {
                        "type": "integer",
                        "description": "Durée du minuteur, en secondes.",
                    }
                },
                "required": ["duree_secondes"],
            },
        },
    },
]

PROMPTS = [
    "Quelle heure est-il ?",
    "Mets un minuteur de cinq minutes.",
    "Raconte-moi une blague courte.",  # ne doit PAS appeler d'outil
    "What time is it right now?",
]


async def probe_model(client: AsyncOpenAI, model: str) -> None:
    print(f"\n===== {model} =====")
    for prompt in PROMPTS:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=TOOLS,
            temperature=0.2,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            calls = [
                f"{c.function.name}({c.function.arguments})" for c in msg.tool_calls
            ]
            print(f"  [{prompt}]\n    -> tool_calls: {calls}")
        else:
            print(f"  [{prompt}]\n    -> texte: {msg.content!r}")


async def main() -> None:
    client = AsyncOpenAI(base_url=DEFAULT_BASE_URL, api_key="ollama")
    for model in (XLAM_MODEL, MINISTRAL_MODEL):
        try:
            await probe_model(client, model)
        except Exception as e:  # pragma: no cover - diagnostic
            print(f"  ERREUR {model}: {e}")


if __name__ == "__main__":
    asyncio.run(main())

"""Outils de démonstration — validation du tool-calling xLAM via Ollama (LOT 2a).

⚠️ Provisoire : sert uniquement à prouver que le LLM front sait déclencher une
fonction et en exploiter le résultat (critère de validation du LOT 2). La
convention de déclaration de skill définitive (nom, description, schéma args,
tags) est figée au LOT 3, et les skills métier réels (GPS…) arrivent ensuite.

Deux outils :
- `obtenir_date_heure` — aucun argument (cas le plus simple).
- `regler_minuteur` — un argument entier (teste l'extraction d'argument).
"""

from __future__ import annotations

import locale
from datetime import datetime, timedelta

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams, LLMService

try:  # affichage lisible en français si la locale est dispo sur la machine
    locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
except locale.Error:  # pragma: no cover - dépend de la machine
    pass


async def _obtenir_date_heure(params: FunctionCallParams) -> None:
    now = datetime.now().astimezone()
    result = {
        "iso": now.isoformat(timespec="minutes"),
        "lisible": now.strftime("%A %d %B %Y à %H:%M"),
    }
    logger.info(f"[probe tool] obtenir_date_heure → {result['lisible']}")
    await params.result_callback(result)


async def _regler_minuteur(params: FunctionCallParams) -> None:
    try:
        duree = int(params.arguments.get("duree_secondes", 0))
    except (TypeError, ValueError):
        await params.result_callback({"erreur": "duree_secondes doit être un entier"})
        return
    echeance = datetime.now().astimezone() + timedelta(seconds=duree)
    result = {"duree_secondes": duree, "echeance": echeance.strftime("%H:%M:%S")}
    logger.info(f"[probe tool] regler_minuteur({duree}s) → échéance {result['echeance']}")
    await params.result_callback(result)


def register_probe_tools(llm: LLMService) -> ToolsSchema:
    """Enregistre les handlers sur le service LLM et renvoie le schéma à passer
    au `LLMContext`."""
    llm.register_function("obtenir_date_heure", _obtenir_date_heure)
    llm.register_function("regler_minuteur", _regler_minuteur)

    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="obtenir_date_heure",
                description="Donne la date et l'heure actuelles.",
                properties={},
                required=[],
            ),
            FunctionSchema(
                name="regler_minuteur",
                description="Règle un minuteur d'une durée donnée en secondes.",
                properties={
                    "duree_secondes": {
                        "type": "integer",
                        "description": "Durée du minuteur, en secondes.",
                    }
                },
                required=["duree_secondes"],
            ),
        ]
    )

"""LLM front servi via Ollama (LOT 2a).

Modèle par défaut : **Ministral 3 3B Instruct** — bascule actée au LOT 2a.
Le plan désignait **xLAM-2-1b-fc-r** en principal, mais son tool-calling via
Ollama est cassé : llama-server renvoie une 500 en tentant de parser le format
de sortie natif de xLAM (`[nom_fonction](...)` pythonic, pas des `tool_calls`
OpenAI). Cf. `tests/manual_llm_toolcall.py` et `ENVIRONMENT.jerry.md`.
Ministral fait du function calling OpenAI-compatible propre out of the box
(args typés corrects, pas de faux appel, FR + EN). xLAM reste accessible via
`JERRY_LLM_MODEL` si un template Ollama correct est trouvé plus tard.

Ollama expose une API OpenAI-compatible sur `/v1` ; pipecat 1.3.0 fournit
`OLLamaLLMService` (sous-classe de `OpenAILLMService`) pour ça.

Config par variables d'environnement :
- `JERRY_LLM_MODEL`    — tag Ollama du modèle (défaut : xLAM).
- `JERRY_LLM_BASE_URL` — endpoint OpenAI-compat d'Ollama (défaut : localhost:11434).
- `JERRY_LLM_TEMPERATURE` — défaut 0.4.
"""

from __future__ import annotations

import os

from loguru import logger

from pipecat.services.ollama.llm import OLLamaLLMService

# Tags Ollama — cf. ENVIRONMENT.jerry.md (pull fait au LOT 0).
XLAM_MODEL = "hf.co/Salesforce/xLAM-2-1b-fc-r-gguf:Q4_K_M"
MINISTRAL_MODEL = "hf.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M"

DEFAULT_MODEL = MINISTRAL_MODEL  # bascule LOT 2a — xLAM tool-calling cassé via Ollama
DEFAULT_BASE_URL = "http://localhost:11434/v1"


def build_llm_service() -> OLLamaLLMService:
    """Construit le service LLM front. Ne charge rien : Ollama charge le modèle
    en mémoire à la première complétion (le warm-start éventuel se fait via un
    appel à vide au démarrage du pipeline)."""
    model = os.environ.get("JERRY_LLM_MODEL", DEFAULT_MODEL)
    base_url = os.environ.get("JERRY_LLM_BASE_URL", DEFAULT_BASE_URL)
    temperature = float(os.environ.get("JERRY_LLM_TEMPERATURE", "0.4"))

    logger.info(f"LLM front : {model} via {base_url} (temp={temperature})")
    return OLLamaLLMService(
        base_url=base_url,
        settings=OLLamaLLMService.Settings(model=model, temperature=temperature),
    )

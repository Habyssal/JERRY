"""LOT 2 — cognition du front : LLM front (Ollama) + conversation + outils.

Sous-lot 2a : branchement du LLM front après le mot de réveil.
- `service.build_llm_service()` : `OLLamaLLMService` — Ministral 3 3B par défaut
  (bascule LOT 2a : le tool-calling xLAM-2-1b-fc-r est cassé via Ollama).
- `context.build_context()` : `LLMContext` universel pipecat + prompt système « JOSS ».
- `conversation.FrontConversation` : gère l'historique et les relances du LLM,
  avec les tours pilotés par `WakeWordGate` (pas par le VAD). Remplace
  `LLMContextAggregatorPair` — voir la docstring du module pour le pourquoi.
- `probe_tools` : outils de démonstration pour valider le tool-calling (2a).
  La convention de skill définitive est figée au LOT 3.
"""

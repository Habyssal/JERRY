"""LOT 2 — cognition du front : LLM front (Ollama) + tour de parole + outils.

Sous-lot 2a : branchement du LLM front entre le mot de réveil et le TTS.
- `service.build_llm_service()` : `OLLamaLLMService` — Ministral 3 3B par défaut
  (bascule LOT 2a : le tool-calling xLAM-2-1b-fc-r est cassé via Ollama).
- `context.build_context()` : `LLMContext` universel pipecat + prompt système « JOSS ».
- `turn.LLMTurnAdapter` : convertit le message agrégé par `WakeWordGate` en tour
  utilisateur LLM (`LLMMessagesAppendFrame`, `run_llm=True`). Hébergera la
  compaction des disfluences au sous-lot 2b.
- `probe_tools` : un outil de démonstration pour valider le tool-calling xLAM (2a).
  La convention de skill définitive est figée au LOT 3.
"""

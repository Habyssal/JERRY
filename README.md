# JERRY — Front vocal

Submodule JOSS. Pipeline vocal (VAD → speaker verification → STT → LLM front → TTS), voir `Doc/plans/Plan-JERRY.md` dans le repo racine JOSS.

## Lancement

```bash
uv sync
uv run python -m front.enroll   # LOT 1.5 — une fois : enrôle l'empreinte vocale
uv run python -m front
```

Sans profil locuteur enrôlé, `python -m front` refuse de démarrer (le gate de
vérification du locuteur échoue au warm-start). Voir `front/speaker/config.py`
pour les seuils (`JERRY_SPEAKER_*`).

## Arborescence

- `front/` — pipeline vocal + LLM front + skills
- `front/skills/` — compétences métier (GPS, agenda, ...)
- `front/db/` — accès SQLite état partagé
- `front/prefilter/` — préfiltre sémantique (sentence-transformers)
- `tests/`

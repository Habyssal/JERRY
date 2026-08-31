# JERRY — Front vocal

Submodule JOSS. Pipeline vocal (VAD → speaker verification → STT → LLM front → TTS), voir `Doc/plans/Plan-JERRY.md` dans le repo racine JOSS.

## Lancement

```bash
uv sync
uv run python -m front
```

## Arborescence

- `front/` — pipeline vocal + LLM front + skills
- `front/skills/` — compétences métier (GPS, agenda, ...)
- `front/db/` — accès SQLite état partagé
- `front/prefilter/` — préfiltre sémantique (sentence-transformers)
- `tests/`

# system/ — config machine-locale versionnée

Fichiers qui ne vivent pas dans `front/` (pas du code Python déployé par
`uv sync`) mais qui font partie de la configuration réelle de la machine
JERRY — versionnés ici pour éviter toute dérive entre le repo et l'état
de la machine.

## pipewire-pulse.conf.d/99-jerry-echo-cancel.conf

Annulation d'écho (AEC) persistante entre le micro et la sortie audio —
sans elle, le micro capte la sortie TTS pendant le barge-in (hallucinations
de transcription, cf. `Doc/plans/Plan-JERRY.md` LOT 1 dans le repo JOSS).

**Installation sur une machine JERRY (une fois) :**

```bash
mkdir -p ~/.config/pipewire/pipewire-pulse.conf.d
ln -s ~/projects/JERRY/system/pipewire-pulse.conf.d/99-jerry-echo-cancel.conf \
      ~/.config/pipewire/pipewire-pulse.conf.d/99-jerry-echo-cancel.conf
systemctl --user restart pipewire-pulse.service
```

Les identifiants `source_master`/`sink_master` (nom exact du micro/sortie
ALSA) sont spécifiques à la machine — à ajuster si le matériel change
(voir `pactl list short sources|sinks` pour les noms réels).

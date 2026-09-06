"""Empreinte vocale ECAPA-TDNN — **non branché dans le pipeline** (décision 2026-09-01).

Le LOT 1.5 utilise un mot de réveil (`front/wakeword.py`), pas de vérification du
locuteur : l'écart de domaine enrôlement (lecture) / runtime (conversationnel)
rendait les scores de la vraie voix trop instables (0.2-0.6, jusqu'à négatif en
souriant / à distance).

Ce paquet est conservé pour le **futur module d'identification passive des voix
récurrentes** : pas d'enrôlement explicite, mais des profils construits en
continu à partir des interactions réelles, pour reconnaître les locuteurs
habituels (cf. `Doc/Backlog.md`). Réutilisables tels quels :
- `embedding.SpeakerEmbedder` — ECAPA-TDNN CPU
- `audio` — mesure d'énergie, découpe des portions non-voisées
- `profile.SpeakerProfile` — centroïde + scoring top-k

`verification.SpeakerVerificationGate` et `front/enroll.py` (enrôlement amont)
ne seront pas repris tels quels par le module passif.
"""

"""Vérification du locuteur (LOT 1.5) — empreinte vocale ECAPA-TDNN, 100% local, CPU.

Cascade d'écoute : VAD (Silero) -> [speaker verification] -> STT (Parakeet).
Le gate rejette toute voix qui n'est pas celle de l'utilisateur enrôlé AVANT que
le STT (GPU) ne tourne — d'où une conso GPU en veille quasi nulle.
"""

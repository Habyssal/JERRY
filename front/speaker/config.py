"""Configuration de la vérification du locuteur — surchargeable par variables d'environnement.

| Variable                       | Défaut                                   | Rôle |
|--------------------------------|------------------------------------------|------|
| JERRY_SPEAKER_PROFILE          | ~/.local/share/jerry/speaker_profile.npz | chemin du profil enrôlé |
| JERRY_SPEAKER_ACCEPT_THRESHOLD | 0.45                                     | score cosinus >= seuil -> accepté |
| JERRY_SPEAKER_REJECT_THRESHOLD | 0.30                                     | score cosinus <  seuil -> rejeté |
| JERRY_SPEAKER_ENROLL_PHRASES   | 5                                        | nb de phrases de référence à l'enrôlement |
| JERRY_SPEAKER_ENROLL_SECONDS   | 4.0                                      | durée d'enregistrement par phrase (s) |

Entre les deux seuils : zone de doute -> événement RTVI `speaker_verification`
status=uncertain (matérialisation vocale de la confirmation : LOT 2). Les seuils
sont à calibrer sur la vraie voix + un test voix tierce (les scores sont logués
à chaque tour).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROFILE_PATH = Path("~/.local/share/jerry/speaker_profile.npz").expanduser()
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class SpeakerConfig:
    """Seuils et chemins de la vérification du locuteur."""

    profile_path: Path
    accept_threshold: float
    reject_threshold: float
    enroll_phrases: int
    enroll_seconds: float
    sample_rate: int = SAMPLE_RATE

    def __post_init__(self) -> None:
        if self.reject_threshold > self.accept_threshold:
            raise ValueError(
                f"reject_threshold ({self.reject_threshold}) > accept_threshold "
                f"({self.accept_threshold}) : incohérent"
            )

    @classmethod
    def from_env(cls) -> "SpeakerConfig":
        return cls(
            profile_path=Path(
                os.environ.get("JERRY_SPEAKER_PROFILE", str(DEFAULT_PROFILE_PATH))
            ).expanduser(),
            accept_threshold=float(os.environ.get("JERRY_SPEAKER_ACCEPT_THRESHOLD", "0.45")),
            reject_threshold=float(os.environ.get("JERRY_SPEAKER_REJECT_THRESHOLD", "0.30")),
            enroll_phrases=int(os.environ.get("JERRY_SPEAKER_ENROLL_PHRASES", "5")),
            enroll_seconds=float(os.environ.get("JERRY_SPEAKER_ENROLL_SECONDS", "4.0")),
        )

"""Profil locuteur enrôlé : centroïde d'embeddings de référence, persisté sur disque.

Donnée **biométrique personnelle** — jamais versionnée (cf. `.gitignore`). Stockée
par défaut hors du repo (`~/.local/share/jerry/speaker_profile.npz`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from front.speaker.embedding import MODEL_SOURCE

_FORMAT_VERSION = 1


@dataclass
class SpeakerProfile:
    """Empreinte vocale de référence de l'utilisateur."""

    centroid: np.ndarray  # (192,), L2-normalisé
    embeddings: np.ndarray  # (n_phrases, 192), L2-normalisés — gardés pour ré-enrôlement/diagnostic
    sample_rate: int
    model_source: str
    created_at: str

    @property
    def n_phrases(self) -> int:
        return int(self.embeddings.shape[0])

    @classmethod
    def from_embeddings(
        cls, embeddings: list[np.ndarray], *, sample_rate: int
    ) -> "SpeakerProfile":
        stack = np.vstack(embeddings).astype(np.float32)
        mean = stack.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        centroid = (mean / norm if norm > 0.0 else mean).astype(np.float32)
        return cls(
            centroid=centroid,
            embeddings=stack,
            sample_rate=sample_rate,
            model_source=MODEL_SOURCE,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def similarity(self, embedding: np.ndarray) -> float:
        """Similarité cosinus au centroïde (les deux vecteurs sont L2-normalisés)."""
        return float(np.dot(self.centroid, embedding.astype(np.float32)))

    def self_consistency(self) -> float:
        """Similarité cosinus minimale entre phrases de référence — sanity check d'enrôlement."""
        n = self.n_phrases
        if n < 2:
            return 1.0
        sims = self.embeddings @ self.embeddings.T
        return float(sims[np.triu_indices(n, k=1)].min())

    def save(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            format_version=np.int64(_FORMAT_VERSION),
            centroid=self.centroid,
            embeddings=self.embeddings,
            sample_rate=np.int64(self.sample_rate),
            model_source=np.array(self.model_source),
            created_at=np.array(self.created_at),
        )
        # np.savez ajoute .npz si absent
        return path if path.suffix == ".npz" else path.with_suffix(".npz")

    @classmethod
    def load(cls, path: str | Path) -> "SpeakerProfile":
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Profil locuteur introuvable : {path}\n"
                f"Lancer l'enrôlement : uv run python -m front.enroll"
            )
        data = np.load(path, allow_pickle=False)
        model_source = str(data["model_source"])
        if model_source != MODEL_SOURCE:
            raise ValueError(
                f"Profil enrôlé avec un autre modèle ({model_source} != {MODEL_SOURCE}) "
                f"— ré-enrôler avec le modèle courant"
            )
        return cls(
            centroid=data["centroid"].astype(np.float32),
            embeddings=data["embeddings"].astype(np.float32),
            sample_rate=int(data["sample_rate"]),
            model_source=model_source,
            created_at=str(data["created_at"]),
        )

"""Embedding de locuteur ECAPA-TDNN (SpeechBrain), exécuté sur CPU.

Modèle : `speechbrain/spkrec-ecapa-voxceleb` — embedding 192-d, entrée 16 kHz mono.
Tourne sur CPU : la VRAM reste réservée au STT / TTS / LLM front. Chargé une fois
au warm-start (`load()`), jamais au fil de l'eau.
"""

from __future__ import annotations

import os

import numpy as np
from loguru import logger

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_SAVEDIR = os.path.expanduser("~/.cache/jerry/ecapa")


class SpeakerEmbedder:
    """Encapsule le modèle ECAPA-TDNN et produit des embeddings L2-normalisés."""

    def __init__(self, *, savedir: str = DEFAULT_SAVEDIR, device: str = "cpu") -> None:
        self._savedir = savedir
        self._device = device
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Charge (et télécharge au premier appel) le modèle ECAPA-TDNN."""
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # SpeechBrain < 1.0
            from speechbrain.pretrained import EncoderClassifier  # type: ignore

        logger.info(f"SpeakerEmbedder: chargement ECAPA-TDNN ({self._device})...")
        self._model = EncoderClassifier.from_hparams(
            source=MODEL_SOURCE,
            savedir=self._savedir,
            run_opts={"device": self._device},
        )
        self._model.eval()
        logger.info("SpeakerEmbedder: modèle chargé.")

    def embed_pcm16(self, pcm: bytes, sample_rate: int = 16000) -> np.ndarray:
        """Embedding L2-normalisé (192-d) d'un segment PCM 16-bit mono."""
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return self.embed_float(audio, sample_rate)

    def embed_float(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Embedding L2-normalisé (192-d) d'un segment float32 mono dans [-1, 1]."""
        if self._model is None:
            raise RuntimeError("SpeakerEmbedder non chargé — appeler load() au warm-start")
        if sample_rate != 16000:
            raise ValueError(f"ECAPA attend du 16 kHz, reçu {sample_rate} Hz")

        import torch

        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            emb = self._model.encode_batch(wav).reshape(-1).cpu().numpy()

        norm = float(np.linalg.norm(emb))
        if norm > 0.0:
            emb = emb / norm
        return emb.astype(np.float32)

"""LOT 1.5 — vérification manuelle de la discrimination ECAPA sur deux voix synthétiques.

Ne nécessite pas de micro : synthétise plusieurs phrases avec deux voix Kokoro
distinctes (proxy « utilisateur » vs « voix tierce »), enrôle la première, puis
score la voix enrôlée (tenue à l'écart) et la voix tierce.

    uv run python tests/manual_speaker_ecapa.py

Sortie attendue : score voix enrôlée nettement > score voix tierce, avec une
marge exploitable pour placer accept/reject.
"""

from __future__ import annotations

import numpy as np

from front.speaker.config import SAMPLE_RATE
from front.speaker.embedding import SpeakerEmbedder
from front.speaker.profile import SpeakerProfile

_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "I would like a cup of coffee and a slice of toast for breakfast today.",
    "She sells seashells by the seashore on a bright and windy afternoon.",
    "We are testing whether the system can recognise my voice reliably.",
    "Please remember to lock the front door before you leave the house.",
    "The train to the city departs from platform four at half past nine.",
]

_ENROLLED_VOICE = "af_heart"
_OTHER_VOICE = "am_michael"


def _synth(kokoro, voice: str, text: str) -> np.ndarray:
    import torchaudio

    samples, sr = kokoro.create(text, voice=voice, lang="en-us", speed=1.0)
    samples = np.asarray(samples, dtype=np.float32)
    if sr != SAMPLE_RATE:
        import torch

        samples = (
            torchaudio.functional.resample(torch.from_numpy(samples), sr, SAMPLE_RATE)
            .numpy()
            .astype(np.float32)
        )
    return samples


def main() -> None:
    from kokoro_onnx import Kokoro
    from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR

    kokoro = Kokoro(
        str(KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"),
        str(KOKORO_CACHE_DIR / "voices-v1.0.bin"),
    )
    voices = set(kokoro.get_voices())
    enrolled_voice = _ENROLLED_VOICE if _ENROLLED_VOICE in voices else sorted(voices)[0]
    other_voice = _OTHER_VOICE if _OTHER_VOICE in voices else sorted(voices)[-1]
    print(f"voix enrôlée = {enrolled_voice} · voix tierce = {other_voice}")

    embedder = SpeakerEmbedder(device="cpu")
    embedder.load()

    enrolled = [_synth(kokoro, enrolled_voice, s) for s in _SENTENCES]
    other = [_synth(kokoro, other_voice, s) for s in _SENTENCES]

    enroll_embs = [embedder.embed_float(a, SAMPLE_RATE) for a in enrolled[:-1]]
    profile = SpeakerProfile.from_embeddings(enroll_embs, sample_rate=SAMPLE_RATE)

    held_out = profile.similarity(embedder.embed_float(enrolled[-1], SAMPLE_RATE))
    other_scores = [profile.similarity(embedder.embed_float(a, SAMPLE_RATE)) for a in other]

    print(f"cohérence interne du profil : {profile.self_consistency():.3f}")
    print(f"score voix enrôlée (tenue à l'écart) : {held_out:.3f}")
    print(f"scores voix tierce : {[round(s, 3) for s in other_scores]}")
    print(f"  min tierce = {min(other_scores):.3f} · max tierce = {max(other_scores):.3f}")
    margin = held_out - max(other_scores)
    print(f"marge (enrôlé - max tierce) : {margin:.3f}")
    verdict = "OK" if margin > 0.1 else "MARGE FAIBLE"
    print(f"verdict : {verdict}")


if __name__ == "__main__":
    main()

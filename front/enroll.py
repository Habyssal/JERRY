"""Enrôlement de l'empreinte vocale de l'utilisateur (LOT 1.5).

Enregistre plusieurs phrases de référence au micro (périphérique d'entrée par
défaut — ici la source anti-echo `jerry_echo_source`, pour rester cohérent avec
ce que voit le pipeline runtime), découpe les silences, calcule un embedding
ECAPA-TDNN par phrase, écarte la prise aberrante éventuelle, et persiste le
centroïde normalisé comme profil locuteur.

    uv run python -m front.enroll                 # enrôlement standard
    uv run python -m front.enroll --list-devices  # lister les micros
    uv run python -m front.enroll --phrases 6 --seconds 5 --device Razer

Le profil est écrit hors du repo par défaut
(`~/.local/share/jerry/speaker_profile.npz`) — donnée biométrique, jamais versionnée.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from loguru import logger

from front.speaker import audio as _audio
from front.speaker.config import SpeakerConfig
from front.speaker.embedding import SpeakerEmbedder
from front.speaker.profile import SpeakerProfile

_REFERENCE_PHRASES = [
    "Bonjour, c'est moi. Je vérifie que tu reconnais bien ma voix ce matin.",
    "Le petit chat dort tranquillement sur le canapé pendant que la pluie tombe dehors.",
    "Aujourd'hui je vais préparer le déjeuner, puis sortir faire quelques courses en ville.",
    "One, two, three, four, five — this is my voice for the speaker verification test.",
    "La montagne est haute et le sentier qui mène au sommet est long, escarpé et sinueux.",
    "J'aime écouter de la musique le soir, en lisant un bon livre, au calme, sans écran.",
    "Il faut battre le fer tant qu'il est chaud, dit le vieux proverbe que tout le monde connaît.",
    "Demain, s'il fait beau, nous irons marcher le long de la rivière jusqu'au vieux pont de pierre.",
]

_FRAMES_PER_BUFFER = 1024
_MIN_VOICED_SECONDS = 1.3  # voix nette exigée par phrase après découpe des silences
_MAX_RETRIES = 3


def _list_devices(pa) -> None:
    print("Périphériques d'entrée disponibles :")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  [{i:2d}] {info['name']}  ({int(info['defaultSampleRate'])} Hz)")
    try:
        print(f"Défaut : {pa.get_default_input_device_info()['name']}")
    except OSError:
        print("Défaut : (aucun)")


def _resolve_device(pa, wanted: str | None) -> int | None:
    if wanted is None:
        return None
    try:
        return int(wanted)
    except ValueError:
        pass
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0 and wanted.lower() in info["name"].lower():
            logger.info(f"Micro : [{i}] {info['name']}")
            return i
    logger.error(f"Aucun micro ne correspond à « {wanted} »")
    sys.exit(2)


def _record(pa, seconds: float, sample_rate: int, device_index: int | None) -> np.ndarray:
    stream = pa.open(
        format=8,  # pyaudio.paInt16
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=_FRAMES_PER_BUFFER,
        input_device_index=device_index,
    )
    buffer = bytearray()
    n_reads = int(sample_rate / _FRAMES_PER_BUFFER * seconds)
    try:
        for _ in range(n_reads):
            buffer += stream.read(_FRAMES_PER_BUFFER, exception_on_overflow=False)
    finally:
        stream.stop_stream()
        stream.close()
    return _audio.to_float(bytes(buffer))


def _countdown(prefix: str) -> None:
    for n in (3, 2, 1):
        print(f"\r{prefix} — {n}...  ", end="", flush=True)
        time.sleep(0.7)
    print("\r" + " " * (len(prefix) + 14) + "\r", end="", flush=True)


def _measure_noise(pa, sample_rate: int, device_index: int | None) -> float:
    print("Mesure du bruit de fond — reste **silencieux** 2 secondes...", flush=True)
    time.sleep(0.4)
    samples = _record(pa, 2.0, sample_rate, device_index)
    noise = _audio.rms(samples)
    verdict = "ok" if noise < 0.015 else "élevé — pense à réduire le bruit / le gain micro"
    print(f"  bruit de fond RMS = {noise:.4f}  ({verdict})\n")
    return noise


def main() -> None:
    config = SpeakerConfig.from_env()

    parser = argparse.ArgumentParser(description="Enrôlement empreinte vocale JERRY (LOT 1.5)")
    parser.add_argument("--phrases", type=int, default=config.enroll_phrases)
    parser.add_argument("--seconds", type=float, default=config.enroll_seconds)
    parser.add_argument("--device", default=None, help="index ou sous-chaîne du nom du micro")
    parser.add_argument("--output", default=str(config.profile_path))
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        if args.list_devices:
            _list_devices(pa)
            return

        n_phrases = max(3, min(args.phrases, len(_REFERENCE_PHRASES)))
        device_index = _resolve_device(pa, args.device)

        embedder = SpeakerEmbedder(device="cpu")
        embedder.load()

        print()
        print("=== Enrôlement locuteur JERRY ===")
        print(f"{n_phrases} phrases · {args.seconds:.0f}s chacune · profil -> {args.output}")
        print("Parle normalement, à ton débit habituel, à distance constante du micro.\n")

        noise_rms = _measure_noise(pa, config.sample_rate, device_index)
        voiced_threshold = max(0.012, noise_rms * 2.0)

        embeddings: list[np.ndarray] = []
        voiced_durations: list[float] = []
        for idx in range(n_phrases):
            phrase = _REFERENCE_PHRASES[idx]
            print(f"[{idx + 1}/{n_phrases}] Lis à voix haute :")
            print(f"    « {phrase} »")

            for attempt in range(1, _MAX_RETRIES + 1):
                input("    (Entrée quand tu es prêt) ")
                _countdown("    Enregistrement dans")
                print("    🎙️  ... parle maintenant", flush=True)
                samples = _record(pa, args.seconds, config.sample_rate, device_index)

                voiced = _audio.keep_voiced(
                    samples, config.sample_rate, threshold=voiced_threshold
                )
                voiced_s = voiced.size / config.sample_rate
                if voiced_s < _MIN_VOICED_SECONDS:
                    print(
                        f"    ⚠️  seulement {voiced_s:.1f}s de voix nette détectée "
                        f"(min {_MIN_VOICED_SECONDS}s) — on recommence cette phrase "
                        f"(essai {attempt}/{_MAX_RETRIES}).\n"
                    )
                    continue

                embeddings.append(embedder.embed_float(voiced, config.sample_rate))
                voiced_durations.append(voiced_s)
                print(f"    ✓ capturé ({voiced_s:.1f}s de voix nette)\n")
                break
            else:
                print("    ✗ phrase abandonnée après plusieurs essais.\n")

        if len(embeddings) < 3:
            logger.error(
                f"Seulement {len(embeddings)} phrase(s) exploitable(s) (min 3) — "
                f"enrôlement abandonné. Réessaie dans un endroit plus calme, "
                f"micro plus proche, ou --device sur le micro brut."
            )
            sys.exit(1)

        profile = SpeakerProfile.from_embeddings(embeddings, sample_rate=config.sample_rate)
        written = profile.save(args.output)

        consistency = profile.self_consistency()
        pairwise = profile.embeddings @ profile.centroid
        dropped = len(embeddings) - profile.n_phrases
        print("=== Profil enregistré ===")
        print(f"  fichier            : {written}")
        print(f"  phrases retenues   : {profile.n_phrases}" + (f" ({dropped} écartée(s))" if dropped else ""))
        print(f"  voix nette / phrase: moy {np.mean(voiced_durations):.1f}s")
        print(f"  cohérence interne  : {consistency:.3f} (min. cosinus entre phrases)")
        print(f"  phrase<->centroïde : min {pairwise.min():.3f} / moy {pairwise.mean():.3f}")
        print()
        if consistency < 0.55:
            print("  ⚠️  cohérence encore faible malgré la découpe des silences.")
            print("      Pistes : endroit plus calme, distance au micro vraiment constante,")
            print("      baisser le gain (`pactl set-source-volume <source> 60%`),")
            print("      ou enrôler sur le micro brut (`--device Razer`).")
        else:
            lo = round(max(0.0, pairwise.mean() - 0.20), 2)
            hi = round(max(lo + 0.05, pairwise.mean() - 0.08), 2)
            print("  Piste de calibrage initiale (à affiner avec un test voix tierce) :")
            print(f"      export JERRY_SPEAKER_REJECT_THRESHOLD={lo}")
            print(f"      export JERRY_SPEAKER_ACCEPT_THRESHOLD={hi}")
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()

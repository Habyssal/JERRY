"""Enrôlement de l'empreinte vocale de l'utilisateur (LOT 1.5).

Enregistre plusieurs prises de **voix naturelle** au micro (périphérique d'entrée
par défaut — ici la source anti-echo `jerry_echo_source`, pour rester cohérent
avec ce que voit le pipeline runtime), découpe les silences, calcule un embedding
ECAPA-TDNN par prise, écarte la prise aberrante éventuelle, et persiste le
centroïde normalisé comme profil locuteur.

Chaque prise est **à ton rythme** : démarre sur [Entrée], arrête sur [Entrée]
(ou au bout de --max-seconds). Parle comme tu parleras à l'assistant — le
contenu importe peu (ECAPA est indépendant du texte), c'est le naturel et la
durée (~10-15 s) qui comptent.

    uv run python -m front.enroll                 # enrôlement standard
    uv run python -m front.enroll --list-devices  # lister les micros
    uv run python -m front.enroll --takes 6 --device Razer

Le profil est écrit hors du repo par défaut
(`~/.local/share/jerry/speaker_profile.npz`) — donnée biométrique, jamais versionnée.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np
from loguru import logger

from front.speaker import audio as _audio
from front.speaker.config import SpeakerConfig
from front.speaker.embedding import SpeakerEmbedder
from front.speaker.profile import SpeakerProfile

# Sujets ouverts : l'empreinte doit refléter la façon dont tu parles VRAIMENT à
# l'assistant (débit, prosodie, distance au micro), pas une lecture appliquée.
# ECAPA est indépendant du texte : ce qui compte c'est ~8-15 s de voix naturelle
# par prise, pas le contenu.
_ENROLL_PROMPTS = [
    "Raconte ta journée d'hier, en quelques phrases, à voix normale.",
    "Décris la pièce où tu es : ce que tu vois autour de toi.",
    "Explique ce que tu attends de cet assistant vocal.",
    "Parle de quelque chose que tu aimes faire (loisir, sujet, lieu).",
    "Compte à voix haute de un à vingt, tranquillement, puis dis l'heure qu'il est.",
    "Dis ce que tu comptes faire ce week-end, ou la semaine prochaine.",
    "Lis à voix haute le premier paragraphe d'un article ou d'un livre à portée de main.",
    "Décris le trajet que tu fais le plus souvent (maison, travail, courses).",
]

_FRAMES_PER_BUFFER = 1024
_MIN_VOICED_SECONDS = 4.0  # voix nette exigée par prise (ECAPA a besoin de plusieurs s)
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


def _read_fixed(pa, seconds: float, sample_rate: int, device_index: int | None) -> np.ndarray:
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


def _read_until_enter(
    pa, max_seconds: float, sample_rate: int, device_index: int | None
) -> np.ndarray:
    """Enregistre jusqu'à ce que l'utilisateur appuie sur [Entrée] (ou max_seconds)."""
    stream = pa.open(
        format=8,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=_FRAMES_PER_BUFFER,
        input_device_index=device_index,
    )
    stop = threading.Event()

    def _wait_for_enter() -> None:
        try:
            input()
        except EOFError:
            pass
        stop.set()

    threading.Thread(target=_wait_for_enter, daemon=True).start()

    buffer = bytearray()
    started = time.monotonic()
    try:
        while not stop.is_set() and (time.monotonic() - started) < max_seconds:
            buffer += stream.read(_FRAMES_PER_BUFFER, exception_on_overflow=False)
    finally:
        stream.stop_stream()
        stream.close()
    return _audio.to_float(bytes(buffer))


def _measure_noise(pa, sample_rate: int, device_index: int | None) -> float:
    print("Mesure du bruit de fond — reste SILENCIEUX 2 secondes...", flush=True)
    time.sleep(0.4)
    samples = _read_fixed(pa, 2.0, sample_rate, device_index)
    noise = _audio.rms(samples)
    verdict = "ok" if noise < 0.015 else "élevé — réduis le bruit ambiant ou le gain micro"
    print(f"  bruit de fond RMS = {noise:.4f}  ({verdict})\n")
    return noise


def main() -> None:
    config = SpeakerConfig.from_env()

    parser = argparse.ArgumentParser(description="Enrôlement empreinte vocale JERRY (LOT 1.5)")
    parser.add_argument("--takes", type=int, default=6, help="nombre de prises de voix")
    parser.add_argument("--max-seconds", type=float, default=25.0, help="garde-fou par prise")
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

        n_takes = max(3, min(args.takes, len(_ENROLL_PROMPTS)))
        device_index = _resolve_device(pa, args.device)

        embedder = SpeakerEmbedder(device="cpu")
        embedder.load()

        print()
        print("=== Enrôlement locuteur JERRY ===")
        print(f"{n_takes} prises · à ton rythme · profil -> {args.output}")
        print("Parle NATURELLEMENT, comme tu parleras à l'assistant — pas une lecture")
        print("appliquée. ~10-15 s par prise. Distance au micro comme à l'usage réel.")
        print("[Entrée] pour démarrer une prise, [Entrée] à nouveau quand tu as fini.\n")

        noise_rms = _measure_noise(pa, config.sample_rate, device_index)
        voiced_threshold = max(0.012, noise_rms * 2.0)

        embeddings: list[np.ndarray] = []
        voiced_durations: list[float] = []
        for idx in range(n_takes):
            prompt = _ENROLL_PROMPTS[idx]
            print(f"[{idx + 1}/{n_takes}] {prompt}")

            for attempt in range(1, _MAX_RETRIES + 1):
                input("    [Entrée] pour démarrer ")
                print("    🎙️  enregistrement — parle, puis [Entrée] pour arrêter", flush=True)
                samples = _read_until_enter(
                    pa, args.max_seconds, config.sample_rate, device_index
                )

                voiced = _audio.keep_voiced(
                    samples, config.sample_rate, threshold=voiced_threshold
                )
                voiced_s = voiced.size / config.sample_rate
                if voiced_s < _MIN_VOICED_SECONDS:
                    print(
                        f"    ⚠️  seulement {voiced_s:.1f}s de voix nette "
                        f"(min {_MIN_VOICED_SECONDS}s) — parle plus longuement, "
                        f"on recommence (essai {attempt}/{_MAX_RETRIES}).\n"
                    )
                    continue

                embeddings.append(embedder.embed_float(voiced, config.sample_rate))
                voiced_durations.append(voiced_s)
                print(f"    ✓ capturé ({voiced_s:.1f}s de voix nette)\n")
                break
            else:
                print("    ✗ prise abandonnée après plusieurs essais.\n")

        if len(embeddings) < 3:
            logger.error(
                f"Seulement {len(embeddings)} prise(s) exploitable(s) (min 3) — "
                f"enrôlement abandonné. Endroit plus calme, micro plus proche, "
                f"ou --device sur le micro brut."
            )
            sys.exit(1)

        profile = SpeakerProfile.from_embeddings(embeddings, sample_rate=config.sample_rate)
        written = profile.save(args.output)

        to_centroid = profile.embeddings @ profile.centroid
        dropped = len(embeddings) - profile.n_phrases
        mean_c = float(to_centroid.mean())

        print("=== Profil enregistré ===")
        print(f"  fichier             : {written}")
        print(
            f"  prises retenues     : {profile.n_phrases}"
            + (f" ({dropped} écartée(s) comme aberrante(s))" if dropped else "")
        )
        print(f"  voix nette / prise  : moy {np.mean(voiced_durations):.1f}s "
              f"(total {np.sum(voiced_durations):.0f}s)")
        print(f"  prise <-> centroïde : moy {mean_c:.3f} / min {to_centroid.min():.3f}")
        print()

        if mean_c < 0.45:
            print("  ⚠️  cohérence trop faible — le profil sera peu fiable.")
            print("      Endroit plus calme, distance au micro VRAIMENT constante,")
            print("      baisser le gain (`pactl set-source-volume jerry_echo_source 60%`),")
            print("      ou enrôler sur le micro brut (`--device Razer`).")
            return

        print("  La cohérence d'enrôlement n'est PAS le score runtime attendu :")
        print("  ta voix en conditions réelles scorera plus bas. Ce qui compte,")
        print("  c'est l'écart avec une voix tierce — mesure-le au test live, puis :")
        print("      REJECT = un peu au-dessus du max des scores voix tierce")
        print("      ACCEPT = un peu en-dessous du min de TES scores (segments > 1.5 s)")
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()

"""Correctif local : le mapping Kokoro de pipecat 1.3.0 résout Language.FR en "fr",
mais le backend espeak-ng installé ici n'accepte que le code locale "fr-fr"
(confirmé manuellement avec kokoro_onnx.Kokoro.create(..., lang="fr-fr")).

Bilingue FR/EN : la langue ET la voix sont mutées par tour de parole via
set_language(), en fonction de la langue détectée côté STT (front/langfilter.py).
"""

from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.transcriptions.language import Language

_SERVICE_LANGUAGE_CODES = {
    Language.FR: "fr-fr",
    Language.EN: "en-us",
}

VOICE_BY_LANGUAGE = {
    Language.FR: "ff_siwis",
    Language.EN: "af_heart",
}


class KokoroTTSServiceFrEn(KokoroTTSService):
    """KokoroTTSService bilingue FR/EN (LOT 1) — langue/voix mutées par tour via set_language()."""

    def language_to_service_language(self, language: Language) -> str:
        return _SERVICE_LANGUAGE_CODES.get(language, _SERVICE_LANGUAGE_CODES[Language.FR])

    def set_language(self, language: Language) -> None:
        """À appeler avant de pousser le texte à vocaliser du prochain tour.

        Stocke directement la chaîne résolue ("fr-fr"/"en-us") dans les
        settings : run_tts() lit self._settings.language tel quel, sans
        repasser par language_to_service_language() à cet endroit.
        """
        self._settings.language = self.language_to_service_language(language)
        self._settings.voice = VOICE_BY_LANGUAGE.get(language, VOICE_BY_LANGUAGE[Language.FR])

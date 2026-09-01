"""Filtre de langue : rejette silencieusement toute transcription qui ne "capte"
pas du français ou de l'anglais, sans jamais traduire.

Le français et l'anglais sont priorisés explicitement : on regarde leur
confiance propre (pas seulement "meilleure langue parmi N"), et on accepte dès
que l'une des deux dépasse un seuil bas. Seul un texte dont la confiance FR ET
EN est très faible (vrai charabia, aucune des deux langues ne correspond) est
rejeté. La détection se fait quand même sur un jeu de langues plus large
(langues européennes proches, sources de confusion courantes) : sans ça, un
charabia forcé entre seulement FR/EN obtient un score artificiellement gonflé
faute d'alternative, ce qui rend le seuil inutilisable.
"""

from lingua import Language, LanguageDetectorBuilder

_CANDIDATE_LANGUAGES = [
    Language.FRENCH,
    Language.ENGLISH,
    Language.SPANISH,
    Language.GERMAN,
    Language.ITALIAN,
    Language.PORTUGUESE,
]

_MIN_CONFIDENCE = 0.20

_detector = None


def build_detector():
    """Construit le détecteur (coûteux au premier appel) — à faire au warm-start."""
    global _detector
    if _detector is None:
        _detector = LanguageDetectorBuilder.from_languages(*_CANDIDATE_LANGUAGES).build()
    return _detector


def detect_fr_or_en(text: str) -> Language | None:
    """Retourne Language.FRENCH ou Language.ENGLISH si l'un des deux dépasse le
    seuil de confiance minimal (les deux sont priorisés à égalité, on garde le
    meilleur des deux), sinon None (charabia, ni l'un ni l'autre)."""
    detector = build_detector()
    confidences = {cv.language: cv.value for cv in detector.compute_language_confidence_values(text)}
    fr_confidence = confidences.get(Language.FRENCH, 0.0)
    en_confidence = confidences.get(Language.ENGLISH, 0.0)

    if fr_confidence < _MIN_CONFIDENCE and en_confidence < _MIN_CONFIDENCE:
        return None
    return Language.FRENCH if fr_confidence >= en_confidence else Language.ENGLISH

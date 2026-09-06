"""LOT 2a — FrontConversation : nettoyage du texte pour le TTS + alternance."""

from __future__ import annotations

from front.llm.conversation import clean_for_speech


def test_strips_emoji():
    assert clean_for_speech("Bonjour ! 😊") == "Bonjour ! "
    # les espaces laissés par les emojis retirés sont compactés
    assert clean_for_speech("C'est parti 🚀🔥 maintenant") == "C'est parti maintenant"


def test_strips_markdown_markers():
    assert clean_for_speech("Il est **19h12** aujourd'hui") == "Il est 19h12 aujourd'hui"
    assert clean_for_speech("# Titre\n- point") == " Titre\n- point"
    assert clean_for_speech("code `foo` ici") == "code foo ici"


def test_keeps_plain_text_and_underscores():
    assert clean_for_speech("un fichier nom_de_variable.py") == "un fichier nom_de_variable.py"
    assert clean_for_speech("phrase normale.") == "phrase normale."

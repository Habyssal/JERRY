"""LOT 2a correctif — ContextAlternationGuard : le contexte reste valide pour
le template Ministral (alternance stricte user/assistant)."""

from __future__ import annotations

from front.llm.context_guard import sanitize_messages

SYS = {"role": "system", "content": "Tu es JOSS."}


def _roles(messages):
    return [m["role"] for m in messages]


def test_consecutive_user_messages_are_merged():
    out = sanitize_messages(
        [
            SYS,
            {"role": "user", "content": "quelle heure il est"},
            {"role": "user", "content": "mets un minuteur"},
            {"role": "user", "content": "dis bonjour"},
        ]
    )
    assert _roles(out) == ["system", "user"]
    assert out[1]["content"] == "quelle heure il est\nmets un minuteur\ndis bonjour"


def test_valid_alternation_is_untouched():
    msgs = [
        SYS,
        {"role": "user", "content": "bonjour"},
        {"role": "assistant", "content": "Bonjour !"},
        {"role": "user", "content": "ça va ?"},
    ]
    assert sanitize_messages(msgs) == msgs


def test_valid_tool_call_flow_is_untouched():
    msgs = [
        SYS,
        {"role": "user", "content": "quelle heure ?"},
        {"role": "assistant", "tool_calls": [{"id": "a", "type": "function",
                                              "function": {"name": "obtenir_date_heure", "arguments": "{}"}}]},
        {"role": "tool", "content": '{"lisible": "18h38"}', "tool_call_id": "a"},
    ]
    assert sanitize_messages(msgs) == msgs


def test_tool_result_then_user_drops_orphan_tool_exchange():
    # séquence exacte du bug live : l'appel d'outil n'a jamais été verbalisé
    # (réponse interrompue), puis l'utilisateur reparle.
    out = sanitize_messages(
        [
            SYS,
            {"role": "user", "content": "quelle heure ?"},
            {"role": "assistant", "tool_calls": [{"id": "a", "type": "function",
                                                  "function": {"name": "obtenir_date_heure", "arguments": "{}"}}]},
            {"role": "tool", "content": '{"lisible": "18h38"}', "tool_call_id": "a"},
            {"role": "user", "content": "mets un minuteur"},
        ]
    )
    assert _roles(out) == ["system", "user"]
    assert out[1]["content"] == "quelle heure ?\nmets un minuteur"


def test_dangling_assistant_tool_calls_is_removed():
    out = sanitize_messages(
        [
            SYS,
            {"role": "user", "content": "quelle heure ?"},
            {"role": "assistant", "tool_calls": [{"id": "a", "type": "function",
                                                  "function": {"name": "x", "arguments": "{}"}}]},
        ]
    )
    assert _roles(out) == ["system", "user"]

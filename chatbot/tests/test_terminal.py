# tests/test_terminal.py
from unittest.mock import MagicMock

from ui.terminal import TerminalUI


def _make_chatbot():
    chatbot = MagicMock()
    chatbot.config.project_name = "TEST"
    chatbot.config.title_boost_enabled = False
    chatbot.available_modules = ["TEST"]
    chatbot.reranker = None
    chatbot.hyde_llm = None
    chatbot.bm25_index = None
    return chatbot


def _run_and_capture_prompts(ui, initial_mode):
    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        return "exit"

    import builtins
    original_input = builtins.input
    builtins.input = fake_input
    try:
        ui.run_interactive(initial_mode=initial_mode)
    finally:
        builtins.input = original_input
    return prompts


def test_run_interactive_starts_in_agentic_mode_when_requested():
    ui = TerminalUI(_make_chatbot(), agentic_chatbot=MagicMock())
    prompts = _run_and_capture_prompts(ui, initial_mode="agentic")
    assert "[AGENTIC]" in prompts[0]


def test_run_interactive_defaults_to_rag_mode_when_requested():
    ui = TerminalUI(_make_chatbot(), agentic_chatbot=MagicMock())
    prompts = _run_and_capture_prompts(ui, initial_mode="rag")
    assert "[AGENTIC]" not in prompts[0]


def test_run_interactive_falls_back_to_rag_when_agentic_not_configured(capsys):
    ui = TerminalUI(_make_chatbot(), agentic_chatbot=None)
    prompts = _run_and_capture_prompts(ui, initial_mode="agentic")
    assert "[AGENTIC]" not in prompts[0]
    assert "starting in RAG mode instead" in capsys.readouterr().out

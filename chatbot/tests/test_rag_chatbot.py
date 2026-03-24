import sys
from unittest.mock import MagicMock

# Gradio may not be installed in test environments; stub it out so WebUI can be imported
if "gradio" not in sys.modules:
    sys.modules["gradio"] = MagicMock()


def test_webui_accepts_agentic_chatbot():
    """WebUI stores agentic_chatbot when provided."""
    from ui.web import WebUI
    rag = MagicMock()
    agentic = MagicMock()
    ui = WebUI(rag, agentic_chatbot=agentic)
    assert ui.agentic_chatbot is agentic


def test_webui_agentic_chatbot_defaults_to_none():
    """WebUI.agentic_chatbot is None when not provided."""
    from ui.web import WebUI
    rag = MagicMock()
    ui = WebUI(rag)
    assert ui.agentic_chatbot is None


def test_handle_message_routes_to_agentic(tmp_path):
    """_handle_message calls agentic_chatbot.ask when mode is Agentic."""
    from ui.web import WebUI
    rag = MagicMock()
    agentic = MagicMock()
    agentic.ask.return_value = {
        "answer": "agentic answer",
        "sources": [],
        "filters": {"mode": "agentic", "rounds_used": 1},
        "error": None,
    }
    ui = WebUI(rag, agentic_chatbot=agentic)
    result = ui._handle_message(
        "question", [],
        mode="Agentic",
        module_filter="All", doc_type_filter="All",
        response_style="Precise (Recommended)", deep_dive=False,
        search_depth=40, reranker_enabled=True, top_n=15, answer_length=2000,
        max_chars_per_page=8000, max_pages_per_round=3, max_pages_round2=2,
    )
    agentic.ask.assert_called_once_with(
        "question",
        max_chars_per_page=8000,
        max_pages_per_round=3,
        max_pages_round2=2,
        max_tokens=2000,
    )
    rag.ask.assert_not_called()
    assert "agentic answer" in result

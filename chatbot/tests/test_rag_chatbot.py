import sys
from unittest.mock import MagicMock

# Gradio may not be installed in test environments; stub it out so WebUI can be imported
if "gradio" not in sys.modules:
    sys.modules["gradio"] = MagicMock()

from pathlib import Path
CHATBOT_DIR = Path(__file__).parent.parent
if str(CHATBOT_DIR) not in sys.path:
    sys.path.insert(0, str(CHATBOT_DIR))

from langchain_core.documents import Document


def _bare_chatbot(bm25_index=None):
    """
    Build a DocumentationChatbot instance without running __init__ (which
    loads a real ChromaDB + LLM). Only sets the attributes _hybrid_retrieve needs.
    """
    from core.rag_chatbot import DocumentationChatbot
    bot = DocumentationChatbot.__new__(DocumentationChatbot)
    bot.bm25_index = bm25_index
    return bot


def test_hybrid_retrieve_returns_vector_docs_when_bm25_disabled():
    bot = _bare_chatbot(bm25_index=None)
    vector_docs = [Document(page_content="a", metadata={"url": "u1", "section_id": "s1", "chunk_position": "1/1"})]
    result = bot._hybrid_retrieve("query", vector_docs, k=5, module_filter=None, doc_category_filter=None)
    assert result == vector_docs


def test_hybrid_retrieve_fuses_bm25_and_vector_results():
    bm25_doc = Document(page_content="bm25 hit", metadata={"url": "u2", "section_id": "s2", "chunk_position": "1/1"})
    fake_bm25 = MagicMock()
    fake_bm25.search.return_value = [bm25_doc]

    bot = _bare_chatbot(bm25_index=fake_bm25)
    vector_docs = [Document(page_content="vector hit", metadata={"url": "u1", "section_id": "s1", "chunk_position": "1/1"})]

    result = bot._hybrid_retrieve("query", vector_docs, k=5, module_filter="SHAPER", doc_category_filter="dev")

    fake_bm25.search.assert_called_once_with("query", k=5, module_filter="SHAPER", doc_category_filter="dev")
    result_urls = {d.metadata["url"] for d in result}
    assert result_urls == {"u1", "u2"}


def test_hybrid_retrieve_noop_preserves_vector_doc_order():
    """With BM25 disabled, _hybrid_retrieve must not reorder or drop vector docs."""
    bot = _bare_chatbot(bm25_index=None)
    vector_docs = [
        Document(page_content=f"doc {i}", metadata={"url": f"u{i}", "section_id": f"s{i}", "chunk_position": "1/1"})
        for i in range(5)
    ]
    result = bot._hybrid_retrieve("q", vector_docs, k=40, module_filter=None, doc_category_filter=None)
    assert result == vector_docs


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

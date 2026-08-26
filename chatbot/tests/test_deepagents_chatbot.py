# tests/test_deepagents_chatbot.py
import json
import tempfile
from unittest.mock import MagicMock

from core.config import ChatbotConfig, AgenticConfig
from core.deepagents_chatbot import DeepAgentsChatbot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path):
    index_entries = [
        {
            "filepath": str(tmp_path / "classModelAPI__Feature.html"),
            "filename": "classModelAPI__Feature.html",
            "title": "ModelAPI_Feature Class Reference",
            "module": "SHAPER",
            "doc_category": "dev",
        },
    ]
    index_path = tmp_path / "page_index.json"
    index_path.write_text(json.dumps(index_entries))

    return ChatbotConfig(
        project_name="SHAPER",
        chromadb_path=str(tmp_path / "chromadb"),
        agentic=AgenticConfig(
            page_index_path=str(index_path),
            max_chars_per_page=8000,
            max_steps=6,
        ),
    )


def _make_bot(tmp_path, reranker_score_fn=None):
    config = _make_config(tmp_path)
    return DeepAgentsChatbot(config, vectorstore=MagicMock(), reranker_score_fn=reranker_score_fn)


# ---------------------------------------------------------------------------
# reranker_score_fn wiring
# ---------------------------------------------------------------------------

def test_constructor_stores_reranker_score_fn(tmp_path):
    score_fn = MagicMock()
    bot = _make_bot(tmp_path, reranker_score_fn=score_fn)
    assert bot._reranker_score_fn is score_fn


def test_search_pages_tool_passes_rerank_fn_through(tmp_path, monkeypatch):
    score_fn = MagicMock()
    bot = _make_bot(tmp_path, reranker_score_fn=score_fn)

    captured = {}

    def fake_search_pages(*args, **kwargs):
        captured["rerank_fn"] = kwargs.get("rerank_fn")
        return []

    import core.deepagents_chatbot as deepagents_chatbot_module
    monkeypatch.setattr(deepagents_chatbot_module, "search_pages", fake_search_pages)

    read_entries = []
    tools = bot._build_tools(8000, read_entries)
    search_pages_tool = next(t for t in tools if t.name == "search_pages_tool")
    search_pages_tool.invoke({"query": "ModelAPI_Feature"})

    assert captured["rerank_fn"] is score_fn


def test_search_pages_tool_passes_k_15(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)

    captured = {}

    def fake_search_pages(*args, **kwargs):
        captured["k"] = kwargs.get("k")
        return []

    import core.deepagents_chatbot as deepagents_chatbot_module
    monkeypatch.setattr(deepagents_chatbot_module, "search_pages", fake_search_pages)

    read_entries = []
    tools = bot._build_tools(8000, read_entries)
    search_pages_tool = next(t for t in tools if t.name == "search_pages_tool")
    search_pages_tool.invoke({"query": "ModelAPI_Feature"})

    assert captured["k"] == 15


def test_read_page_tool_looks_up_by_filename_not_filepath(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)

    import core.deepagents_chatbot as deepagents_chatbot_module
    monkeypatch.setattr(deepagents_chatbot_module, "parse_page",
                         lambda filepath, doc_category, max_chars: "page text")

    read_entries = []
    tools = bot._build_tools(8000, read_entries)
    read_page_tool = next(t for t in tools if t.name == "read_page_tool")

    known_filename = bot._page_index[0]["filename"]
    result = read_page_tool.invoke({"filename": known_filename, "doc_category": "dev"})

    assert "not a known documentation page" not in result
    assert len(read_entries) == 1
    assert read_entries[0]["filename"] == known_filename


def test_search_pages_tool_with_no_reranker_score_fn_passes_none(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path, reranker_score_fn=None)

    captured = {}

    def fake_search_pages(*args, **kwargs):
        captured["rerank_fn"] = kwargs.get("rerank_fn")
        return []

    import core.deepagents_chatbot as deepagents_chatbot_module
    monkeypatch.setattr(deepagents_chatbot_module, "search_pages", fake_search_pages)

    read_entries = []
    tools = bot._build_tools(8000, read_entries)
    search_pages_tool = next(t for t in tools if t.name == "search_pages_tool")
    search_pages_tool.invoke({"query": "ModelAPI_Feature"})

    assert captured["rerank_fn"] is None

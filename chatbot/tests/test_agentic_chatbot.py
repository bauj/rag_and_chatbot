# tests/test_agentic_chatbot.py
import json
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

from core.config import ChatbotConfig, AgenticConfig
from core.agentic_chatbot import AgenticChatbot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path, index_entries=None):
    """Build a ChatbotConfig with a real page_index.json on disk."""
    if index_entries is None:
        index_entries = [
            {
                "filepath": str(tmp_path / "classModelAPI__Feature.html"),
                "filename": "classModelAPI__Feature.html",
                "title": "ModelAPI_Feature Class Reference",
                "module": "SHAPER",
                "doc_category": "dev",
            },
            {
                "filepath": str(tmp_path / "classModelAPI__Object.html"),
                "filename": "classModelAPI__Object.html",
                "title": "ModelAPI_Object Class Reference",
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
            max_pages_per_round=3,
            max_pages_round2=2,
        ),
    )


def _make_bot(tmp_path, index_entries=None):
    """Construct AgenticChatbot with mocked LLM."""
    config = _make_config(tmp_path, index_entries)
    with patch("core.agentic_chatbot.ChatOpenAI"):
        bot = AgenticChatbot(config)
    return bot


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_missing_page_index_raises_at_construction(tmp_path):
    config = ChatbotConfig(
        project_name="test",
        chromadb_path=str(tmp_path),
        agentic=AgenticConfig(page_index_path=str(tmp_path / "missing.json")),
    )
    with pytest.raises(FileNotFoundError):
        with patch("core.agentic_chatbot.ChatOpenAI"):
            AgenticChatbot(config)


def test_no_agentic_config_raises_at_construction(tmp_path):
    config = ChatbotConfig(project_name="test", chromadb_path=str(tmp_path))
    with pytest.raises(ValueError):
        with patch("core.agentic_chatbot.ChatOpenAI"):
            AgenticChatbot(config)


# ---------------------------------------------------------------------------
# Round 1 — answer found without needing Round 2
# ---------------------------------------------------------------------------

def test_answers_in_round1(tmp_path):
    bot = _make_bot(tmp_path)

    # Mock page read
    with patch("core.agentic_chatbot.parse_page", return_value="All methods: execute, data"):
        # LLM page selection returns valid filename
        bot.llm.invoke = MagicMock(side_effect=[
            MagicMock(content='["classModelAPI__Feature.html"]'),   # page selection
            MagicMock(content="The methods are execute() and data()."),  # answer (no marker)
        ])
        result = bot.ask("What are the methods of ModelAPI_Feature?")

    assert result["error"] is None
    assert "execute" in result["answer"]
    assert result["filters"]["mode"] == "agentic"
    assert result["filters"]["rounds_used"] == 1
    assert len(result["sources"]) == 1


# ---------------------------------------------------------------------------
# Round 2 — NEED_MORE_INFO triggers refinement
# ---------------------------------------------------------------------------

def test_triggers_round2_on_need_more_info(tmp_path):
    bot = _make_bot(tmp_path)

    with patch("core.agentic_chatbot.parse_page", return_value="Some text"):
        bot.llm.invoke = MagicMock(side_effect=[
            MagicMock(content='["classModelAPI__Feature.html"]'),   # round 1 page selection
            MagicMock(content="Partial answer.\nNEED_MORE_INFO: missing object methods"),  # round 1 answer
            MagicMock(content='["classModelAPI__Object.html"]'),    # round 2 page selection
            MagicMock(content="Full answer with all methods."),     # round 2 answer
        ])
        result = bot.ask("List all methods")

    assert result["filters"]["rounds_used"] == 2
    assert result["error"] is None
    assert "NEED_MORE_INFO" not in result["answer"]


def test_need_more_info_description_passed_to_round2(tmp_path):
    bot = _make_bot(tmp_path)
    captured_prompts = []

    def capture_invoke(prompt):
        captured_prompts.append(str(prompt))
        responses = [
            MagicMock(content='["classModelAPI__Feature.html"]'),
            MagicMock(content="Partial.\nNEED_MORE_INFO: object lifecycle methods"),
            MagicMock(content='["classModelAPI__Object.html"]'),
            MagicMock(content="Complete answer."),
        ]
        return responses[len(captured_prompts) - 1]

    with patch("core.agentic_chatbot.parse_page", return_value="text"):
        bot.llm.invoke = MagicMock(side_effect=capture_invoke)
        bot.ask("List methods")

    # The round 2 prompt (3rd call) should contain the NEED_MORE_INFO description
    assert "object lifecycle methods" in captured_prompts[2]


# ---------------------------------------------------------------------------
# Round 2 edge cases
# ---------------------------------------------------------------------------

def test_round2_excludes_already_read_pages(tmp_path):
    bot = _make_bot(tmp_path)
    pages_read = []

    def fake_parse(filepath, doc_category, max_chars=8000):
        pages_read.append(filepath)
        return "content"

    with patch("core.agentic_chatbot.parse_page", side_effect=fake_parse):
        bot.llm.invoke = MagicMock(side_effect=[
            MagicMock(content='["classModelAPI__Feature.html"]'),
            MagicMock(content="Partial.\nNEED_MORE_INFO: more info"),
            MagicMock(content='["classModelAPI__Object.html"]'),
            MagicMock(content="Full answer."),
        ])
        bot.ask("List methods")

    # No filepath should appear twice
    assert len(pages_read) == len(set(pages_read))


def test_round2_no_new_results_returns_round1_answer(tmp_path):
    # Only one entry in index — round 2 will find no new pages
    single_entry_config = _make_config(tmp_path, index_entries=[
        {
            "filepath": str(tmp_path / "classModelAPI__Feature.html"),
            "filename": "classModelAPI__Feature.html",
            "title": "ModelAPI_Feature Class Reference",
            "module": "SHAPER",
            "doc_category": "dev",
        }
    ])
    with patch("core.agentic_chatbot.ChatOpenAI"):
        bot = AgenticChatbot(single_entry_config)

    with patch("core.agentic_chatbot.parse_page", return_value="content"):
        bot.llm.invoke = MagicMock(side_effect=[
            MagicMock(content='["classModelAPI__Feature.html"]'),
            MagicMock(content="Partial answer.\nNEED_MORE_INFO: more info"),
            # No more LLM calls expected
        ])
        result = bot.ask("List methods")

    assert result["filters"]["rounds_used"] == 1  # stayed at round 1
    assert "Partial answer" in result["answer"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_empty_search_results_returns_error(tmp_path):
    empty_config = _make_config(tmp_path, index_entries=[])
    with patch("core.agentic_chatbot.ChatOpenAI"):
        bot = AgenticChatbot(empty_config)
    result = bot.ask("Completely unrelated query xyz123")
    assert result["error"] is not None
    assert result["answer"] is None
    assert result["sources"] == []
    assert result["filters"]["rounds_used"] == 0


def test_parse_page_file_not_found_skipped(tmp_path):
    bot = _make_bot(tmp_path)
    bot.llm.invoke = MagicMock(side_effect=[
        MagicMock(content='["classModelAPI__Feature.html"]'),
        MagicMock(content="Answer from remaining pages."),
    ])

    def fake_parse(filepath, doc_category, max_chars=8000):
        raise FileNotFoundError(f"not found: {filepath}")

    with patch("core.agentic_chatbot.parse_page", side_effect=fake_parse):
        result = bot.ask("some question")

    # Should not crash; answer still returned (from empty context)
    assert result["error"] is None


def test_llm_selection_malformed_json_falls_back_to_first_n(tmp_path):
    bot = _make_bot(tmp_path)
    with patch("core.agentic_chatbot.parse_page", return_value="content"):
        bot.llm.invoke = MagicMock(side_effect=[
            MagicMock(content="not valid json at all"),  # malformed selection
            MagicMock(content="Fallback answer."),
        ])
        result = bot.ask("ModelAPI_Feature methods")

    assert result["error"] is None
    assert len(result["sources"]) > 0  # used fallback (first N from search)


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------

def test_returns_correct_output_format(tmp_path):
    bot = _make_bot(tmp_path)
    with patch("core.agentic_chatbot.parse_page", return_value="content"):
        bot.llm.invoke = MagicMock(side_effect=[
            MagicMock(content='["classModelAPI__Feature.html"]'),
            MagicMock(content="The answer."),
        ])
        result = bot.ask("question")

    assert set(result.keys()) == {"answer", "sources", "filters", "error"}
    assert isinstance(result["sources"], list)
    assert isinstance(result["filters"], dict)
    assert "mode" in result["filters"]
    assert "rounds_used" in result["filters"]

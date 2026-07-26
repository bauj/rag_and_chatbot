# tests/test_agentic_smol_chatbot.py
import json
import sys
from unittest.mock import MagicMock
import pytest

from core.config import ChatbotConfig, AgenticConfig
from core.agentic_smol_chatbot import AgenticSmolChatbot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path, index_entries=None):
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
            max_steps=6,
        ),
    )


def _make_bot(tmp_path, index_entries=None):
    config = _make_config(tmp_path, index_entries)
    return AgenticSmolChatbot(config)


class _FakeAgent:
    """Stand-in for smolagents.CodeAgent, driven by a run_fn(task) -> answer."""

    def __init__(self, run_fn, steps=3):
        self._run_fn = run_fn
        self.memory = MagicMock()
        self.memory.steps = [MagicMock() for _ in range(steps)]

    def run(self, task):
        return self._run_fn(task)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_no_agentic_config_raises_at_construction(tmp_path):
    config = ChatbotConfig(project_name="test", chromadb_path=str(tmp_path))
    with pytest.raises(ValueError):
        AgenticSmolChatbot(config)


def test_missing_page_index_raises_at_construction(tmp_path):
    config = ChatbotConfig(
        project_name="test",
        chromadb_path=str(tmp_path),
        agentic=AgenticConfig(page_index_path=str(tmp_path / "missing.json")),
    )
    with pytest.raises(FileNotFoundError):
        AgenticSmolChatbot(config)


def test_missing_smolagents_raises_importerror(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "smolagents", None)
    config = _make_config(tmp_path)
    with pytest.raises(ImportError):
        AgenticSmolChatbot(config)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def test_search_pages_tool_forward_returns_candidates(tmp_path):
    bot = _make_bot(tmp_path)
    tool = bot._SearchPagesToolCls(bot._page_index)
    result = tool.forward("ModelAPI_Feature")
    assert "classModelAPI__Feature.html" in result


def test_read_page_tool_rejects_unknown_filepath(tmp_path):
    bot = _make_bot(tmp_path)
    read_entries = []
    tool = bot._ReadPageToolCls(bot._entry_by_filepath, 8000, read_entries)
    result = tool.forward("/etc/passwd", "dev")
    assert "not a known documentation page" in result.lower() or "error" in result.lower()
    assert read_entries == []


def test_read_page_tool_records_entry_and_returns_content(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)
    read_entries = []
    tool = bot._ReadPageToolCls(bot._entry_by_filepath, 8000, read_entries)

    known_filepath = bot._page_index[0]["filepath"]
    monkeypatch.setattr(
        "core.agentic_smol_chatbot.parse_page",
        lambda filepath, doc_category, max_chars=8000: "The parsed content.",
    )
    result = tool.forward(known_filepath, "dev")

    assert result == "The parsed content."
    assert len(read_entries) == 1
    assert read_entries[0]["filepath"] == known_filepath
    assert read_entries[0]["title"] == "ModelAPI_Feature Class Reference"


def test_read_page_tool_handles_file_not_found(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)
    read_entries = []
    tool = bot._ReadPageToolCls(bot._entry_by_filepath, 8000, read_entries)

    known_filepath = bot._page_index[0]["filepath"]

    def fake_parse(filepath, doc_category, max_chars=8000):
        raise FileNotFoundError(f"not found: {filepath}")

    monkeypatch.setattr("core.agentic_smol_chatbot.parse_page", fake_parse)
    result = tool.forward(known_filepath, "dev")

    assert "error" in result.lower() or "not found" in result.lower()
    assert read_entries == []


# ---------------------------------------------------------------------------
# ask() orchestration
# ---------------------------------------------------------------------------

def test_ask_returns_error_dict_on_agent_exception(tmp_path):
    bot = _make_bot(tmp_path)
    bot._CodeAgent = MagicMock(
        return_value=_FakeAgent(run_fn=lambda task: (_ for _ in ()).throw(RuntimeError("boom")))
    )
    bot._OpenAIServerModel = MagicMock()

    result = bot.ask("question")

    assert result["error"] is not None
    assert result["answer"] is None
    assert result["sources"] == []
    assert result["filters"]["mode"] == "agentic-smol"


def test_ask_builds_sources_and_grounded_true_when_pages_read(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)
    known_filepath = bot._page_index[0]["filepath"]
    monkeypatch.setattr(
        "core.agentic_smol_chatbot.parse_page",
        lambda filepath, doc_category, max_chars=8000: "page text",
    )

    captured_tools = {}

    def run_fn(task):
        # Simulate the agent calling read_page during its loop.
        for tool in captured_tools["tools"]:
            if tool.name == "read_page":
                tool.forward(known_filepath, "dev")
        return "The final answer."

    def fake_code_agent(tools, model, max_steps, **kwargs):
        captured_tools["tools"] = tools
        return _FakeAgent(run_fn=run_fn, steps=4)

    bot._CodeAgent = fake_code_agent
    bot._OpenAIServerModel = MagicMock()

    result = bot.ask("question")

    assert result["error"] is None
    assert result["answer"] == "The final answer."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["filepath"] == known_filepath
    assert result["filters"]["grounded"] is True
    assert result["filters"]["steps_used"] == 4


def test_ask_grounded_false_and_empty_sources_when_no_pages_read(tmp_path):
    bot = _make_bot(tmp_path)
    bot._CodeAgent = MagicMock(return_value=_FakeAgent(run_fn=lambda task: "Answer with no reads."))
    bot._OpenAIServerModel = MagicMock()

    result = bot.ask("question")

    assert result["error"] is None
    assert result["sources"] == []
    assert result["filters"]["grounded"] is False


def test_ask_coerces_non_str_answer_to_str(tmp_path):
    class _Wrapper:
        def __str__(self):
            return "wrapped answer"

    bot = _make_bot(tmp_path)
    bot._CodeAgent = MagicMock(return_value=_FakeAgent(run_fn=lambda task: _Wrapper()))
    bot._OpenAIServerModel = MagicMock()

    result = bot.ask("question")

    assert result["answer"] == "wrapped answer"
    assert isinstance(result["answer"], str)


def test_ask_max_steps_override_passed_to_agent(tmp_path):
    bot = _make_bot(tmp_path)
    captured = {}

    def fake_code_agent(tools, model, max_steps, **kwargs):
        captured["max_steps"] = max_steps
        return _FakeAgent(run_fn=lambda task: "answer")

    bot._CodeAgent = fake_code_agent
    bot._OpenAIServerModel = MagicMock()

    bot.ask("question", max_steps=2)

    assert captured["max_steps"] == 2


def test_ask_uses_config_max_steps_by_default(tmp_path):
    bot = _make_bot(tmp_path)
    captured = {}

    def fake_code_agent(tools, model, max_steps, **kwargs):
        captured["max_steps"] = max_steps
        return _FakeAgent(run_fn=lambda task: "answer")

    bot._CodeAgent = fake_code_agent
    bot._OpenAIServerModel = MagicMock()

    bot.ask("question")

    assert captured["max_steps"] == 6


def test_returns_correct_output_format(tmp_path):
    bot = _make_bot(tmp_path)
    bot._CodeAgent = MagicMock(return_value=_FakeAgent(run_fn=lambda task: "answer"))
    bot._OpenAIServerModel = MagicMock()

    result = bot.ask("question")

    assert set(result.keys()) == {"answer", "sources", "filters", "error"}
    assert isinstance(result["sources"], list)
    assert isinstance(result["filters"], dict)
    assert "mode" in result["filters"]
    assert "steps_used" in result["filters"]
    assert "grounded" in result["filters"]

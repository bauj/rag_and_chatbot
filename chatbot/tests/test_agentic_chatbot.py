# tests/test_agentic_chatbot.py
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from core.config import ChatbotConfig, AgenticConfig
from core.agentic_chatbot import AgenticChatbot


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


def _make_bot(tmp_path, index_entries=None, bm25_index=None, reranker_score_fn=None):
    config = _make_config(tmp_path, index_entries)
    return AgenticChatbot(config, vectorstore=MagicMock(), bm25_index=bm25_index,
                          reranker_score_fn=reranker_score_fn)


class _FakeMsg:
    """Stand-in for a LangChain AIMessage in the run's message log."""

    def __init__(self, content, usage_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata


class _FakeDeepAgent:
    """Stand-in for a deepagents/LangGraph agent.

    respond_fn(input_messages) -> list of messages the run appended. invoke()
    returns {"messages": input + appended}, mirroring LangGraph's add_messages.
    """

    def __init__(self, respond_fn):
        self._respond_fn = respond_fn
        self.captured_recursion_limit = None

    def invoke(self, payload, config=None):
        if config:
            self.captured_recursion_limit = config.get("recursion_limit")
        appended = self._respond_fn(payload["messages"])
        return {"messages": list(payload["messages"]) + list(appended)}

    def stream(self, payload, config=None, stream_mode=None):
        if config:
            self.captured_recursion_limit = config.get("recursion_limit")
        for msg in self._respond_fn(payload["messages"]):
            yield {"agent": {"messages": [msg]}}


def _install_fake_agent(bot, respond_fn, capture=None):
    """Wire bot so ask() builds a _FakeDeepAgent and a MagicMock model."""
    bot._ChatOpenAI = lambda **kwargs: MagicMock()

    def factory(model, tools, system_prompt):
        agent = _FakeDeepAgent(respond_fn)
        if capture is not None:
            capture["tools"] = tools
            capture["agent"] = agent
            capture["system_prompt"] = system_prompt
        return agent

    bot._create_deep_agent = factory


def _read_tool(tools):
    return next(t for t in tools if t.name == "read_page_tool")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_no_agentic_config_raises_at_construction(tmp_path):
    config = ChatbotConfig(project_name="test", chromadb_path=str(tmp_path))
    with pytest.raises(ValueError):
        AgenticChatbot(config, vectorstore=MagicMock())


def test_missing_page_index_raises_at_construction(tmp_path):
    config = ChatbotConfig(
        project_name="test",
        chromadb_path=str(tmp_path),
        agentic=AgenticConfig(page_index_path=str(tmp_path / "missing.json")),
    )
    with pytest.raises(FileNotFoundError):
        AgenticChatbot(config, vectorstore=MagicMock())


def test_missing_deepagents_raises_importerror(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "deepagents", None)
    config = _make_config(tmp_path)
    with pytest.raises(ImportError):
        AgenticChatbot(config, vectorstore=MagicMock())


# ---------------------------------------------------------------------------
# Tools
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

    monkeypatch.setattr("core.agentic_chatbot.search_pages", fake_search_pages)

    tools = bot._build_tools(8000, [])
    search_tool = next(t for t in tools if t.name == "search_pages_tool")
    search_tool.invoke({"query": "ModelAPI_Feature"})

    assert captured["rerank_fn"] is score_fn


def test_search_pages_tool_passes_k_15(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)

    captured = {}

    def fake_search_pages(*args, **kwargs):
        captured["k"] = kwargs.get("k")
        return []

    monkeypatch.setattr("core.agentic_chatbot.search_pages", fake_search_pages)

    tools = bot._build_tools(8000, [])
    search_tool = next(t for t in tools if t.name == "search_pages_tool")
    search_tool.invoke({"query": "ModelAPI_Feature"})

    assert captured["k"] == 15


def test_search_pages_tool_with_no_reranker_score_fn_passes_none(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path, reranker_score_fn=None)

    captured = {}

    def fake_search_pages(*args, **kwargs):
        captured["rerank_fn"] = kwargs.get("rerank_fn")
        return []

    monkeypatch.setattr("core.agentic_chatbot.search_pages", fake_search_pages)

    tools = bot._build_tools(8000, [])
    search_tool = next(t for t in tools if t.name == "search_pages_tool")
    search_tool.invoke({"query": "ModelAPI_Feature"})

    assert captured["rerank_fn"] is None


def test_read_page_tool_looks_up_by_filename_and_records_entry(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)
    monkeypatch.setattr("core.agentic_chatbot.parse_page",
                        lambda filepath, doc_category, max_chars: "page text")

    read_entries = []
    tools = bot._build_tools(8000, read_entries)
    known_filename = bot._page_index[0]["filename"]
    result = _read_tool(tools).invoke({"filename": known_filename, "doc_category": "dev"})

    assert "not a known documentation page" not in result
    assert len(read_entries) == 1
    assert read_entries[0]["filename"] == known_filename


def test_read_page_tool_rejects_unknown_filename(tmp_path):
    bot = _make_bot(tmp_path)
    read_entries = []
    tools = bot._build_tools(8000, read_entries)
    result = _read_tool(tools).invoke({"filename": "/etc/passwd", "doc_category": "dev"})

    assert "not a known documentation page" in result.lower()
    assert read_entries == []


def test_read_page_tool_handles_file_not_found(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)

    def fake_parse(filepath, doc_category, max_chars):
        raise FileNotFoundError(f"not found: {filepath}")

    monkeypatch.setattr("core.agentic_chatbot.parse_page", fake_parse)
    read_entries = []
    tools = bot._build_tools(8000, read_entries)
    known_filename = bot._page_index[0]["filename"]
    result = _read_tool(tools).invoke({"filename": known_filename, "doc_category": "dev"})

    assert "not found" in result.lower() or "error" in result.lower()
    assert read_entries == []


# ---------------------------------------------------------------------------
# ask() orchestration
# ---------------------------------------------------------------------------

def test_ask_degrades_gracefully_on_graph_recursion_error(tmp_path, monkeypatch):
    """deepagents/LangGraph raises GraphRecursionError (a hard crash, answer=None)
    when the step budget is exhausted. Unlike a real error, this should return a
    best-effort answer built from whatever pages were read before the budget ran
    out — error stays None, degraded_answer True."""
    from langgraph.errors import GraphRecursionError

    bot = _make_bot(tmp_path)
    known_filename = bot._page_index[0]["filename"]
    monkeypatch.setattr("core.agentic_chatbot.parse_page",
                        lambda filepath, doc_category, max_chars: "page text the agent read")

    capture = {}

    def respond_fn(_messages):
        # agent reads one page, then blows the recursion budget
        _read_tool(capture["tools"]).invoke({"filename": known_filename, "doc_category": "dev"})
        raise GraphRecursionError("Recursion limit of 14 reached")

    _install_fake_agent(bot, respond_fn, capture)

    result = bot.ask("question")

    assert result["error"] is None
    assert result["answer"] is not None
    assert result["filters"]["mode"] == "agentic"
    assert result["filters"]["degraded_answer"] is True
    assert result["filters"]["grounded"] is True
    assert len(result["sources"]) == 1
    assert result["sources"][0]["filename"] == known_filename
    assert result["sources"][0]["content"] == "page text the agent read"


def test_ask_graph_recursion_error_with_no_pages_read_still_degrades(tmp_path):
    from langgraph.errors import GraphRecursionError

    bot = _make_bot(tmp_path)

    def respond_fn(_messages):
        raise GraphRecursionError("Recursion limit reached")

    _install_fake_agent(bot, respond_fn)

    result = bot.ask("question")

    assert result["error"] is None
    assert result["answer"] is not None
    assert result["filters"]["degraded_answer"] is True
    assert result["filters"]["grounded"] is False
    assert result["sources"] == []


def test_ask_returns_error_dict_on_agent_exception(tmp_path):
    bot = _make_bot(tmp_path)

    def respond_fn(_messages):
        raise RuntimeError("boom")

    _install_fake_agent(bot, respond_fn)

    result = bot.ask("question")

    assert result["error"] is not None
    assert result["answer"] is None
    assert result["sources"] == []
    assert result["filters"]["mode"] == "agentic"


def test_ask_builds_sources_and_grounded_true_when_pages_read(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)
    known_filename = bot._page_index[0]["filename"]
    monkeypatch.setattr("core.agentic_chatbot.parse_page",
                        lambda filepath, doc_category, max_chars: "page text")

    capture = {}

    def respond_fn(_messages):
        _read_tool(capture["tools"]).invoke({"filename": known_filename, "doc_category": "dev"})
        return [_FakeMsg("The final answer.")]

    _install_fake_agent(bot, respond_fn, capture)

    result = bot.ask("question")

    assert result["error"] is None
    assert result["answer"] == "The final answer."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["filename"] == known_filename
    assert result["filters"]["grounded"] is True


def test_ask_sources_carry_the_parsed_page_content(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)
    known_filename = bot._page_index[0]["filename"]
    monkeypatch.setattr("core.agentic_chatbot.parse_page",
                        lambda filepath, doc_category, max_chars: "The actual page text.")

    capture = {}

    def respond_fn(_messages):
        _read_tool(capture["tools"]).invoke({"filename": known_filename, "doc_category": "dev"})
        return [_FakeMsg("The final answer.")]

    _install_fake_agent(bot, respond_fn, capture)

    result = bot.ask("question")

    assert result["sources"][0]["content"] == "The actual page text."


def test_ask_grounded_false_and_empty_sources_when_no_pages_read(tmp_path):
    bot = _make_bot(tmp_path)
    _install_fake_agent(bot, lambda _m: [_FakeMsg("Answer with no reads.")])

    result = bot.ask("question")

    assert result["error"] is None
    assert result["sources"] == []
    assert result["filters"]["grounded"] is False


def test_ask_steps_used_is_message_count(tmp_path):
    bot = _make_bot(tmp_path)
    _install_fake_agent(bot, lambda _m: [_FakeMsg("a"), _FakeMsg("b")])

    result = bot.ask("question")

    # 1 input message + 2 appended
    assert result["filters"]["steps_used"] == 3


def test_ask_max_steps_override_maps_to_recursion_limit(tmp_path):
    bot = _make_bot(tmp_path)
    capture = {}
    _install_fake_agent(bot, lambda _m: [_FakeMsg("answer")], capture)

    bot.ask("question", max_steps=2)

    assert capture["agent"].captured_recursion_limit == 2 * 2 + 2


def test_ask_uses_config_max_steps_by_default(tmp_path):
    bot = _make_bot(tmp_path)
    capture = {}
    _install_fake_agent(bot, lambda _m: [_FakeMsg("answer")], capture)

    bot.ask("question")

    assert capture["agent"].captured_recursion_limit == 2 * 6 + 2


def test_ask_ambiguous_overload_flag_alone_does_not_degrade_after_retry(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)
    calls = {"n": 0}

    def fake_check_grounding(answer, observations, llm_classify_fn=None):
        calls["n"] += 1
        return [{"call": "addBox", "kind": "ambiguous_overload", "reason": "multiple documented signatures"}]

    monkeypatch.setattr("core.agentic_chatbot.check_grounding", fake_check_grounding)
    _install_fake_agent(bot, lambda _m: [_FakeMsg("corrected code answer")])

    result = bot.ask("question")

    assert calls["n"] == 2  # initial check + one retry check
    assert result["filters"]["degraded_answer"] is False
    assert result["answer"] == "corrected code answer"
    assert result["filters"]["grounding_retries_used"] == 1


def test_ask_unknown_name_flag_surviving_retry_still_degrades(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)

    def fake_check_grounding(answer, observations, llm_classify_fn=None):
        return [{"call": "invented", "kind": "unknown_name", "reason": "unknown name"}]

    monkeypatch.setattr("core.agentic_chatbot.check_grounding", fake_check_grounding)
    _install_fake_agent(bot, lambda _m: [_FakeMsg("still bad answer")])

    result = bot.ask("question")

    assert result["filters"]["degraded_answer"] is True
    assert result["answer"] != "still bad answer"


def test_ask_normal_answer_is_not_flagged_degraded(tmp_path):
    bot = _make_bot(tmp_path)
    _install_fake_agent(bot, lambda _m: [_FakeMsg("A clean prose answer.")])

    result = bot.ask("question")

    assert result["answer"] == "A clean prose answer."
    assert result["filters"]["degraded_answer"] is False


def test_ask_token_usage_summed_from_message_log(tmp_path):
    bot = _make_bot(tmp_path)
    _install_fake_agent(bot, lambda _m: [
        _FakeMsg("partial", usage_metadata={"input_tokens": 100, "output_tokens": 20}),
        _FakeMsg("final", usage_metadata={"input_tokens": 150, "output_tokens": 30}),
    ])

    result = bot.ask("question")

    assert result["filters"]["token_usage"]["total_tokens"] == 300


def test_returns_correct_output_format(tmp_path):
    bot = _make_bot(tmp_path)
    _install_fake_agent(bot, lambda _m: [_FakeMsg("answer")])

    result = bot.ask("question")

    assert set(result.keys()) == {"answer", "sources", "filters", "error"}
    assert isinstance(result["sources"], list)
    assert isinstance(result["filters"], dict)
    for key in ("mode", "steps_used", "grounded", "degraded_answer"):
        assert key in result["filters"]


# ---------------------------------------------------------------------------
# Debug trace writing
# ---------------------------------------------------------------------------

def test_ask_writes_debug_trace_when_debug_true(tmp_path):
    from core.debug_trace_writer import load_traces

    trace_dir = tmp_path / "traces"
    config = _make_config(tmp_path)
    config.agentic.debug_dir = str(trace_dir)
    bot = AgenticChatbot(config, vectorstore=MagicMock())
    _install_fake_agent(bot, lambda _m: [_FakeMsg("the answer")])

    bot.ask("what is a sketch?", debug=True)

    traces = load_traces(trace_dir)
    assert len(traces) == 1
    assert traces[0]["question"] == "what is a sketch?"
    assert any(m["content"] == "the answer" for m in traces[0]["messages"])


def test_ask_does_not_write_debug_trace_when_debug_false(tmp_path):
    from core.debug_trace_writer import load_traces

    trace_dir = tmp_path / "traces"
    config = _make_config(tmp_path)
    config.agentic.debug_dir = str(trace_dir)
    bot = AgenticChatbot(config, vectorstore=MagicMock())
    _install_fake_agent(bot, lambda _m: [_FakeMsg("the answer")])

    bot.ask("q", debug=False)

    assert load_traces(trace_dir) == []


# ---------------------------------------------------------------------------
# Parameter parity: temperature, max_tokens, ssl_cert_file
# ---------------------------------------------------------------------------

def test_ask_passes_temperature_and_max_tokens_to_model(tmp_path):
    bot = _make_bot(tmp_path)
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    bot._ChatOpenAI = fake_chat_openai
    bot._create_deep_agent = lambda model, tools, system_prompt: _FakeDeepAgent(lambda _m: [_FakeMsg("answer")])

    bot.ask("question", temperature=0.5, max_tokens=1234)

    assert captured["temperature"] == 0.5
    assert captured["max_tokens"] == 1234


def test_ask_defaults_temperature_and_max_tokens_from_config(tmp_path):
    config = _make_config(tmp_path)
    config.temperature = 0.3
    config.max_tokens = 999
    bot = AgenticChatbot(config, vectorstore=MagicMock())
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    bot._ChatOpenAI = fake_chat_openai
    bot._create_deep_agent = lambda model, tools, system_prompt: _FakeDeepAgent(lambda _m: [_FakeMsg("answer")])

    bot.ask("question")

    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 999


def test_ask_wires_ssl_cert_file_into_http_client(tmp_path):
    system_bundles = [
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
        "/etc/ssl/certs/ca-certificates.crt",
    ]
    bundle = next((p for p in system_bundles if Path(p).exists()), None)
    if bundle is None:
        pytest.skip("no system CA bundle found to use as a fake cert file")
    cert_path = tmp_path / "cert.pem"
    cert_path.write_text(Path(bundle).read_text())

    config = _make_config(tmp_path)
    config.llm.ssl_cert_file = str(cert_path)
    bot = AgenticChatbot(config, vectorstore=MagicMock())
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    bot._ChatOpenAI = fake_chat_openai
    bot._create_deep_agent = lambda model, tools, system_prompt: _FakeDeepAgent(lambda _m: [_FakeMsg("answer")])

    bot.ask("question")

    assert "http_client" in captured


def test_ask_missing_ssl_cert_file_raises(tmp_path):
    config = _make_config(tmp_path)
    config.llm.ssl_cert_file = str(tmp_path / "missing_cert.pem")
    bot = AgenticChatbot(config, vectorstore=MagicMock())
    bot._ChatOpenAI = MagicMock()
    bot._create_deep_agent = MagicMock()

    with pytest.raises(FileNotFoundError):
        bot.ask("question")

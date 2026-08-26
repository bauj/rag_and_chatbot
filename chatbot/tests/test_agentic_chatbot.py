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


def _make_bot(tmp_path, index_entries=None, bm25_index=None):
    config = _make_config(tmp_path, index_entries)
    return AgenticChatbot(config, vectorstore=MagicMock(), bm25_index=bm25_index)


class _FakeAgent:
    """Stand-in for smolagents.CodeAgent, driven by a run_fn(task) -> answer."""

    def __init__(self, run_fn, steps=3):
        self._run_fn = run_fn
        self.memory = MagicMock()
        self.memory.steps = [MagicMock() for _ in range(steps)]

    def run(self, task, reset=True):
        return self._run_fn(task)


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


def test_missing_smolagents_raises_importerror(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "smolagents", None)
    config = _make_config(tmp_path)
    with pytest.raises(ImportError):
        AgenticChatbot(config, vectorstore=MagicMock())


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def test_search_pages_tool_forward_returns_candidates(tmp_path):
    bot = _make_bot(tmp_path)

    class _FakeDoc:
        def __init__(self, filename):
            self.metadata = {"url": f"https://example.com/{filename}"}
            self.page_content = filename

    bot._vectorstore.similarity_search = lambda query, k: [_FakeDoc("classModelAPI__Feature.html")]
    tool = bot._SearchPagesToolCls(bot._vectorstore, bot._page_index)
    result = tool.forward("ModelAPI_Feature")
    assert "classModelAPI__Feature.html" in result


def test_constructor_stores_reranker_score_fn(tmp_path):
    score_fn = MagicMock()
    config = _make_config(tmp_path)
    bot = AgenticChatbot(config, vectorstore=MagicMock(), reranker_score_fn=score_fn)
    assert bot._reranker_score_fn is score_fn


def test_search_pages_tool_forward_passes_rerank_fn_through(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)
    score_fn = MagicMock()

    captured = {}

    def fake_search_pages(*args, **kwargs):
        captured["rerank_fn"] = kwargs.get("rerank_fn")
        return []

    import core.agentic_chatbot as agentic_chatbot_module
    monkeypatch.setattr(agentic_chatbot_module, "search_pages", fake_search_pages)

    tool = bot._SearchPagesToolCls(bot._vectorstore, bot._page_index, rerank_fn=score_fn)
    tool.forward("ModelAPI_Feature")
    assert captured["rerank_fn"] is score_fn


def test_search_pages_tool_forward_passes_k_15(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)

    captured = {}

    def fake_search_pages(*args, **kwargs):
        captured["k"] = kwargs.get("k")
        return []

    import core.agentic_chatbot as agentic_chatbot_module
    monkeypatch.setattr(agentic_chatbot_module, "search_pages", fake_search_pages)

    tool = bot._SearchPagesToolCls(bot._vectorstore, bot._page_index)
    tool.forward("ModelAPI_Feature")
    assert captured["k"] == 15


def test_ask_passes_reranker_score_fn_to_search_tool(tmp_path, monkeypatch):
    score_fn = MagicMock()
    config = _make_config(tmp_path)
    bot = AgenticChatbot(config, vectorstore=MagicMock(), reranker_score_fn=score_fn)

    captured = {}
    original_cls = bot._SearchPagesToolCls

    def capturing_cls(*args, **kwargs):
        captured["rerank_fn"] = kwargs.get("rerank_fn")
        return original_cls(*args, **kwargs)

    bot._SearchPagesToolCls = capturing_cls
    bot._OpenAIServerModel = MagicMock()
    bot._CodeAgent = lambda **kwargs: _FakeAgent(lambda task: "answer")

    bot.ask("What is a Sketch?")
    assert captured["rerank_fn"] is score_fn


def test_search_pages_tool_forward_omits_full_filepath(tmp_path):
    """Candidate lines show filename, not the full absolute filepath — the
    filepath is what was bloating search_pages_tool's output (task: cut
    agentic mode's per-turn token cost)."""
    bot = _make_bot(tmp_path)

    class _FakeDoc:
        def __init__(self, filename):
            self.metadata = {"url": f"https://example.com/{filename}"}
            self.page_content = filename

    bot._vectorstore.similarity_search = lambda query, k: [_FakeDoc("classModelAPI__Feature.html")]
    tool = bot._SearchPagesToolCls(bot._vectorstore, bot._page_index)
    result = tool.forward("ModelAPI_Feature")

    full_filepath = bot._page_index[0]["filepath"]
    assert full_filepath not in result
    assert "classModelAPI__Feature.html" in result


def test_read_page_tool_rejects_unknown_filename(tmp_path):
    bot = _make_bot(tmp_path)
    read_entries = []
    tool = bot._ReadPageToolCls(bot._entry_by_filename, 8000, read_entries)
    result = tool.forward("/etc/passwd", "dev")
    assert "not a known documentation page" in result.lower() or "error" in result.lower()
    assert read_entries == []


def test_read_page_tool_records_entry_and_returns_content(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)
    read_entries = []
    tool = bot._ReadPageToolCls(bot._entry_by_filename, 8000, read_entries)

    known_filename = bot._page_index[0]["filename"]
    monkeypatch.setattr(
        "core.agentic_chatbot.parse_page",
        lambda filepath, doc_category, max_chars=8000: "The parsed content.",
    )
    result = tool.forward(known_filename, "dev")

    assert result == "The parsed content."
    assert len(read_entries) == 1
    assert read_entries[0]["filename"] == known_filename
    assert read_entries[0]["title"] == "ModelAPI_Feature Class Reference"


def test_read_page_tool_handles_file_not_found(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)
    read_entries = []
    tool = bot._ReadPageToolCls(bot._entry_by_filename, 8000, read_entries)

    known_filename = bot._page_index[0]["filename"]

    def fake_parse(filepath, doc_category, max_chars=8000):
        raise FileNotFoundError(f"not found: {filepath}")

    monkeypatch.setattr("core.agentic_chatbot.parse_page", fake_parse)
    result = tool.forward(known_filename, "dev")

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
    assert result["filters"]["mode"] == "agentic"


def test_ask_builds_sources_and_grounded_true_when_pages_read(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)
    known_filename = bot._page_index[0]["filename"]
    monkeypatch.setattr(
        "core.agentic_chatbot.parse_page",
        lambda filepath, doc_category, max_chars=8000: "page text",
    )

    captured_tools = {}

    def run_fn(task):
        # Simulate the agent calling read_page during its loop.
        for tool in captured_tools["tools"]:
            if tool.name == "read_page":
                tool.forward(known_filename, "dev")
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
    assert result["sources"][0]["filename"] == known_filename
    assert result["filters"]["grounded"] is True
    assert result["filters"]["steps_used"] == 4


def test_ask_sources_carry_the_parsed_page_content(tmp_path, monkeypatch):
    """
    #114: the evaluator's groundedness/retrieval_relevance grading reads
    source["content"] to build the FACTS block. Without it every agentic
    answer looks ungrounded regardless of how good the retrieval was.
    """
    bot = _make_bot(tmp_path)
    known_filename = bot._page_index[0]["filename"]
    monkeypatch.setattr(
        "core.agentic_chatbot.parse_page",
        lambda filepath, doc_category, max_chars=8000: "The actual page text the agent read.",
    )

    captured_tools = {}

    def run_fn(task):
        for tool in captured_tools["tools"]:
            if tool.name == "read_page":
                tool.forward(known_filename, "dev")
        return "The final answer."

    def fake_code_agent(tools, model, max_steps, **kwargs):
        captured_tools["tools"] = tools
        return _FakeAgent(run_fn=run_fn, steps=4)

    bot._CodeAgent = fake_code_agent
    bot._OpenAIServerModel = MagicMock()

    result = bot.ask("question")

    assert result["sources"][0]["content"] == "The actual page text the agent read."


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


def test_ask_replaces_leaked_code_answer_with_fallback_when_sources_read(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)
    known_filename = bot._page_index[0]["filename"]
    monkeypatch.setattr(
        "core.agentic_chatbot.parse_page",
        lambda filepath, doc_category, max_chars=8000: "page text",
    )

    captured_tools = {}

    def run_fn(task):
        for tool in captured_tools["tools"]:
            if tool.name == "read_page":
                tool.forward(known_filename, "dev")
        return '<code>\nread_page(filepath="foo.html", doc_category="dev")\n</code>'

    def fake_code_agent(tools, model, max_steps, **kwargs):
        captured_tools["tools"] = tools
        return _FakeAgent(run_fn=run_fn, steps=8)

    bot._CodeAgent = fake_code_agent
    bot._OpenAIServerModel = MagicMock()

    result = bot.ask("question")

    assert result["error"] is None
    assert "<code>" not in result["answer"]
    assert "read_page(" not in result["answer"]
    assert "ModelAPI_Feature Class Reference" in result["answer"]
    assert result["filters"]["degraded_answer"] is True


def test_ask_replaces_leaked_code_answer_with_fallback_when_no_sources_read(tmp_path):
    bot = _make_bot(tmp_path)
    bot._CodeAgent = MagicMock(
        return_value=_FakeAgent(
            run_fn=lambda task: "Calling tools:\n[{'id': 'call_7', 'type': 'function'}]",
            steps=8,
        )
    )
    bot._OpenAIServerModel = MagicMock()

    result = bot.ask("question")

    assert result["error"] is None
    assert "Calling tools:" not in result["answer"]
    assert result["filters"]["degraded_answer"] is True
    assert result["filters"]["grounded"] is False


def test_ask_ambiguous_overload_flag_alone_does_not_degrade_after_retry(tmp_path, monkeypatch):
    # An ambiguous_overload flag can never clear on its own (see
    # grounding_judge.check_grounding's docstring) — it must not, by itself,
    # cause an otherwise-good retried answer to be discarded.
    bot = _make_bot(tmp_path)
    calls = {"n": 0}

    def fake_check_grounding(answer, agent, llm_classify_fn=None):
        calls["n"] += 1
        return [{"call": "addBox", "kind": "ambiguous_overload", "reason": "multiple documented signatures"}]

    monkeypatch.setattr("core.agentic_chatbot.check_grounding", fake_check_grounding)
    bot._CodeAgent = MagicMock(return_value=_FakeAgent(run_fn=lambda task: "corrected code answer"))
    bot._OpenAIServerModel = MagicMock()

    result = bot.ask("question")

    assert calls["n"] == 2  # initial check + one retry check
    assert result["filters"]["degraded_answer"] is False
    assert result["answer"] == "corrected code answer"
    assert result["filters"]["grounding_retries_used"] == 1


def test_ask_unknown_name_flag_surviving_retry_still_degrades(tmp_path, monkeypatch):
    bot = _make_bot(tmp_path)

    def fake_check_grounding(answer, agent, llm_classify_fn=None):
        return [{"call": "invented", "kind": "unknown_name", "reason": "unknown name"}]

    monkeypatch.setattr("core.agentic_chatbot.check_grounding", fake_check_grounding)
    bot._CodeAgent = MagicMock(return_value=_FakeAgent(run_fn=lambda task: "still bad answer"))
    bot._OpenAIServerModel = MagicMock()

    result = bot.ask("question")

    assert result["filters"]["degraded_answer"] is True
    assert result["answer"] != "still bad answer"


def test_ask_normal_answer_is_not_flagged_degraded(tmp_path):
    bot = _make_bot(tmp_path)
    bot._CodeAgent = MagicMock(return_value=_FakeAgent(run_fn=lambda task: "A clean prose answer."))
    bot._OpenAIServerModel = MagicMock()

    result = bot.ask("question")

    assert result["answer"] == "A clean prose answer."
    assert result["filters"]["degraded_answer"] is False


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
    assert "degraded_answer" in result["filters"]


# ---------------------------------------------------------------------------
# Parameter parity: temperature, max_tokens, ssl_cert_file
# ---------------------------------------------------------------------------

def test_ask_passes_temperature_and_max_tokens_to_model(tmp_path):
    bot = _make_bot(tmp_path)
    captured = {}

    def fake_openai_server_model(model_id, api_base, api_key, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    def fake_code_agent(tools, model, max_steps, **kwargs):
        return _FakeAgent(run_fn=lambda task: "answer")

    bot._CodeAgent = fake_code_agent
    bot._OpenAIServerModel = fake_openai_server_model

    bot.ask("question", temperature=0.5, max_tokens=1234)

    assert captured["temperature"] == 0.5
    assert captured["max_tokens"] == 1234


def test_ask_defaults_temperature_and_max_tokens_from_config(tmp_path):
    config = _make_config(tmp_path)
    config.temperature = 0.3
    config.max_tokens = 999
    bot = AgenticChatbot(config, vectorstore=MagicMock())
    captured = {}

    def fake_openai_server_model(model_id, api_base, api_key, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    def fake_code_agent(tools, model, max_steps, **kwargs):
        return _FakeAgent(run_fn=lambda task: "answer")

    bot._CodeAgent = fake_code_agent
    bot._OpenAIServerModel = fake_openai_server_model

    bot.ask("question")

    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 999


def test_ask_wires_ssl_cert_file_into_client_kwargs(tmp_path):
    # httpx.Client(verify=<path>) actually parses the file, so it must be a
    # real PEM bundle — reuse the system CA bundle rather than pull in a new
    # test dependency (e.g. `cryptography`) just to mint a throwaway cert.
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

    def fake_openai_server_model(model_id, api_base, api_key, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    def fake_code_agent(tools, model, max_steps, **kwargs):
        return _FakeAgent(run_fn=lambda task: "answer")

    bot._CodeAgent = fake_code_agent
    bot._OpenAIServerModel = fake_openai_server_model

    bot.ask("question")

    assert "client_kwargs" in captured
    assert "http_client" in captured["client_kwargs"]


def test_ask_missing_ssl_cert_file_raises(tmp_path):
    config = _make_config(tmp_path)
    config.llm.ssl_cert_file = str(tmp_path / "missing_cert.pem")
    bot = AgenticChatbot(config, vectorstore=MagicMock())
    bot._CodeAgent = MagicMock()
    bot._OpenAIServerModel = MagicMock()

    with pytest.raises(FileNotFoundError):
        bot.ask("question")

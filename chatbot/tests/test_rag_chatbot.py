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
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate


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


def test_hybrid_retrieve_skips_bm25_when_runtime_disabled():
    """A loaded BM25 index must still be bypassed when the UI toggle is off."""
    fake_bm25 = MagicMock()
    bot = _bare_chatbot(bm25_index=fake_bm25)
    vector_docs = [Document(page_content="a", metadata={"url": "u1", "section_id": "s1", "chunk_position": "1/1"})]

    result = bot._hybrid_retrieve(
        "query", vector_docs, k=5, module_filter=None,
        doc_category_filter=None, bm25_enabled=False,
    )

    fake_bm25.search.assert_not_called()
    assert result == vector_docs


# ---------------------------------------------------------------------------
# retrieve_k: per-channel retrieval depth decoupled from the reranker pool (k)
#
# reciprocal_rank_fusion([dense_k, bm25_k], k) merges two k-length lists and
# truncates the result back to k, so each channel effectively contributes only
# ~k/2 of its candidates to the fused pool. A doc ranked outside that in only
# one channel never reaches the pool even when a deeper look would find it —
# measured on the real corpus: a target chunk sat at dense rank 46-47 / BM25
# rank 44, inside both channels' top-60/80, and still missed the fused pool
# when retrieve depth == pool size. retrieve_k lets each channel search deeper
# without growing k (the reranker cost driver).
# ---------------------------------------------------------------------------

def test_hybrid_retrieve_uses_retrieve_k_for_bm25_search_depth():
    """retrieve_k, not k, sets how deep BM25 is asked to search."""
    fake_bm25 = MagicMock()
    fake_bm25.search.return_value = []
    bot = _bare_chatbot(bm25_index=fake_bm25)
    vector_docs = [Document(page_content="v", metadata={"url": "u1", "section_id": "s1", "chunk_position": "1/1"})]

    bot._hybrid_retrieve(
        "query", vector_docs, k=5, module_filter=None, doc_category_filter=None, retrieve_k=80,
    )

    fake_bm25.search.assert_called_once_with("query", k=80, module_filter=None, doc_category_filter=None)


def test_hybrid_retrieve_retrieve_k_defaults_to_k_when_unset():
    """Backwards compatibility: omitting retrieve_k reproduces today's behaviour."""
    fake_bm25 = MagicMock()
    fake_bm25.search.return_value = []
    bot = _bare_chatbot(bm25_index=fake_bm25)
    vector_docs = [Document(page_content="v", metadata={"url": "u1", "section_id": "s1", "chunk_position": "1/1"})]

    bot._hybrid_retrieve("query", vector_docs, k=5, module_filter=None, doc_category_filter=None)

    fake_bm25.search.assert_called_once_with("query", k=5, module_filter=None, doc_category_filter=None)


def test_hybrid_retrieve_still_truncates_fused_pool_to_k_regardless_of_retrieve_k():
    """Deeper per-channel search must not inflate the reranker pool size."""
    fake_bm25 = MagicMock()
    fake_bm25.search.return_value = [
        Document(page_content=f"b{i}", metadata={"url": f"b{i}", "section_id": f"b{i}", "chunk_position": "1/1"})
        for i in range(80)
    ]
    bot = _bare_chatbot(bm25_index=fake_bm25)
    vector_docs = [
        Document(page_content=f"v{i}", metadata={"url": f"v{i}", "section_id": f"v{i}", "chunk_position": "1/1"})
        for i in range(80)
    ]

    result = bot._hybrid_retrieve(
        "query", vector_docs, k=5, module_filter=None, doc_category_filter=None, retrieve_k=80,
    )

    assert len(result) == 5


def test_hybrid_retrieve_retrieve_k_recovers_doc_beyond_pool_k():
    """
    The measured recall leak, reproduced: a doc ranked 46th (index 45) in both
    the dense and BM25 channels reaches the fused pool when each channel is
    searched to retrieve_k=80, but is invisible to both channels — and so to
    fusion — when retrieval depth is left at the pool size (k=40), today's
    behaviour and the backwards-compatible default when retrieve_k is unset.
    """
    target = Document(page_content="target", metadata={"url": "target", "section_id": "target", "chunk_position": "1/1"})

    def _filler(prefix, n):
        return [
            Document(page_content=f"{prefix}{i}", metadata={"url": f"{prefix}{i}", "section_id": f"{prefix}{i}", "chunk_position": "1/1"})
            for i in range(n)
        ]

    # target sits at index 45 (rank 46) in both channels' full 80-deep ranking.
    dense_full = _filler("v", 45) + [target] + _filler("v_tail", 34)
    bm25_full = _filler("b", 45) + [target] + _filler("b_tail", 34)

    fake_bm25 = MagicMock()
    fake_bm25.search.side_effect = lambda query, k, module_filter=None, doc_category_filter=None: bm25_full[:k]
    bot = _bare_chatbot(bm25_index=fake_bm25)

    # New behaviour: vector retriever and BM25 both searched to retrieve_k=80.
    result_deep = bot._hybrid_retrieve(
        "query", dense_full, k=40, module_filter=None, doc_category_filter=None, retrieve_k=80,
    )
    assert "target" in {d.metadata["url"] for d in result_deep}

    # Old / backwards-compatible behaviour: both channels limited to the pool size (40).
    result_shallow = bot._hybrid_retrieve(
        "query", dense_full[:40], k=40, module_filter=None, doc_category_filter=None,
    )
    assert "target" not in {d.metadata["url"] for d in result_shallow}


# ---------------------------------------------------------------------------
# Title/identifier channel: a third RRF list, gated by title_boost_enabled
#
# A question naming a class is an entity lookup, not a semantic search: the
# discriminating identifier is one token among ~150 in a chunk's body (diluted
# further by every subclass page that mentions the parent class) but is the
# whole of the page's title. RRF only looks at rank, so even a short,
# high-precision title list can out-rank a body/dense channel where the same
# page is buried at rank 40+.
# ---------------------------------------------------------------------------

def test_hybrid_retrieve_skips_title_channel_by_default():
    fake_bm25 = MagicMock()
    fake_bm25.search.return_value = []
    bot = _bare_chatbot(bm25_index=fake_bm25)
    vector_docs = [Document(page_content="v", metadata={"url": "u1", "section_id": "s1", "chunk_position": "1/1"})]

    bot._hybrid_retrieve("query", vector_docs, k=5, module_filter=None, doc_category_filter=None)

    fake_bm25.search_titles.assert_not_called()


def test_hybrid_retrieve_adds_title_channel_when_enabled():
    title_doc = Document(page_content="class summary", metadata={"url": "u3", "section_id": "s3", "chunk_position": "1/4"})
    fake_bm25 = MagicMock()
    fake_bm25.search.return_value = []
    fake_bm25.search_titles.return_value = [title_doc]
    bot = _bare_chatbot(bm25_index=fake_bm25)
    vector_docs = [Document(page_content="v", metadata={"url": "u1", "section_id": "s1", "chunk_position": "1/1"})]

    result = bot._hybrid_retrieve(
        "ModelAPI_Feature", vector_docs, k=5, module_filter=None, doc_category_filter=None,
        bm25_enabled=False, title_boost_enabled=True,
    )

    fake_bm25.search_titles.assert_called_once_with(
        "ModelAPI_Feature", k=10, module_filter=None, doc_category_filter=None,
    )
    assert "u3" in {d.metadata["url"] for d in result}


def test_hybrid_retrieve_title_channel_promotes_a_deeply_buried_page():
    """
    The reason this channel exists: a page ranked 21st in dense search (and
    absent from body BM25) should reach rank 0 of the fused pool once the
    title channel places it at rank 0 of its own short list.
    """
    target = Document(page_content="deep hit", metadata={"url": "target", "section_id": "target", "chunk_position": "1/1"})
    filler_vector = [
        Document(page_content=f"v{i}", metadata={"url": f"v{i}", "section_id": f"v{i}", "chunk_position": "1/1"})
        for i in range(20)
    ]
    vector_docs = filler_vector + [target]

    fake_bm25 = MagicMock()
    fake_bm25.search.return_value = []
    fake_bm25.search_titles.return_value = [target]

    bot = _bare_chatbot(bm25_index=fake_bm25)
    result = bot._hybrid_retrieve(
        "ModelAPI_Feature", vector_docs, k=5, module_filter=None, doc_category_filter=None,
        bm25_enabled=False, title_boost_enabled=True,
    )

    assert result[0].metadata["url"] == "target"


# ---------------------------------------------------------------------------
# _create_chain: k_retrieve sets the vector retriever's search depth,
# decoupled from k (the reranker pool). Deep dive must stay untouched — it
# deliberately bypasses rerank/BM25/HyDE and keeps using k_deep_dive.
# ---------------------------------------------------------------------------

def _chain_bot(k_retrieve=None, bm25_index=None):
    """
    Build a DocumentationChatbot with just enough real/stub attributes for
    _create_chain() to run end to end without loading a real ChromaDB or LLM:
    base_prompt is a real PromptTemplate (LCEL needs a real Runnable to pipe
    through) and _initialize_llm returns a RunnableLambda stand-in.
    """
    from core.rag_chatbot import DocumentationChatbot
    bot = DocumentationChatbot.__new__(DocumentationChatbot)
    bot.config = MagicMock(
        k_standard=40, k_deep_dive=60, k_retrieve=k_retrieve,
        deep_dive_batch_size=10, top_n_after_rerank=15,
        bm25_enabled=bm25_index is not None, title_boost_enabled=False,
        temperature=0.0, max_tokens=2000, project_name="P",
    )
    bot.reranker = None
    bot.hyde_llm = None
    bot.bm25_index = bm25_index
    bot.base_prompt = PromptTemplate.from_template("{context} {question}")
    bot._initialize_llm = MagicMock(return_value=RunnableLambda(lambda x: "stub"))
    bot.vectorstore = MagicMock()
    return bot


def test_create_chain_uses_k_retrieve_for_vector_search_depth():
    """k_retrieve, not k_standard, sets the Chroma retriever's per-channel depth."""
    bot = _chain_bot(k_retrieve=80)
    bot._create_chain()
    kwargs = bot.vectorstore.as_retriever.call_args.kwargs
    assert kwargs["search_kwargs"]["k"] == 80


def test_create_chain_defaults_retrieve_depth_to_pool_k_when_k_retrieve_unset():
    """Backwards compatible: k_retrieve=None reproduces today's behaviour."""
    bot = _chain_bot(k_retrieve=None)
    bot._create_chain()
    kwargs = bot.vectorstore.as_retriever.call_args.kwargs
    assert kwargs["search_kwargs"]["k"] == 40


def test_create_chain_deep_dive_ignores_k_retrieve():
    """Deep dive deliberately bypasses this pipeline; it must keep using k_deep_dive."""
    bot = _chain_bot(k_retrieve=80)
    bot._create_chain(deep_dive=True)
    kwargs = bot.vectorstore.as_retriever.call_args.kwargs
    assert kwargs["search_kwargs"]["k"] == 60


def test_ask_reports_title_boost_filter_when_enabled():
    """title_boost must appear in filters, mirroring bm25/hyde reporting."""
    doc = Document(page_content="x", metadata={"title": "T", "url": "u"})
    bot = _bare_chatbot(bm25_index=MagicMock())
    bot.config = MagicMock(top_n_after_rerank=15, title_boost_enabled=True)
    bot.reranker = None
    bot.hyde_llm = None
    bot.available_modules = ["M"]
    holder = [doc]
    chain = MagicMock()
    chain.invoke.return_value = "answer"
    bot._create_chain = MagicMock(return_value=(chain, MagicMock(), holder))

    from core.rag_chatbot import DocumentationChatbot
    result = DocumentationChatbot.ask(bot, "q")

    assert result["filters"]["title_boost"] is True


def test_ask_title_boost_false_when_config_disabled():
    doc = Document(page_content="x", metadata={"title": "T", "url": "u"})
    bot = _bare_chatbot(bm25_index=MagicMock())
    bot.config = MagicMock(top_n_after_rerank=15, title_boost_enabled=False)
    bot.reranker = None
    bot.hyde_llm = None
    bot.available_modules = ["M"]
    holder = [doc]
    chain = MagicMock()
    chain.invoke.return_value = "answer"
    bot._create_chain = MagicMock(return_value=(chain, MagicMock(), holder))

    from core.rag_chatbot import DocumentationChatbot
    result = DocumentationChatbot.ask(bot, "q")

    assert result["filters"]["title_boost"] is False


def test_ask_title_boost_false_when_no_bm25_index_even_if_config_true():
    """The title index lives inside BM25Index — no index loaded, no title channel."""
    doc = Document(page_content="x", metadata={"title": "T", "url": "u"})
    bot = _bare_chatbot(bm25_index=None)
    bot.config = MagicMock(top_n_after_rerank=15, title_boost_enabled=True)
    bot.reranker = None
    bot.hyde_llm = None
    bot.available_modules = ["M"]
    holder = [doc]
    chain = MagicMock()
    chain.invoke.return_value = "answer"
    bot._create_chain = MagicMock(return_value=(chain, MagicMock(), holder))

    from core.rag_chatbot import DocumentationChatbot
    result = DocumentationChatbot.ask(bot, "q")

    assert result["filters"]["title_boost"] is False


def test_ask_title_boost_runtime_override_can_disable():
    """The UI can turn title boost OFF even when config enabled it at startup."""
    doc = Document(page_content="x", metadata={"title": "T", "url": "u"})
    bot = _bare_chatbot(bm25_index=MagicMock())
    bot.config = MagicMock(top_n_after_rerank=15, title_boost_enabled=True)
    bot.reranker = None
    bot.hyde_llm = None
    bot.available_modules = ["M"]
    holder = [doc]
    chain = MagicMock()
    chain.invoke.return_value = "answer"
    bot._create_chain = MagicMock(return_value=(chain, MagicMock(), holder))

    from core.rag_chatbot import DocumentationChatbot
    result = DocumentationChatbot.ask(bot, "q", title_boost_enabled=False)

    assert result["filters"]["title_boost"] is False


def test_response_styles_only_declare_temperature():
    """Presets must not declare keys the UI ignores (k/deep_dive were dead config)."""
    from core.config import ChatbotConfig
    for name, preset in ChatbotConfig.RESPONSE_STYLES.items():
        assert set(preset) == {"temperature"}, f"{name} has stale preset keys"


def test_config_default_style_is_not_a_preset():
    """The 'use config.temperature' sentinel must fall through to no override."""
    from core.config import ChatbotConfig
    assert ChatbotConfig.CONFIG_DEFAULT_STYLE not in ChatbotConfig.RESPONSE_STYLES
    assert ChatbotConfig.RESPONSE_STYLES.get(ChatbotConfig.CONFIG_DEFAULT_STYLE, {}).get("temperature") is None


def test_format_answer_warns_when_not_grounded():
    """grounded=False (agentic-smol answered without reading a page) must be visible."""
    from ui.web import WebUI
    ui = WebUI(MagicMock())
    out = ui._format_answer_markdown({
        "answer": "some answer", "sources": [],
        "filters": {"mode": "agentic-smol", "steps_used": 2, "grounded": False},
        "error": None,
    })
    assert "Warning" in out
    assert "without reading any documentation page" in out


def test_format_answer_no_warning_when_grounded():
    from ui.web import WebUI
    ui = WebUI(MagicMock())
    out = ui._format_answer_markdown({
        "answer": "some answer", "sources": [],
        "filters": {"mode": "agentic-smol", "steps_used": 2, "grounded": True},
        "error": None,
    })
    assert "Warning" not in out
    assert "Agent steps used: 2" in out


def test_format_answer_reports_deep_dive_bypass():
    """Deep dive must announce which retrieval stages it silently skipped."""
    from ui.web import WebUI
    ui = WebUI(MagicMock())
    out = ui._format_answer_markdown({
        "answer": "a", "sources": [],
        "filters": {"mode": "rag", "deep_dive": True,
                    "bypassed_by_deep_dive": ["reranker", "hyde", "bm25"]},
        "error": None,
    })
    assert "skipped" in out
    assert "reranking" in out and "HyDE" in out and "BM25" in out


def test_format_answer_reports_title_boost_bypass():
    """Deep dive skipping the title channel must be visible too."""
    from ui.web import WebUI
    ui = WebUI(MagicMock())
    out = ui._format_answer_markdown({
        "answer": "a", "sources": [],
        "filters": {"mode": "rag", "deep_dive": True,
                    "bypassed_by_deep_dive": ["title_boost"]},
        "error": None,
    })
    assert "skipped" in out
    assert "title" in out.lower()


def test_format_answer_reports_active_retrieval_stages():
    from ui.web import WebUI
    ui = WebUI(MagicMock())
    out = ui._format_answer_markdown({
        "answer": "a", "sources": [],
        "filters": {"mode": "rag", "bm25": True, "hyde": False, "reranker": True},
        "error": None,
    })
    assert "BM25 hybrid" in out
    assert "reranker" in out
    assert "HyDE" not in out


def test_format_answer_reports_title_boost_stage():
    from ui.web import WebUI
    ui = WebUI(MagicMock())
    out = ui._format_answer_markdown({
        "answer": "a", "sources": [],
        "filters": {"mode": "rag", "bm25": False, "hyde": False, "reranker": False, "title_boost": True},
        "error": None,
    })
    assert "title" in out.lower()


def test_emit_json_writes_single_parseable_document():
    """--json-output must put exactly one JSON doc on the stream, nothing else."""
    import io, json
    from chatbot import _emit_json
    buf = io.StringIO()
    _emit_json({
        "answer": "an answer", "error": None, "filters": {"mode": "rag", "bm25": True},
        "sources": [{"title": "T", "module": "M", "doc_category": "dev",
                     "doc_type": "d", "url": "u", "content": "short preview...",
                     "full_content": "the whole chunk"}],
    }, buf)
    parsed = json.loads(buf.getvalue())          # strict: parses the entire buffer
    assert parsed["answer"] == "an answer"
    assert parsed["filters"]["bm25"] is True
    assert len(parsed["sources"]) == 1


def test_emit_json_prefers_full_content_over_preview():
    """Groundedness judging needs the untruncated chunk, not the 200-char preview."""
    import io, json
    from chatbot import _emit_json
    buf = io.StringIO()
    _emit_json({
        "answer": "a", "error": None, "filters": {},
        "sources": [{"content": "trunc...", "full_content": "the complete chunk text"}],
    }, buf)
    assert json.loads(buf.getvalue())["sources"][0]["content"] == "the complete chunk text"


def test_emit_json_falls_back_to_content_for_agentic_sources():
    """Agentic-mode sources carry no chunk text; emitting must not crash or drop them."""
    import io, json
    from chatbot import _emit_json
    buf = io.StringIO()
    _emit_json({
        "answer": "a", "error": None, "filters": {"mode": "agentic-smol", "grounded": True},
        "sources": [{"title": "T", "module": "M", "doc_category": "dev"}],
    }, buf)
    src = json.loads(buf.getvalue())["sources"][0]
    assert src["title"] == "T"
    assert src["content"] is None
    assert src["url"] is None


def test_emit_json_omits_unexpected_keys():
    """Only whitelisted fields are emitted, so new result keys can't leak to stdout."""
    import io, json
    from chatbot import _emit_json
    buf = io.StringIO()
    _emit_json({
        "answer": "a", "error": None, "filters": {}, "sources": [],
        "internal_secret": "must not appear",
    }, buf)
    out = buf.getvalue()
    assert "must not appear" not in out
    assert sorted(json.loads(out)) == ["answer", "error", "filters", "sources"]


def test_ask_sources_include_untruncated_full_content():
    """rag_chatbot.ask() keeps 'content' as a preview but adds the full chunk."""
    long_text = "x" * 500
    doc = Document(page_content=long_text, metadata={"title": "T", "url": "u"})

    bot = _bare_chatbot()
    bot.config = MagicMock(top_n_after_rerank=15)
    bot.reranker = None
    bot.hyde_llm = None
    bot.available_modules = ["M"]

    holder = [doc]
    chain = MagicMock()
    chain.invoke.return_value = "the answer"
    bot._create_chain = MagicMock(return_value=(chain, MagicMock(), holder))

    from core.rag_chatbot import DocumentationChatbot
    result = DocumentationChatbot.ask(bot, "q")

    src = result["sources"][0]
    assert src["content"].endswith("...") and len(src["content"]) == 203
    assert src["full_content"] == long_text


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
        search_depth=40, reranker_enabled=True, top_n=15, hyde_enabled=False,
        bm25_enabled=False, answer_length=2000,
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


def test_handle_message_forwards_bm25_toggle():
    """The BM25 checkbox must reach DocumentationChatbot.ask()."""
    from ui.web import WebUI
    rag = MagicMock()
    rag.ask.return_value = {"answer": "a", "sources": [], "filters": {}, "error": None}
    ui = WebUI(rag)
    ui._handle_message(
        "question", [],
        mode="RAG",
        module_filter="All", doc_type_filter="All",
        response_style="Precise (Recommended)", deep_dive=False,
        search_depth=40, reranker_enabled=True, top_n=15, hyde_enabled=True,
        bm25_enabled=True, answer_length=2000,
        max_chars_per_page=8000, max_pages_per_round=3, max_pages_round2=2,
    )
    assert rag.ask.call_args.kwargs["bm25_enabled"] is True


def test_handle_message_config_default_style_sends_no_temperature():
    """'Config default' style must pass temperature=None so config.temperature applies."""
    from ui.web import WebUI
    from core.config import ChatbotConfig
    rag = MagicMock()
    rag.ask.return_value = {"answer": "a", "sources": [], "filters": {}, "error": None}
    ui = WebUI(rag)
    ui._handle_message(
        "question", [],
        mode="RAG",
        module_filter="All", doc_type_filter="All",
        response_style=ChatbotConfig.CONFIG_DEFAULT_STYLE, deep_dive=False,
        search_depth=40, reranker_enabled=True, top_n=15, hyde_enabled=False,
        bm25_enabled=False, answer_length=2000,
        max_chars_per_page=8000, max_pages_per_round=3, max_pages_round2=2,
    )
    assert rag.ask.call_args.kwargs["temperature"] is None


# ---------------------------------------------------------------------------
# Inherited-member deduplication (task #89)
#
# Doxygen's INLINE_INHERITED_MEMB copies every inherited member's documentation
# verbatim onto every subclass page. All copies — the declaring class's own copy
# included — share one Doxygen anchor id, so anchor_id is the dedup key; section
# chunks carry no anchor and fall back to a content hash.
# ---------------------------------------------------------------------------

def _memitem(page: str, anchor: str, hierarchy: str, content: str = "inherited doc"):
    url = f"https://docs.example.org/SHAPER/{page}#{anchor}"
    return Document(
        page_content=content,
        metadata={
            "url": url,
            "anchor_id": anchor,
            "hierarchy": hierarchy,
            "section_id": url,
            "chunk_position": "1/1",
        },
    )


def _rerank_bot(scores=None):
    """Chatbot stub whose reranker scores docs in the order they are given."""
    bot = _bare_chatbot()
    bot.config = MagicMock(top_n_after_rerank=15)
    bot.reranker = MagicMock()
    if scores is None:
        bot.reranker.predict = lambda pairs: [float(len(pairs) - i) for i in range(len(pairs))]
    else:
        bot.reranker.predict = lambda pairs: scores
    return bot


def test_rerank_collapses_inherited_copies_sharing_an_anchor():
    """15 subclass copies of one inherited member must not fill the result set."""
    anchor = "a4e26d803cc4c58a9342279512e01be6d"
    docs = [
        _memitem(f"classSub{i}.html", anchor, "ModelAPI_Feature::lastResult")
        for i in range(15)
    ]
    docs.append(_memitem("classOther.html", "bdifferent", "Other::thing", "different doc"))

    bot = _rerank_bot()
    result = bot._rerank_and_expand("q", docs)

    assert len(result) == 2
    assert {d.metadata["anchor_id"] for d in result} == {anchor, "bdifferent"}


def test_rerank_prefers_the_declaring_class_copy_when_collapsing():
    """The survivor should cite classModelAPI__Feature.html, not a random subclass."""
    anchor = "a4e26d803cc4c58a9342279512e01be6d"
    hierarchy = "std::shared_ptr< ModelAPI_Result > ModelAPI_Feature::lastResult"
    docs = [
        _memitem("classSketchPlugin__ConstraintRigid.html", anchor, hierarchy),
        _memitem("classFeaturesPlugin__Rotation.html", anchor, hierarchy),
        # declaring class ranked last by the reranker — it must still win
        _memitem("classModelAPI__Feature.html", anchor, hierarchy),
    ]

    bot = _rerank_bot()
    result = bot._rerank_and_expand("q", docs)

    assert len(result) == 1
    assert "classModelAPI__Feature.html" in result[0].metadata["url"]


def test_rerank_falls_back_to_retrieval_order_when_declaring_page_absent():
    """Copies in a group are byte-identical, so reranker scores tie; keep the best-retrieved."""
    anchor = "a4e26d803cc4c58a9342279512e01be6d"
    hierarchy = "void ModelAPI_Entity::emptyFunction"
    docs = [
        _memitem("classSketchPlugin__Circle.html", anchor, hierarchy),
        _memitem("classBuildPlugin__Face.html", anchor, hierarchy),
    ]

    bot = _rerank_bot()
    result = bot._rerank_and_expand("q", docs)

    assert len(result) == 1
    assert "classSketchPlugin__Circle.html" in result[0].metadata["url"]


def test_rerank_deduplicates_anchorless_chunks_by_content():
    """Section chunks carry no anchor_id; identical text must still collapse."""
    def section(page, text):
        url = f"https://docs.example.org/SHAPER/{page}"
        return Document(
            page_content=text,
            metadata={"url": url, "section_id": url, "chunk_position": "1/1"},
        )

    docs = [section("a.html", "same text"), section("b.html", "same text"),
            section("c.html", "other text")]

    bot = _rerank_bot()
    result = bot._rerank_and_expand("q", docs)

    assert len(result) == 2
    assert {d.page_content for d in result} == {"same text", "other text"}


def test_rerank_keeps_distinct_symbols():
    docs = [_memitem(f"classA.html", f"anchor{i}", f"A::m{i}", f"doc {i}") for i in range(5)]

    bot = _rerank_bot()
    result = bot._rerank_and_expand("q", docs)

    assert len(result) == 5


def test_dedup_applies_when_reranker_is_disabled():
    """Duplicates flood the context regardless of reranking, so dedup runs first."""
    anchor = "a4e26d803cc4c58a9342279512e01be6d"
    docs = [
        _memitem("classModelAPI__Feature.html", anchor, "ModelAPI_Feature::lastResult"),
        _memitem("classSub.html", anchor, "ModelAPI_Feature::lastResult"),
        _memitem("classOther.html", "bdifferent", "Other::thing", "different doc"),
    ]

    bot = _rerank_bot()
    result = bot._rerank_and_expand("q", docs, reranker_enabled=False)

    assert len(result) == 2
    assert "classModelAPI__Feature.html" in result[0].metadata["url"]


def test_dedup_applies_when_no_reranker_is_configured():
    anchor = "a4e26d803cc4c58a9342279512e01be6d"
    docs = [
        _memitem("classModelAPI__Feature.html", anchor, "ModelAPI_Feature::lastResult"),
        _memitem("classSub.html", anchor, "ModelAPI_Feature::lastResult"),
    ]

    bot = _rerank_bot()
    bot.reranker = None
    result = bot._rerank_and_expand("q", docs)

    assert len(result) == 1

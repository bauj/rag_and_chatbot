# tests/test_rag_chatbot.py
from core.config import ChatbotConfig
from core.rag_chatbot import DocumentationChatbot


def test_import_documentation_chatbot():
    assert DocumentationChatbot is not None


def test_old_name_not_exported():
    import core
    assert not hasattr(core, "SALOMEChatbot")


def test_create_prompt_contains_project_name():
    """_create_prompt must embed project_name without a live ChromaDB"""
    cfg = ChatbotConfig(project_name="my_sphinx_docs")
    bot = DocumentationChatbot.__new__(DocumentationChatbot)
    bot.config = cfg
    bot.available_modules = ["API", "GUIDE"]
    prompt = bot._create_prompt()
    prompt_str = prompt.template
    assert "my_sphinx_docs" in prompt_str


def test_create_prompt_lists_available_modules():
    cfg = ChatbotConfig(project_name="proj")
    bot = DocumentationChatbot.__new__(DocumentationChatbot)
    bot.config = cfg
    bot.available_modules = ["CORE", "UTILS"]
    prompt_str = bot._create_prompt().template
    assert "CORE" in prompt_str
    assert "UTILS" in prompt_str


from unittest.mock import MagicMock
from langchain_core.documents import Document


def _make_bot_with_reranker(reranker=None, top_n=3):
    """Build a DocumentationChatbot bypassing __init__ heavy work."""
    from core.config import ChatbotConfig, RerankerConfig
    cfg = ChatbotConfig(project_name="test_proj")
    cfg.top_n_after_rerank = top_n
    bot = DocumentationChatbot.__new__(DocumentationChatbot)
    bot.config = cfg
    bot.reranker = reranker
    bot.available_modules = []
    return bot


def test_rerank_and_expand_no_reranker_returns_docs_unchanged():
    bot = _make_bot_with_reranker(reranker=None)
    docs = [Document(page_content="chunk A"), Document(page_content="chunk B")]
    result = bot._rerank_and_expand("query", docs)
    assert result == docs


def test_rerank_and_expand_reranks_by_score():
    """The doc with the higher score should come first."""
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.1, 0.9]  # doc B scores higher
    bot = _make_bot_with_reranker(reranker=mock_reranker, top_n=2)
    docs = [
        Document(page_content="chunk A", metadata={}),
        Document(page_content="chunk B", metadata={}),
    ]
    result = bot._rerank_and_expand("query", docs)
    assert result[0].page_content == "chunk B"
    assert result[1].page_content == "chunk A"


def test_rerank_and_expand_limits_to_top_n():
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.5, 0.4, 0.3, 0.2]
    bot = _make_bot_with_reranker(reranker=mock_reranker, top_n=2)
    docs = [Document(page_content=f"chunk {i}", metadata={}) for i in range(4)]
    result = bot._rerank_and_expand("query", docs)
    assert len(result) == 2


def test_rerank_and_expand_expands_to_section_text():
    """When section_text is in metadata, page_content must be replaced."""
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.9]
    bot = _make_bot_with_reranker(reranker=mock_reranker, top_n=5)
    docs = [Document(
        page_content="small chunk",
        metadata={"section_id": "sec1", "section_text": "full section text here"},
    )]
    result = bot._rerank_and_expand("query", docs)
    assert result[0].page_content == "full section text here"


def test_rerank_and_expand_deduplicates_by_section_id():
    """Two chunks from the same section should result in a single expanded doc."""
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.9, 0.8]
    bot = _make_bot_with_reranker(reranker=mock_reranker, top_n=5)
    shared_meta = {"section_id": "same_section", "section_text": "full section"}
    docs = [
        Document(page_content="chunk 1", metadata=shared_meta),
        Document(page_content="chunk 2", metadata=shared_meta),
    ]
    result = bot._rerank_and_expand("query", docs)
    assert len(result) == 1
    assert result[0].page_content == "full section"

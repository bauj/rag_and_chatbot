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

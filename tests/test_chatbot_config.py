# tests/test_chatbot_config.py
import json
import pytest
from core.config import ChatbotConfig, LLMConfig, EmbeddingConfig, RerankerConfig


def test_defaults():
    cfg = ChatbotConfig()
    assert cfg.project_name == "docs"
    assert cfg.llm.model == "mistral"
    assert cfg.llm.base_url == "http://localhost:8080/v1"
    assert cfg.embedding.model == "all-MiniLM-L6-v2"
    assert cfg.embedding.type == "local"
    assert cfg.reranker is None


def test_chromadb_default_derived_from_project_name():
    cfg = ChatbotConfig(project_name="myproject")
    assert "myproject" in cfg.chromadb_path


def test_explicit_chromadb_path_not_clobbered():
    """Explicit chromadb_path must not be overwritten by __post_init__"""
    cfg = ChatbotConfig(project_name="myproject", chromadb_path="/custom/path/chromadb")
    assert cfg.chromadb_path == "/custom/path/chromadb"


def test_load_from_json(tmp_path):
    config_data = {
        "project_name": "sphinx_project",
        "llm": {"model": "gpt-4o", "base_url": "https://api.openai.com/v1", "api_key": "sk-xxx"},
        "embedding": {"model": "text-embedding-3-small", "type": "api",
                      "base_url": "https://api.openai.com/v1", "api_key": "sk-xxx"},
        "reranker": {"model": "BAAI/bge-reranker-v2-m3", "type": "local"},
    }
    f = tmp_path / "config.json"
    f.write_text(json.dumps(config_data))
    cfg = ChatbotConfig.load(str(f))
    assert cfg.project_name == "sphinx_project"
    assert cfg.llm.model == "gpt-4o"
    assert cfg.embedding.type == "api"
    assert cfg.reranker.model == "BAAI/bge-reranker-v2-m3"


def test_load_from_json_partial_override(tmp_path):
    """Only overridden fields change; rest keeps defaults"""
    config_data = {"project_name": "partial", "llm": {"model": "llama3"}}
    f = tmp_path / "config.json"
    f.write_text(json.dumps(config_data))
    cfg = ChatbotConfig.load(str(f))
    assert cfg.project_name == "partial"
    assert cfg.llm.model == "llama3"
    assert cfg.llm.base_url == "http://localhost:8080/v1"  # default kept


def test_comment_keys_in_nested_blocks_are_stripped(tmp_path):
    """_comment_* keys inside nested blocks must not cause TypeError"""
    config_data = {
        "project_name": "test",
        "llm": {
            "model": "mistral",
            "_comment_base_url": "this is a comment",
            "api_key": "dummy"
        }
    }
    f = tmp_path / "config.json"
    f.write_text(json.dumps(config_data))
    cfg = ChatbotConfig.load(str(f))  # must not raise
    assert cfg.llm.model == "mistral"


def test_invalid_key_raises(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text(json.dumps({"not_a_field": "oops"}))
    with pytest.raises(ValueError, match="Invalid config keys"):
        ChatbotConfig.load(str(f))


def test_collection_name():
    cfg = ChatbotConfig(project_name="my_docs")
    assert cfg.collection_name == "my_docs_documentation"


def test_null_reranker_in_json(tmp_path):
    """Explicit null reranker in JSON should result in cfg.reranker is None"""
    f = tmp_path / "config.json"
    f.write_text(json.dumps({"reranker": None}))
    cfg = ChatbotConfig.load(str(f))
    assert cfg.reranker is None


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        ChatbotConfig.load("/nonexistent/path/config.json")


def test_invalid_json_raises(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{not valid json")
    with pytest.raises(ValueError, match="Invalid JSON"):
        ChatbotConfig.load(str(f))

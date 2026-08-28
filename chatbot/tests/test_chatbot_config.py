"""Tests for chatbot configuration management."""

import json
import pytest
from core.config import ChatbotConfig


def test_agentic_config_defaults():
    from core.config import AgenticConfig
    cfg = AgenticConfig(page_index_path="/tmp/idx.json")
    assert cfg.max_chars_per_page == 8000


def test_agentic_config_custom_values():
    from core.config import AgenticConfig
    cfg = AgenticConfig(
        page_index_path="/tmp/idx.json",
        max_chars_per_page=4000,
    )
    assert cfg.max_chars_per_page == 4000


def test_agentic_config_section_defaults():
    from core.config import AgenticConfig
    cfg = AgenticConfig(page_index_path="/tmp/idx.json")
    assert cfg.section_top_n == 5
    assert cfg.section_char_budget == 15000


def test_agentic_config_section_custom_from_json(tmp_path):
    config_json = tmp_path / "config.json"
    config_json.write_text(json.dumps({
        "agentic": {
            "page_index_path": "/tmp/page_index.json",
            "section_top_n": 8,
            "section_char_budget": 25000,
        }
    }))
    cfg = ChatbotConfig.load(str(config_json))
    assert cfg.agentic.section_top_n == 8
    assert cfg.agentic.section_char_budget == 25000


def test_chatbot_config_agentic_from_json(tmp_path):
    config_json = tmp_path / "config.json"
    config_json.write_text(json.dumps({
        "project_name": "test",
        "chromadb_path": "/tmp/db",
        "agentic": {
            "page_index_path": "/tmp/page_index.json",
            "max_chars_per_page": 4000,
        }
    }))
    cfg = ChatbotConfig.load(str(config_json))
    assert cfg.agentic is not None
    assert cfg.agentic.page_index_path == "/tmp/page_index.json"
    assert cfg.agentic.max_chars_per_page == 4000


def test_chatbot_config_agentic_null_from_json(tmp_path):
    config_json = tmp_path / "config.json"
    config_json.write_text(json.dumps({
        "project_name": "test",
        "chromadb_path": "/tmp/db",
        "agentic": None,
    }))
    cfg = ChatbotConfig.load(str(config_json))
    assert cfg.agentic is None


def test_bm25_enabled_defaults_true():
    cfg = ChatbotConfig(project_name="test", chromadb_path="/tmp/db/chromadb")
    assert cfg.bm25_enabled is True


def test_bm25_enabled_from_json(tmp_path):
    config_json = tmp_path / "config.json"
    config_json.write_text(json.dumps({
        "project_name": "test",
        "chromadb_path": "/tmp/db/chromadb",
        "bm25_enabled": True,
    }))
    cfg = ChatbotConfig.load(str(config_json))
    assert cfg.bm25_enabled is True


def test_bm25_jsonl_path_derived_from_chromadb_path():
    cfg = ChatbotConfig(project_name="test", chromadb_path="/tmp/test_docs_extracted/chromadb")
    assert cfg.bm25_jsonl_path == "/tmp/test_docs_extracted/test_docs.jsonl"


def test_agentic_config_max_steps_default():
    from core.config import AgenticConfig
    cfg = AgenticConfig(page_index_path="/tmp/idx.json")
    assert cfg.max_steps == 20


def test_agentic_config_max_steps_custom():
    from core.config import AgenticConfig
    cfg = AgenticConfig(page_index_path="/tmp/idx.json", max_steps=10)
    assert cfg.max_steps == 10


def test_k_retrieve_defaults_none():
    """None = backwards compatible: per-channel retrieval depth equals the pool size."""
    cfg = ChatbotConfig(project_name="test", chromadb_path="/tmp/db/chromadb")
    assert cfg.k_retrieve is None


def test_k_retrieve_from_json(tmp_path):
    config_json = tmp_path / "config.json"
    config_json.write_text(json.dumps({
        "project_name": "test",
        "chromadb_path": "/tmp/db/chromadb",
        "k_retrieve": 80,
    }))
    cfg = ChatbotConfig.load(str(config_json))
    assert cfg.k_retrieve == 80


def test_title_boost_enabled_defaults_true():
    cfg = ChatbotConfig(project_name="test", chromadb_path="/tmp/db/chromadb")
    assert cfg.title_boost_enabled is True


def test_title_boost_enabled_from_json(tmp_path):
    config_json = tmp_path / "config.json"
    config_json.write_text(json.dumps({
        "project_name": "test",
        "chromadb_path": "/tmp/db/chromadb",
        "title_boost_enabled": True,
    }))
    cfg = ChatbotConfig.load(str(config_json))
    assert cfg.title_boost_enabled is True

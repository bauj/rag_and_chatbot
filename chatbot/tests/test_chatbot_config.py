"""Tests for chatbot configuration management."""

import json
import pytest
from core.config import ChatbotConfig


def test_agentic_config_defaults():
    from core.config import AgenticConfig
    cfg = AgenticConfig(page_index_path="/tmp/idx.json")
    assert cfg.max_chars_per_page == 8000
    assert cfg.max_pages_per_round == 3
    assert cfg.max_pages_round2 == 2


def test_agentic_config_custom_values():
    from core.config import AgenticConfig
    cfg = AgenticConfig(
        page_index_path="/tmp/idx.json",
        max_chars_per_page=4000,
        max_pages_per_round=5,
        max_pages_round2=1,
    )
    assert cfg.max_chars_per_page == 4000
    assert cfg.max_pages_per_round == 5
    assert cfg.max_pages_round2 == 1


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
    assert cfg.agentic.max_pages_per_round == 3  # default


def test_chatbot_config_agentic_null_from_json(tmp_path):
    config_json = tmp_path / "config.json"
    config_json.write_text(json.dumps({
        "project_name": "test",
        "chromadb_path": "/tmp/db",
        "agentic": None,
    }))
    cfg = ChatbotConfig.load(str(config_json))
    assert cfg.agentic is None


def test_bm25_enabled_defaults_false():
    cfg = ChatbotConfig(project_name="test", chromadb_path="/tmp/db/chromadb")
    assert cfg.bm25_enabled is False


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

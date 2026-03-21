# tests/test_extraction_config.py
import json, pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))


def test_processor_created_from_config(tmp_path):
    from process_docs import DocumentationProcessor
    config = {
        "project_name": "my_docs",
        "output_dir": str(tmp_path / "out"),
        "modules": {
            "MODULE_A": {
                "description": "Test module A",
                "dev_path": str(tmp_path / "module_a" / "html"),
                "user_path": str(tmp_path / "module_a" / "html_gui"),
            }
        }
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(config))
    processor = DocumentationProcessor.from_config_file(str(cfg_file))
    assert processor.project_name == "my_docs"
    assert "MODULE_A" in processor.modules


def test_collection_name_from_project_name(tmp_path):
    from process_docs import DocumentationProcessor
    config = {"project_name": "sphinx_docs", "output_dir": str(tmp_path)}
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(config))
    processor = DocumentationProcessor.from_config_file(str(cfg_file))
    assert processor.collection_name == "sphinx_docs_documentation"


def test_output_filenames_from_project_name(tmp_path):
    from process_docs import DocumentationProcessor
    p = DocumentationProcessor(project_name="test_proj", output_dir=str(tmp_path))
    assert p.docs_jsonl_path.name == "test_proj_docs.jsonl"
    assert p.docs_json_path.name == "test_proj_docs.json"


def test_no_default_modules(tmp_path):
    """Processor must not have any hardcoded modules on init"""
    from process_docs import DocumentationProcessor
    p = DocumentationProcessor(project_name="empty", output_dir=str(tmp_path))
    assert len(p.modules) == 0

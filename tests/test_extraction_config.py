# tests/test_extraction_config.py
import json, pytest
from pathlib import Path


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


from bs4 import BeautifulSoup


def _make_processor():
    from process_docs import DocumentationProcessor
    return DocumentationProcessor(project_name="test")


def test_split_into_sections_with_h2_headings():
    html = (
        "<div><h2>Section A</h2><p>Content A</p>"
        "<h2>Section B</h2><p>Content B</p></div>"
    )
    element = BeautifulSoup(html, "html.parser").find("div")
    sections = _make_processor()._split_into_sections(element)
    assert len(sections) == 2
    assert sections[0][0] == "Section A"
    assert "Content A" in sections[0][1]
    assert sections[1][0] == "Section B"
    assert "Content B" in sections[1][1]


def test_split_into_sections_no_headings_returns_single_section():
    html = "<div><p>Just content here</p></div>"
    element = BeautifulSoup(html, "html.parser").find("div")
    sections = _make_processor()._split_into_sections(element)
    assert len(sections) == 1
    assert sections[0][0] is None
    assert "Just content here" in sections[0][1]


def test_split_into_sections_nested_h3():
    html = (
        "<div><h2>Top</h2><p>Intro</p>"
        "<div><h3>Sub</h3><p>Sub content</p></div></div>"
    )
    element = BeautifulSoup(html, "html.parser").find("div")
    sections = _make_processor()._split_into_sections(element)
    assert len(sections) == 2
    assert sections[0][0] == "Top"
    assert "Intro" in sections[0][1]
    assert sections[1][0] == "Sub"
    assert "Sub content" in sections[1][1]


def test_split_into_sections_preamble_before_first_heading():
    """Text before the first heading becomes a (None, text) section."""
    html = "<div><p>Preamble</p><h2>Section</h2><p>Body</p></div>"
    element = BeautifulSoup(html, "html.parser").find("div")
    sections = _make_processor()._split_into_sections(element)
    assert len(sections) == 2
    assert sections[0][0] is None
    assert "Preamble" in sections[0][1]
    assert sections[1][0] == "Section"

# tests/test_extraction_config.py
import json, pytest
from pathlib import Path
from bs4 import BeautifulSoup
import tempfile, os


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


def test_extract_sections_with_soup_dev_uses_contents_div():
    """For dev docs, extract_sections_with_soup should use div.contents."""
    html = """<html><body>
    <div class="contents">
        <h2>API Section</h2><p>Some API content here</p>
    </div>
    </body></html>"""
    soup = BeautifulSoup(html, "html.parser")
    sections = _make_processor().extract_sections_with_soup(soup, "dev")
    assert len(sections) == 1
    assert sections[0][0] == "API Section"
    assert "API content" in sections[0][1]


def test_extract_sections_with_soup_user_uses_role_main():
    """For user docs, extract_sections_with_soup should use div[role=main]."""
    html = """<html><body>
    <div role="main">
        <h2>User Guide</h2><p>Some user guide content here</p>
    </div>
    </body></html>"""
    soup = BeautifulSoup(html, "html.parser")
    sections = _make_processor().extract_sections_with_soup(soup, "user")
    assert len(sections) == 1
    assert sections[0][0] == "User Guide"
    assert "user guide content" in sections[0][1]


def test_process_file_chunks_have_section_metadata(tmp_path):
    """Each chunk carries section_id; section_text is no longer duplicated into it."""
    html_content = """<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<div class="contents">
<h2>Alpha Section</h2>
<p>This is the alpha section with enough words to pass quality filter and produce
at least one chunk of documentation content for testing purposes here.</p>
<h2>Beta Section</h2>
<p>This is the beta section with enough words to pass quality filter and produce
at least one chunk of documentation content for testing purposes here.</p>
</div>
</body>
</html>"""
    html_file = tmp_path / "test.html"
    html_file.write_text(html_content)

    from process_docs import DocumentationProcessor
    processor = DocumentationProcessor(project_name="test_proj")
    processor.quality_threshold = 0.0  # disable quality filter
    chunks = processor.process_file(html_file, "MOD", "dev")

    assert len(chunks) > 0
    for chunk in chunks:
        assert 'section_id' in chunk.metadata
        assert 'section_text' not in chunk.metadata

    # Two sections in the HTML — at least 2 unique section_ids
    section_ids = {c.metadata['section_id'] for c in chunks}
    assert len(section_ids) >= 2


# ---------------------------------------------------------------------------
# section_text used to be copied into EVERY chunk of its section and capped at
# 5,000 chars to contain the resulting bloat (measured on the SHAPER corpus:
# 4.4 MB of chunk content carrying 9.5 MB of duplicated section_text). The cap
# made half the class summaries unanswerable in full — a 60-method class was
# expanded back to its first ~6 methods. The chatbot now rebuilds a section
# from its own chunks, so the copy is redundant and the cap is gone; what the
# chunks must guarantee instead is that they cover the whole section.
# ---------------------------------------------------------------------------

def test_process_file_chunks_cover_a_section_longer_than_the_old_5000_char_cap(tmp_path):
    """A long section must be fully recoverable from its chunks, uncapped."""
    body = " ".join(f"paragraph{i} describes behaviour number {i} in detail." for i in range(400))
    html_file = tmp_path / "long.html"
    html_file.write_text(
        f'<!DOCTYPE html><html><head><title>Long Page</title></head>'
        f'<body><div class="contents"><h2>Big Section</h2><p>{body}</p></div></body></html>'
    )

    from process_docs import DocumentationProcessor
    processor = DocumentationProcessor(project_name="test_proj")
    processor.quality_threshold = 0.0
    chunks = processor.process_file(html_file, "MOD", "dev")

    joined = " ".join(c.content for c in chunks)
    assert len(body) > 5000, "fixture must exceed the old cap to be meaningful"
    assert "paragraph399" in joined
    assert "paragraph0 " in joined

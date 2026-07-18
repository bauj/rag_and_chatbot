import json
import sys
from pathlib import Path
import pytest

EXTRACTION_DIR = Path(__file__).parent.parent.parent / "extraction"
if str(EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_DIR))

from process_docs import DocumentationProcessor
from bs4 import BeautifulSoup


def test_page_index_entries_initialized_empty(tmp_path):
    """DocumentationProcessor starts with empty _page_index_entries."""
    config = {
        "project_name": "test",
        "output_dir": str(tmp_path),
        "modules": {},
        "embedding": {"model": "all-MiniLM-L6-v2", "type": "local"},
        "chunking": {"max_tokens": 384, "overlap_tokens": 50, "char_chunk_size": 1000, "char_overlap": 200},
        "quality": {"min_score": 0.3, "min_word_count": 50, "substantial_word_count": 100},
    }
    processor = DocumentationProcessor(config)
    assert hasattr(processor, '_page_index_entries')
    assert processor._page_index_entries == []


def test_process_file_adds_page_index_entry(tmp_path):
    """process_file adds an entry to _page_index_entries for quality-passing pages."""
    # Create minimal HTML that passes quality filters
    html = """<html><body>
    <h1>MyClass Reference</h1>
    <p>""" + " ".join(["word"] * 120) + """</p>
    </body></html>"""
    html_file = tmp_path / "classMyClass.html"
    html_file.write_text(html)

    config = {
        "project_name": "test",
        "output_dir": str(tmp_path),
        "modules": {},
        "embedding": {"model": "all-MiniLM-L6-v2", "type": "local"},
        "chunking": {"max_tokens": 384, "overlap_tokens": 50, "char_chunk_size": 1000, "char_overlap": 200},
        "quality": {"min_score": 0.3, "min_word_count": 50, "substantial_word_count": 100},
    }
    processor = DocumentationProcessor(config)
    processor.process_file(str(html_file), "test_module", "dev")

    assert len(processor._page_index_entries) == 1
    entry = processor._page_index_entries[0]
    assert entry["filepath"] == str(html_file)
    assert entry["filename"] == "classMyClass.html"
    assert entry["module"] == "test_module"
    assert entry["doc_category"] == "dev"
    assert "title" in entry


def test_process_file_skips_low_quality_pages(tmp_path):
    """process_file does NOT add entry for pages that fail the quality filter."""
    # Too few words to pass quality filter
    html = "<html><body><p>too short</p></body></html>"
    html_file = tmp_path / "short.html"
    html_file.write_text(html)

    config = {
        "project_name": "test",
        "output_dir": str(tmp_path),
        "modules": {},
        "embedding": {"model": "all-MiniLM-L6-v2", "type": "local"},
        "chunking": {"max_tokens": 384, "overlap_tokens": 50, "char_chunk_size": 1000, "char_overlap": 200},
        "quality": {"min_score": 0.3, "min_word_count": 50, "substantial_word_count": 100},
    }
    processor = DocumentationProcessor(config)
    processor.process_file(str(html_file), "test_module", "dev")

    assert processor._page_index_entries == []


def test_extract_content_doxygen_preserves_code_block():
    html = (
        "<html><body><div class='contents'>"
        "<p>Usage:</p><pre>int x = 1;</pre>"
        "</div></body></html>"
    )
    soup = BeautifulSoup(html, "html.parser")
    processor = DocumentationProcessor("test")
    result = processor.extract_content_doxygen(soup)
    assert "```" in result["content"]
    assert "int x = 1;" in result["content"]


def test_extract_content_sphinx_preserves_code_block():
    html = (
        "<html><body><div role='main'>"
        "<p>Usage:</p><pre>int y = 2;</pre>"
        "</div></body></html>"
    )
    soup = BeautifulSoup(html, "html.parser")
    processor = DocumentationProcessor("test")
    result = processor.extract_content_sphinx(soup)
    assert "```" in result["content"]
    assert "int y = 2;" in result["content"]


def _memitem_html():
    return """<html><head><title>ModelAPI_Feature Class Reference</title></head>
    <body><div class="contents">
    <p>""" + " ".join(["word"] * 60) + """</p>
    <a id="a1a2b3c4d5" name="a1a2b3c4d5"></a>
    <h2 class="memtitle">execute()</h2>
    <div class="memitem">
    <div class="memproto">
    <table class="memname">
    <tr><td class="memname">bool ModelAPI_Feature::execute </td>
    <td>(</td><td class="paramname"><em>name</em></td><td>)</td></tr>
    </table>
    </div>
    <div class="memdoc"><p>""" + " ".join(["Executes", "the", "feature."] * 20) + """</p></div>
    </div>
    <a id="b2c3d4e5f6" name="b2c3d4e5f6"></a>
    <h2 class="memtitle">isValid()</h2>
    <div class="memitem">
    <div class="memproto">
    <table class="memname">
    <tr><td class="memname">bool ModelAPI_Feature::isValid </td>
    <td>(</td><td>)</td></tr>
    </table>
    </div>
    <div class="memdoc"><p>""" + " ".join(["Checks", "validity."] * 20) + """</p></div>
    </div>
    </div></body></html>"""


def _memitem_config(tmp_path):
    return {
        "project_name": "test",
        "output_dir": str(tmp_path),
        "modules": {},
        "embedding": {"model": "all-MiniLM-L6-v2", "type": "local"},
        "chunking": {"max_tokens": 384, "overlap_tokens": 50, "char_chunk_size": 1000, "char_overlap": 200},
        "quality": {"min_score": 0.3, "min_word_count": 50, "substantial_word_count": 100},
    }


def test_process_file_emits_one_chunk_per_memitem(tmp_path):
    html_file = tmp_path / "classModelAPI__Feature.html"
    html_file.write_text(_memitem_html())
    processor = DocumentationProcessor(_memitem_config(tmp_path))
    chunks = processor.process_file(str(html_file), "test_module", "dev")

    symbol_names = {c.metadata["symbol_name"] for c in chunks}
    assert symbol_names == {
        "bool ModelAPI_Feature::execute",
        "bool ModelAPI_Feature::isValid",
    }


def test_process_file_memitem_chunk_has_anchor_and_hierarchy(tmp_path):
    html_file = tmp_path / "classModelAPI__Feature.html"
    html_file.write_text(_memitem_html())
    processor = DocumentationProcessor(_memitem_config(tmp_path))
    chunks = processor.process_file(str(html_file), "test_module", "dev")

    execute_chunk = next(c for c in chunks if c.metadata["symbol_name"] == "bool ModelAPI_Feature::execute")
    assert execute_chunk.metadata["anchor_id"] == "a1a2b3c4d5"
    assert execute_chunk.hierarchy == "bool ModelAPI_Feature::execute"
    assert execute_chunk.url.endswith("#a1a2b3c4d5")


def test_process_file_memitem_signature_in_content(tmp_path):
    html_file = tmp_path / "classModelAPI__Feature.html"
    html_file.write_text(_memitem_html())
    processor = DocumentationProcessor(_memitem_config(tmp_path))
    chunks = processor.process_file(str(html_file), "test_module", "dev")

    execute_chunk = next(c for c in chunks if c.metadata["symbol_name"] == "bool ModelAPI_Feature::execute")
    assert "ModelAPI_Feature::execute" in execute_chunk.content


def test_process_file_without_memitems_uses_section_path(tmp_path):
    html = "<html><body><h1>Overview</h1><p>" + " ".join(["word"] * 120) + "</p></body></html>"
    html_file = tmp_path / "overview.html"
    html_file.write_text(html)
    processor = DocumentationProcessor(_memitem_config(tmp_path))
    chunks = processor.process_file(str(html_file), "test_module", "dev")

    assert len(chunks) >= 1
    assert "symbol_name" not in chunks[0].metadata

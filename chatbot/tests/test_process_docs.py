import json
import sys
from pathlib import Path
import pytest

EXTRACTION_DIR = Path(__file__).parent.parent.parent / "extraction"
if str(EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_DIR))

from process_docs import DocumentationProcessor


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

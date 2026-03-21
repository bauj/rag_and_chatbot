# tests/test_html_parser.py
import json
import sys
from pathlib import Path
import pytest

# html_parser lives in extraction/ — add it to sys.path
EXTRACTION_DIR = Path(__file__).parent.parent.parent / "extraction"
if str(EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_DIR))

from html_parser import get_page_title, parse_page, save_page_index, search_pages


# ---------------------------------------------------------------------------
# get_page_title
# ---------------------------------------------------------------------------

def test_get_page_title_from_h1(tmp_path):
    html = "<html><body><h1>My Class Reference</h1><p>text</p></body></html>"
    f = tmp_path / "test.html"
    f.write_text(html)
    assert get_page_title(str(f)) == "My Class Reference"


def test_get_page_title_from_title_tag(tmp_path):
    html = "<html><head><title>Page Title</title></head><body><p>text</p></body></html>"
    f = tmp_path / "test.html"
    f.write_text(html)
    assert get_page_title(str(f)) == "Page Title"


def test_get_page_title_fallback_to_filename(tmp_path):
    html = "<html><body><p>no title</p></body></html>"
    f = tmp_path / "myfile.html"
    f.write_text(html)
    assert get_page_title(str(f)) == "myfile"


def test_get_page_title_file_not_found():
    with pytest.raises(FileNotFoundError):
        get_page_title("/nonexistent/path.html")


# ---------------------------------------------------------------------------
# parse_page
# ---------------------------------------------------------------------------

def test_parse_page_returns_string(tmp_path):
    html = "<html><body><h1>Title</h1><p>Some content here</p></body></html>"
    f = tmp_path / "page.html"
    f.write_text(html)
    result = parse_page(str(f), "dev")
    assert isinstance(result, str)
    assert len(result) > 0


def test_parse_page_truncates_at_max_chars(tmp_path):
    long_text = "word " * 10000
    html = f"<html><body><p>{long_text}</p></body></html>"
    f = tmp_path / "long.html"
    f.write_text(html)
    result = parse_page(str(f), "dev", max_chars=500)
    assert len(result) <= 600  # some tolerance for truncation marker


def test_parse_page_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_page("/nonexistent/page.html", "dev")


# ---------------------------------------------------------------------------
# save_page_index
# ---------------------------------------------------------------------------

def test_save_page_index_creates_valid_json(tmp_path):
    entries = [
        {"filepath": "/a/b.html", "filename": "b.html", "title": "B", "module": "X", "doc_category": "dev"},
    ]
    output = tmp_path / "index.json"
    save_page_index(entries, str(output))
    loaded = json.loads(output.read_text())
    assert loaded == entries


def test_save_page_index_creates_parent_dirs(tmp_path):
    entries = [{"filepath": "/a.html", "filename": "a.html", "title": "A", "module": "X", "doc_category": "dev"}]
    output = tmp_path / "subdir" / "index.json"
    save_page_index(entries, str(output))
    assert output.exists()


# ---------------------------------------------------------------------------
# search_pages
# ---------------------------------------------------------------------------

def _make_index():
    return [
        {"filepath": "/docs/classModelAPI__Feature.html",
         "filename": "classModelAPI__Feature.html",
         "title": "ModelAPI_Feature Class Reference",
         "module": "SHAPER", "doc_category": "dev"},
        {"filepath": "/docs/group__ModelAPI.html",
         "filename": "group__ModelAPI.html",
         "title": "ModelAPI Group",
         "module": "SHAPER", "doc_category": "dev"},
        {"filepath": "/docs/tutorial_mesh.html",
         "filename": "tutorial_mesh.html",
         "title": "Mesh Tutorial",
         "module": "SHAPER", "doc_category": "user"},
    ]


def test_search_pages_finds_by_filename_token(dummy=None):
    idx = _make_index()
    results = search_pages(idx, "ModelAPI_Feature")
    filenames = [r["filename"] for r in results]
    assert "classModelAPI__Feature.html" in filenames


def test_search_pages_finds_by_title_token():
    idx = _make_index()
    results = search_pages(idx, "mesh tutorial")
    filenames = [r["filename"] for r in results]
    assert "tutorial_mesh.html" in filenames


def test_search_pages_no_match_returns_empty():
    idx = _make_index()
    results = search_pages(idx, "zzznomatchzzz")
    assert results == []


def test_search_pages_deduplicates():
    idx = _make_index()
    # Query that matches same page via filename AND title
    results = search_pages(idx, "ModelAPI Feature class reference")
    filepaths = [r["filepath"] for r in results]
    assert len(filepaths) == len(set(filepaths))


def test_search_pages_exclude_filepaths():
    idx = _make_index()
    excluded = {"/docs/classModelAPI__Feature.html"}
    results = search_pages(idx, "ModelAPI", exclude_filepaths=excluded)
    filepaths = {r["filepath"] for r in results}
    assert "/docs/classModelAPI__Feature.html" not in filepaths

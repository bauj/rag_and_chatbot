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
from html_parser import _split_into_sections, extract_sections_with_soup
from html_parser import html_to_markdown
from html_parser import has_memitems, extract_memitems_with_soup


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


# ---------------------------------------------------------------------------
# _split_into_sections
# ---------------------------------------------------------------------------

def _make_tag(html: str):
    """Helper: parse html and return the <body> Tag."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    return soup.find("body")


def test_split_into_sections_returns_list_of_tuples():
    html = "<body><h2>Section A</h2><p>Content A</p><h2>Section B</h2><p>Content B</p></body>"
    tag = _make_tag(html)
    result = _split_into_sections(tag)
    assert isinstance(result, list)
    assert len(result) >= 1
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2


def test_split_into_sections_captures_h2_headings():
    html = "<body><h2>Alpha</h2><p>First paragraph.</p><h2>Beta</h2><p>Second paragraph.</p></body>"
    tag = _make_tag(html)
    result = _split_into_sections(tag)
    headings = [heading for heading, _ in result]
    assert "Alpha" in headings
    assert "Beta" in headings


def test_split_into_sections_no_headings_returns_single_entry():
    html = "<body><p>Some plain text with no headings.</p></body>"
    tag = _make_tag(html)
    result = _split_into_sections(tag)
    assert len(result) == 1
    heading, text = result[0]
    assert heading is None
    assert "plain text" in text


def test_split_into_sections_no_headings_preserves_code_block():
    html = "<body><p>Intro.</p><pre>int z = 3;</pre></body>"
    tag = _make_tag(html)
    result = _split_into_sections(tag)
    assert len(result) == 1
    _, text = result[0]
    assert "```" in text
    assert "int z = 3;" in text


def test_split_into_sections_h3_creates_boundary():
    html = "<body><h3>Sub-section</h3><p>Sub content.</p></body>"
    tag = _make_tag(html)
    result = _split_into_sections(tag)
    headings = [heading for heading, _ in result]
    assert "Sub-section" in headings


def test_split_into_sections_fences_pre_code_blocks():
    html = (
        "<body><h2>Example</h2><p>Usage:</p>"
        "<pre>int x = 1;\nint y = 2;</pre></body>"
    )
    tag = _make_tag(html)
    result = _split_into_sections(tag)
    text = dict(result)["Example"]
    assert "```" in text
    assert "int x = 1;" in text
    assert "int y = 2;" in text


def test_split_into_sections_fences_doxygen_fragment_blocks():
    html = (
        "<body><h2>Example</h2><p>Usage:</p>"
        "<div class=\"fragment\">"
        "<div class=\"line\">int x = 1;</div>"
        "<div class=\"line\">int y = 2;</div>"
        "</div></body>"
    )
    tag = _make_tag(html)
    result = _split_into_sections(tag)
    text = dict(result)["Example"]
    assert "```" in text
    assert "int x = 1;" in text
    assert "int y = 2;" in text


def test_split_into_sections_skips_empty_code_blocks():
    html = "<body><h2>Example</h2><pre>   </pre><p>Real content.</p></body>"
    tag = _make_tag(html)
    result = _split_into_sections(tag)
    text = dict(result)["Example"]
    assert "```" not in text
    assert "Real content." in text


# ---------------------------------------------------------------------------
# extract_sections_with_soup
# ---------------------------------------------------------------------------

def test_extract_sections_with_soup_returns_list():
    from bs4 import BeautifulSoup
    html = "<html><body><div role='main'><h2>Intro</h2><p>Hello world.</p></div></body></html>"
    soup = BeautifulSoup(html, "html.parser")
    result = extract_sections_with_soup(soup, "user")
    assert isinstance(result, list)
    assert len(result) >= 1


def test_extract_sections_with_soup_entries_are_tuples():
    from bs4 import BeautifulSoup
    html = "<html><body><div role='main'><h2>Overview</h2><p>Details here.</p></div></body></html>"
    soup = BeautifulSoup(html, "html.parser")
    result = extract_sections_with_soup(soup, "user")
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2


def test_extract_sections_with_soup_dev_category_uses_contents_div():
    from bs4 import BeautifulSoup
    html = "<html><body><div class='contents'><h2>API</h2><p>API reference text.</p></div></body></html>"
    soup = BeautifulSoup(html, "html.parser")
    result = extract_sections_with_soup(soup, "dev")
    headings = [h for h, _ in result]
    assert "API" in headings


def test_extract_sections_with_soup_strips_script_tags():
    from bs4 import BeautifulSoup
    html = (
        "<html><body><div role='main'>"
        "<script>alert('x')</script>"
        "<h2>Clean</h2><p>Visible content.</p>"
        "</div></body></html>"
    )
    soup = BeautifulSoup(html, "html.parser")
    result = extract_sections_with_soup(soup, "user")
    all_text = " ".join(text for _, text in result)
    assert "alert" not in all_text


# ---------------------------------------------------------------------------
# html_to_markdown
# ---------------------------------------------------------------------------

def test_html_to_markdown_preserves_pre_as_fenced_code():
    from bs4 import BeautifulSoup
    html = "<div><p>Intro text.</p><pre>int x = 1;\nint y = 2;</pre></div>"
    soup = BeautifulSoup(html, "html.parser")
    result = html_to_markdown(soup.find("div"))
    assert "```" in result
    assert "int x = 1;" in result


def test_html_to_markdown_collapses_excess_blank_lines():
    from bs4 import BeautifulSoup
    html = "<div><p>A</p><p></p><p></p><p></p><p>B</p></div>"
    soup = BeautifulSoup(html, "html.parser")
    result = html_to_markdown(soup.find("div"))
    assert "\n\n\n" not in result


# ---------------------------------------------------------------------------
# extract_memitems_with_soup / has_memitems
# ---------------------------------------------------------------------------

def _make_memitem_page():
    from bs4 import BeautifulSoup
    html = """
    <html><body><div class="contents">
    <a id="a1a2b3c4d5" name="a1a2b3c4d5"></a>
    <h2 class="memtitle">execute()</h2>
    <div class="memitem">
    <div class="memproto">
    <table class="memname">
    <tr>
    <td class="memname">bool ModelAPI_Feature::execute </td>
    <td>(</td>
    <td class="paramtype">const std::string&amp;&#160;</td>
    <td class="paramname"><em>name</em></td>
    <td>)</td>
    </tr>
    </table>
    </div>
    <div class="memdoc">
    <p>Executes the feature and returns success status.</p>
    </div>
    </div>
    </div></body></html>
    """
    return BeautifulSoup(html, "html.parser")


def test_has_memitems_true_when_present():
    soup = _make_memitem_page()
    assert has_memitems(soup) is True


def test_has_memitems_false_when_absent():
    from bs4 import BeautifulSoup
    soup = BeautifulSoup("<html><body><p>plain page</p></body></html>", "html.parser")
    assert has_memitems(soup) is False


def test_extract_memitems_returns_one_entry():
    soup = _make_memitem_page()
    items = extract_memitems_with_soup(soup)
    assert len(items) == 1


def test_extract_memitems_symbol_name():
    soup = _make_memitem_page()
    items = extract_memitems_with_soup(soup)
    assert items[0]["symbol_name"] == "bool ModelAPI_Feature::execute"


def test_extract_memitems_signature_kept_verbatim():
    soup = _make_memitem_page()
    items = extract_memitems_with_soup(soup)
    signature = items[0]["signature"]
    assert "ModelAPI_Feature::execute" in signature
    assert "const std::string" in signature


def test_extract_memitems_description():
    soup = _make_memitem_page()
    items = extract_memitems_with_soup(soup)
    assert "Executes the feature" in items[0]["description"]


def test_extract_memitems_anchor_id():
    soup = _make_memitem_page()
    items = extract_memitems_with_soup(soup)
    assert items[0]["anchor_id"] == "a1a2b3c4d5"


def test_extract_memitems_empty_list_when_no_memitems():
    from bs4 import BeautifulSoup
    soup = BeautifulSoup("<html><body><p>plain page</p></body></html>", "html.parser")
    assert extract_memitems_with_soup(soup) == []

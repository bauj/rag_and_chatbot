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
from html_parser import extract_class_summary


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


class _FakeDoc:
    """Stand-in for a langchain Document — .metadata['url'] is read by search_pages,
    .page_content is what a rerank_fn/dedup would score/hash. Extra metadata
    (anchor_id, hierarchy, section_id, chunk_position) supports the dedup tests."""

    def __init__(self, filename, page_content="", **extra_metadata):
        self.metadata = {"url": f"https://example.com/{filename}", **extra_metadata}
        self.page_content = page_content or filename


class _FakeVectorStore:
    """similarity_search returns a fixed ranked list of filenames, regardless of query/k."""

    def __init__(self, ranked_filenames):
        self._ranked_filenames = ranked_filenames

    def similarity_search(self, query, k):
        return [_FakeDoc(f) for f in self._ranked_filenames[:k]]


class _FakeVectorStoreDocs:
    """Like _FakeVectorStore, but returns pre-built Documents directly — for
    tests that need specific per-chunk metadata (anchor_id, hierarchy) rather
    than just a ranked filename list."""

    def __init__(self, docs):
        self._docs = list(docs)

    def similarity_search(self, query, k):
        return self._docs[:k]


class _FakeBM25Index:
    """search()/search_titles() each return a fixed ranked list of filenames."""

    def __init__(self, body_ranked=(), title_ranked=()):
        self._body_ranked = list(body_ranked)
        self._title_ranked = list(title_ranked)

    def search(self, query, k):
        return [_FakeDoc(f) for f in self._body_ranked[:k]]

    def search_titles(self, query, k):
        return [_FakeDoc(f) for f in self._title_ranked[:k]]


def test_search_pages_finds_dense_hit(dummy=None):
    idx = _make_index()
    vs = _FakeVectorStore(["classModelAPI__Feature.html"])
    results = search_pages(vs, idx, "ModelAPI_Feature")
    filenames = [r["filename"] for r in results]
    assert "classModelAPI__Feature.html" in filenames


def test_search_pages_no_match_returns_empty():
    idx = _make_index()
    vs = _FakeVectorStore([])
    results = search_pages(vs, idx, "zzznomatchzzz")
    assert results == []


def test_search_pages_skips_hits_with_no_matching_page_index_entry():
    idx = _make_index()
    vs = _FakeVectorStore(["unknown_page.html", "tutorial_mesh.html"])
    results = search_pages(vs, idx, "mesh")
    filenames = [r["filename"] for r in results]
    assert filenames == ["tutorial_mesh.html"]


def test_search_pages_exclude_filepaths():
    idx = _make_index()
    vs = _FakeVectorStore(["classModelAPI__Feature.html", "group__ModelAPI.html"])
    excluded = {"/docs/classModelAPI__Feature.html"}
    results = search_pages(vs, idx, "ModelAPI", exclude_filepaths=excluded)
    filepaths = {r["filepath"] for r in results}
    assert "/docs/classModelAPI__Feature.html" not in filepaths


def test_search_pages_fuses_bm25_body_channel_when_index_given():
    idx = _make_index()
    # Dense channel finds nothing useful; BM25 body channel does.
    vs = _FakeVectorStore([])
    bm25 = _FakeBM25Index(body_ranked=["tutorial_mesh.html"])
    results = search_pages(vs, idx, "mesh tutorial", bm25_index=bm25)
    filenames = [r["filename"] for r in results]
    assert "tutorial_mesh.html" in filenames


def test_search_pages_ignores_title_channel_when_not_boosted():
    idx = _make_index()
    vs = _FakeVectorStore([])
    bm25 = _FakeBM25Index(title_ranked=["group__ModelAPI.html"])
    results = search_pages(vs, idx, "ModelAPI", bm25_index=bm25, title_boost_enabled=False)
    assert results == []


def test_search_pages_fuses_title_channel_when_boosted():
    idx = _make_index()
    vs = _FakeVectorStore([])
    bm25 = _FakeBM25Index(title_ranked=["group__ModelAPI.html"])
    results = search_pages(vs, idx, "ModelAPI", bm25_index=bm25, title_boost_enabled=True)
    filenames = [r["filename"] for r in results]
    assert "group__ModelAPI.html" in filenames


def test_search_pages_deduplicates_across_channels():
    idx = _make_index()
    # Same page ranked in both the dense and BM25 channels — must appear once.
    vs = _FakeVectorStore(["classModelAPI__Feature.html"])
    bm25 = _FakeBM25Index(body_ranked=["classModelAPI__Feature.html"])
    results = search_pages(vs, idx, "ModelAPI Feature class reference", bm25_index=bm25)
    filepaths = [r["filepath"] for r in results]
    assert len(filepaths) == len(set(filepaths))


def test_search_pages_reorders_by_rerank_fn_score():
    idx = _make_index()
    # RRF/dense order puts group__ModelAPI.html first; rerank_fn scores the
    # other page higher, and the final order must follow the score, not RRF rank.
    vs = _FakeVectorStore(["group__ModelAPI.html", "classModelAPI__Feature.html"])

    def rerank_fn(query, docs):
        return [1.0 if d.metadata["url"].endswith("classModelAPI__Feature.html") else 0.0
                for d in docs]

    results = search_pages(vs, idx, "ModelAPI", rerank_fn=rerank_fn)
    filenames = [r["filename"] for r in results]
    assert filenames == ["classModelAPI__Feature.html", "group__ModelAPI.html"]


def test_search_pages_rerank_fn_scores_the_actual_chunk_content():
    idx = _make_index()
    vs = _FakeVectorStore(["classModelAPI__Feature.html"])
    seen_content = []

    def rerank_fn(query, docs):
        seen_content.extend(d.page_content for d in docs)
        return [0.0 for _ in docs]

    search_pages(vs, idx, "ModelAPI", rerank_fn=rerank_fn)
    assert seen_content == ["classModelAPI__Feature.html"]


def _symbol_dedup_index():
    return [
        {"filepath": "/docs/classSub1.html", "filename": "classSub1.html",
         "title": "Sub1 Class Reference", "module": "SHAPER", "doc_category": "dev"},
        {"filepath": "/docs/classSub2.html", "filename": "classSub2.html",
         "title": "Sub2 Class Reference", "module": "SHAPER", "doc_category": "dev"},
        {"filepath": "/docs/classModelAPI__Feature.html", "filename": "classModelAPI__Feature.html",
         "title": "ModelAPI_Feature Class Reference", "module": "SHAPER", "doc_category": "dev"},
    ]


def test_search_pages_dedups_inherited_symbol_copies_across_pages():
    """Two subclass pages carry byte-identical copies of an inherited method
    (Doxygen's INLINE_INHERITED_MEMB) alongside the declaring class's own
    page — all three must collapse to just the declaring page's page."""
    idx = _symbol_dedup_index()
    anchor = "a1"
    hierarchy = "ModelAPI_Feature::lastResult"

    # Declaring page ranked LAST — dedup must still prefer it over the subclasses.
    vs = _FakeVectorStoreDocs([
        _FakeDoc("classSub1.html", anchor_id=anchor, hierarchy=hierarchy),
        _FakeDoc("classSub2.html", anchor_id=anchor, hierarchy=hierarchy),
        _FakeDoc("classModelAPI__Feature.html", anchor_id=anchor, hierarchy=hierarchy),
    ])

    results = search_pages(vs, idx, "lastResult")
    filenames = [r["filename"] for r in results]

    assert filenames == ["classModelAPI__Feature.html"]


def test_search_pages_keeps_distinct_symbols_separate():
    """Different anchor_ids must not be collapsed together."""
    idx = _symbol_dedup_index()
    vs = _FakeVectorStoreDocs([
        _FakeDoc("classSub1.html", anchor_id="a1", hierarchy="ModelAPI_Feature::lastResult"),
        _FakeDoc("classSub2.html", anchor_id="a2", hierarchy="ModelAPI_Feature::firstResult"),
    ])

    results = search_pages(vs, idx, "result")
    filenames = {r["filename"] for r in results}

    assert filenames == {"classSub1.html", "classSub2.html"}


def test_search_pages_dedup_sees_every_chunk_not_just_first_ranked():
    """The dense channel's first-ranked chunk for classSub1.html is a DIFFERENT
    symbol (no anchor) than its second, later-ranked chunk (which does share
    the duplicated anchor) — dedup must still catch the duplicate, proving
    collapse-to-one-per-page does not happen before dedup runs."""
    idx = _symbol_dedup_index()
    anchor = "a1"
    hierarchy = "ModelAPI_Feature::lastResult"

    vs = _FakeVectorStoreDocs([
        _FakeDoc("classSub1.html", page_content="unrelated content on this page"),  # no anchor
        _FakeDoc("classModelAPI__Feature.html", anchor_id=anchor, hierarchy=hierarchy),
        _FakeDoc("classSub1.html", page_content="the inherited copy", anchor_id=anchor, hierarchy=hierarchy),
    ])

    results = search_pages(vs, idx, "lastResult")
    filenames = [r["filename"] for r in results]

    # classSub1.html survives once (its first, non-duplicate chunk) but its
    # duplicate chunk must not also win the declaring page's spot.
    assert filenames.count("classModelAPI__Feature.html") == 1
    assert "classSub1.html" in filenames


def test_search_pages_without_rerank_fn_keeps_rrf_order():
    idx = _make_index()
    vs = _FakeVectorStore(["group__ModelAPI.html", "classModelAPI__Feature.html"])
    results = search_pages(vs, idx, "ModelAPI")
    filenames = [r["filename"] for r in results]
    assert filenames == ["group__ModelAPI.html", "classModelAPI__Feature.html"]


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


# ---------------------------------------------------------------------------
# Inherited memitems (task #91)
#
# Doxygen runs with INLINE_INHERITED_MEMB, which copies every inherited member's
# documentation verbatim onto every subclass page — 57% of the SHAPER corpus. The
# copies are marked <span class="mlabel inherited">inherited</span> inside memproto.
# Match that class, never the rendered text, which also occurs in prose.
# ---------------------------------------------------------------------------

def _make_page_with_inherited_memitem():
    from bs4 import BeautifulSoup
    html = """
    <html><body><div class="contents">
    <a id="adeclared0" name="adeclared0"></a>
    <div class="memitem">
    <div class="memproto">
    <table class="mlabels"><tr><td class="mlabels-left">
    <table class="memname"><tr><td class="memname">bool SketchPlugin_Circle::execute </td></tr></table>
    </td><td class="mlabels-right"><span class="mlabel">virtual</span></td></tr></table>
    </div>
    <div class="memdoc"><p>Declared on this page.</p></div>
    </div>
    <a id="ainherited0" name="ainherited0"></a>
    <div class="memitem">
    <div class="memproto">
    <table class="mlabels"><tr><td class="mlabels-left">
    <table class="memname"><tr><td class="memname">void ModelAPI_Entity::emptyFunction </td></tr></table>
    </td><td class="mlabels-right"><span class="mlabel inline">inline</span><span class="mlabel inherited">inherited</span></td></tr></table>
    </div>
    <div class="memdoc"><p>Copied here from the base class.</p></div>
    </div>
    </div></body></html>
    """
    return BeautifulSoup(html, "html.parser")


def test_extract_memitems_skips_inherited_copies():
    soup = _make_page_with_inherited_memitem()
    items = extract_memitems_with_soup(soup)
    assert len(items) == 1
    assert items[0]["symbol_name"] == "bool SketchPlugin_Circle::execute"


def test_extract_memitems_keeps_declared_members_with_other_labels():
    """virtual/inline/static/protected labels must not be mistaken for inherited."""
    soup = _make_page_with_inherited_memitem()
    items = extract_memitems_with_soup(soup)
    assert [i["anchor_id"] for i in items] == ["adeclared0"]


def test_extract_memitems_does_not_match_the_word_inherited_in_prose():
    from bs4 import BeautifulSoup
    html = """
    <html><body>
    <a id="aprose0" name="aprose0"></a>
    <div class="memitem">
    <div class="memproto"><table class="memname"><tr>
    <td class="memname">void Foo::bar </td></tr></table></div>
    <div class="memdoc"><p>This value is inherited from the parent widget.</p></div>
    </div>
    </body></html>
    """
    items = extract_memitems_with_soup(BeautifulSoup(html, "html.parser"))
    assert len(items) == 1


# ---------------------------------------------------------------------------
# extract_class_summary (task #92)
#
# Doxygen renders a "Public Member Functions" memberdecls table listing every
# method with its brief description. It is the only page-level view of a class's
# API, and the memitem branch used to discard it — so aggregate questions like
# "what are the public methods of X" had nothing in the corpus to match.
#
# With INLINE_INHERITED_MEMB the table MERGES inherited members with no marker on
# the row, so rows are filtered by the anchors declared on this page.
# ---------------------------------------------------------------------------

def _make_class_page():
    from bs4 import BeautifulSoup
    html = """
    <html><body><div class="contents">
    <div class="textblock"><p>Feature function that represents the particular functionality.</p></div>
    <table class="memberdecls">
    <tr class="heading"><td><h2 class="groupheader">Public Member Functions</h2></td></tr>
    <tr class="memitem:adeclared0"><td class="memItemRight">virtual void execute ()=0</td></tr>
    <tr class="memdesc:adeclared0"><td class="mdescRight">Computes or recomputes the results.</td></tr>
    <tr class="memitem:ainherited0"><td class="memItemRight">virtual void emptyFunction () const</td></tr>
    <tr class="memdesc:ainherited0"><td class="mdescRight">Empty function for interface virtualisation.</td></tr>
    </table>
    <table class="memberdecls">
    <tr class="heading"><td><h2 class="groupheader">Static Public Member Functions</h2></td></tr>
    <tr class="memitem:adeclared1"><td class="memItemRight">static std::string group ()</td></tr>
    <tr class="memdesc:adeclared1"><td class="mdescRight">Returns the group identifier.</td></tr>
    </table>
    </div></body></html>
    """
    return BeautifulSoup(html, "html.parser")


def test_class_summary_includes_declared_members():
    summary = extract_class_summary(_make_class_page(), {"adeclared0", "adeclared1"})
    assert "virtual void execute ()=0" in summary
    assert "Computes or recomputes the results." in summary


def test_class_summary_excludes_inherited_members():
    """The merged table lists inherited members with no marker; anchors filter them."""
    summary = extract_class_summary(_make_class_page(), {"adeclared0", "adeclared1"})
    assert "emptyFunction" not in summary


def test_class_summary_groups_under_table_headings():
    summary = extract_class_summary(_make_class_page(), {"adeclared0", "adeclared1"})
    assert "Public Member Functions" in summary
    assert "Static Public Member Functions" in summary
    assert summary.index("Public Member Functions") < summary.index("Static Public")


def test_class_summary_includes_class_description():
    summary = extract_class_summary(_make_class_page(), {"adeclared0"})
    assert "particular functionality" in summary


def test_class_summary_empty_when_no_declared_rows_survive():
    summary = extract_class_summary(_make_class_page(), set())
    assert "execute" not in summary
    assert "emptyFunction" not in summary


def test_class_summary_empty_string_when_no_memberdecls():
    from bs4 import BeautifulSoup
    soup = BeautifulSoup("<html><body><p>no tables here</p></body></html>", "html.parser")
    assert extract_class_summary(soup, {"a1"}) == ""


def test_class_summary_drops_heading_with_no_surviving_rows():
    """A table whose rows are all inherited must not leave a bare heading behind."""
    summary = extract_class_summary(_make_class_page(), {"adeclared1"})
    assert "execute" not in summary
    assert "Static Public Member Functions" in summary
    # the first table contributed nothing, so its heading must be gone; the only
    # remaining occurrence of the substring is inside "Static Public Member Functions"
    assert summary.count("Public Member Functions") == 1


# ---------------------------------------------------------------------------
# Qualified member names in the class summary (task #93)
#
# Doxygen's member table lists bare member names, so every class summary reads
# alike and they compete with each other for "methods of <class>" questions.
# The per-symbol detail chunks do NOT have this problem because each carries the
# qualified name inline — which is why symbol questions resolve and page-level
# ones do not.
# ---------------------------------------------------------------------------

def _make_qualified_class_page():
    from bs4 import BeautifulSoup
    html = """
    <html><body><div class="contents">
    <div class="textblock"><p>Feature function of this operation.</p></div>
    <table class="memberdecls">
    <tr class="heading"><td><h2 class="groupheader">Public Member Functions</h2></td></tr>
    <tr class="memitem:a001"><td class="memItemRight">virtual const std::string &amp;
        <a class="el" href="classModelAPI__Feature.html#a001">getKind</a> ()=0</td></tr>
    <tr class="memdesc:a001"><td class="mdescRight">Returns the unique kind of a feature.</td></tr>
    <tr class="memitem:a002"><td class="memItemRight">virtual std::shared_ptr&lt;
        <a class="el" href="classModelAPI__Document.html">ModelAPI_Document</a> &gt;
        <a class="el" href="classModelAPI__Feature.html#a002">document</a> () const</td></tr>
    <tr class="memdesc:a002"><td class="mdescRight">Returns the document.</td></tr>
    </table>
    </div></body></html>
    """
    return BeautifulSoup(html, "html.parser")


def test_class_summary_qualifies_each_member_with_the_class_name():
    summary = extract_class_summary(
        _make_qualified_class_page(), {"a001", "a002"}, class_name="ModelAPI_Feature"
    )
    assert "ModelAPI_Feature::getKind" in summary
    assert "ModelAPI_Feature::document" in summary


def test_class_summary_picks_the_member_link_not_the_return_type():
    """The first <a> in a row can be the return type; the member is the link whose
    href matches the row's own anchor."""
    summary = extract_class_summary(
        _make_qualified_class_page(), {"a002"}, class_name="ModelAPI_Feature"
    )
    assert "ModelAPI_Feature::document" in summary
    assert "ModelAPI_Feature::ModelAPI_Document" not in summary


def test_class_summary_repeats_the_class_name_once_per_member():
    summary = extract_class_summary(
        _make_qualified_class_page(), {"a001", "a002"}, class_name="ModelAPI_Feature"
    )
    # once in each of the two member lines; the header is added by process_docs
    assert summary.count("ModelAPI_Feature::") == 2


def test_class_summary_keeps_signature_and_brief_alongside_the_qualified_name():
    summary = extract_class_summary(
        _make_qualified_class_page(), {"a001"}, class_name="ModelAPI_Feature"
    )
    assert "Returns the unique kind of a feature." in summary
    assert "()=0" in summary


def test_class_summary_without_class_name_is_unchanged():
    """class_name is optional; omitting it must preserve the previous output."""
    summary = extract_class_summary(_make_qualified_class_page(), {"a001", "a002"})
    assert "ModelAPI_Feature::" not in summary
    assert "getKind" in summary

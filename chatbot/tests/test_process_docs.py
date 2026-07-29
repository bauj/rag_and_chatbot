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


def _processor_with_citation_url(tmp_path):
    """
    Processor whose module HAS a citation URL — the case that exposed the bug.

    Note DocumentationProcessor.__init__ ignores config["modules"]; only
    from_config_file() feeds them through add_module(). Without this explicit
    add_module call MODULE_INFO stays empty, base_url is '', and the old buggy
    code silently fell back to a page-unique id — hiding the collision.
    """
    processor = DocumentationProcessor(_memitem_config(tmp_path))
    processor.add_module(
        name="test_module",
        description="test",
        dev_path=str(tmp_path),
        url_for_sources_citation="https://docs.example.org/latest/tui/TEST",
    )
    return processor


def test_section_id_is_page_scoped_not_module_scoped(tmp_path):
    """
    Two different pages sharing a heading must not share a section_id.

    section_id used to be built from the MODULE base_url, which is identical for
    every page, so any two pages with the same heading collided. Since
    _rerank_and_expand() deduplicates by section_id, those collisions silently
    dropped good chunks: in the SHAPER corpus a single '#root' id covered 1288
    unrelated pages, starving results below top_n_after_rerank.
    """
    body = "<p>" + " ".join(["word"] * 120) + "</p>"
    first = tmp_path / "pageOne.html"
    second = tmp_path / "pageTwo.html"
    first.write_text(f"<html><body><h1>Overview</h1>{body}</body></html>")
    second.write_text(f"<html><body><h1>Overview</h1>{body}</body></html>")

    processor = _processor_with_citation_url(tmp_path)
    chunks_one = processor.process_file(str(first), "test_module", "dev")
    chunks_two = processor.process_file(str(second), "test_module", "dev")

    ids_one = {c.metadata["section_id"] for c in chunks_one}
    ids_two = {c.metadata["section_id"] for c in chunks_two}

    assert ids_one and ids_two
    assert not (ids_one & ids_two), (
        f"section_id collision across pages: {ids_one & ids_two}"
    )
    # And the id should name the page it came from.
    assert all("pageOne" in i for i in ids_one)
    assert all("pageTwo" in i for i in ids_two)


def test_memitem_section_id_is_page_scoped(tmp_path):
    """The per-memitem branch must be page-scoped too, not just section chunking."""
    html_file = tmp_path / "classModelAPI__Feature.html"
    html_file.write_text(_memitem_html())
    processor = _processor_with_citation_url(tmp_path)
    chunks = processor.process_file(str(html_file), "test_module", "dev")

    section_ids = [c.metadata["section_id"] for c in chunks]
    assert section_ids
    assert all("classModelAPI__Feature" in sid for sid in section_ids)
    # Distinct symbols on the same page keep distinct ids.
    assert len(set(section_ids)) == len(set(c.metadata["symbol_name"] for c in chunks))


def test_process_file_without_memitems_uses_section_path(tmp_path):
    html = "<html><body><h1>Overview</h1><p>" + " ".join(["word"] * 120) + "</p></body></html>"
    html_file = tmp_path / "overview.html"
    html_file.write_text(html)
    processor = DocumentationProcessor(_memitem_config(tmp_path))
    chunks = processor.process_file(str(html_file), "test_module", "dev")

    assert len(chunks) >= 1
    assert "symbol_name" not in chunks[0].metadata


def _all_inherited_memitem_html():
    """A page whose only memitem is an inherited copy (task #91).

    Doxygen produces these for classes that declare nothing of their own. Once
    inherited memitems are skipped, the memitem branch extracts zero chunks, so
    process_file must fall back to section chunking or the page disappears from
    the corpus along with its class description.
    """
    return """<html><head><title>ExchangePlugin_ExportFormatValidator Class Reference</title></head>
    <body><div class="contents">
    <p>""" + " ".join(["Validates", "the", "export", "format", "of", "a", "shape."] * 20) + """</p>
    <a id="ainherited0" name="ainherited0"></a>
    <div class="memitem">
    <div class="memproto">
    <table class="mlabels"><tr><td class="mlabels-left">
    <table class="memname"><tr><td class="memname">void ModelAPI_Entity::emptyFunction </td></tr></table>
    </td><td class="mlabels-right"><span class="mlabel inherited">inherited</span></td></tr></table>
    </div>
    <div class="memdoc"><p>""" + " ".join(["Copied", "from", "the", "base."] * 20) + """</p></div>
    </div>
    </div></body></html>"""


def test_page_with_only_inherited_memitems_still_produces_chunks(tmp_path):
    html_file = tmp_path / "classExchangePlugin__ExportFormatValidator.html"
    html_file.write_text(_all_inherited_memitem_html())
    processor = DocumentationProcessor(_memitem_config(tmp_path))
    chunks = processor.process_file(str(html_file), "test_module", "dev")

    assert chunks, "page fell out of the corpus entirely"
    assert any("export format" in c.content.lower() for c in chunks)


def test_page_with_only_inherited_memitems_uses_section_path(tmp_path):
    """Falling back means section chunks, so no memitem-only metadata."""
    html_file = tmp_path / "classExchangePlugin__ExportFormatValidator.html"
    html_file.write_text(_all_inherited_memitem_html())
    processor = DocumentationProcessor(_memitem_config(tmp_path))
    chunks = processor.process_file(str(html_file), "test_module", "dev")

    assert all(not c.metadata.get("symbol_name") for c in chunks)


def test_process_file_skips_inherited_memitems(tmp_path):
    """A declared member alongside an inherited copy: only the declared one survives."""
    html = _memitem_html().replace(
        "</div></body></html>",
        """<a id="cinherited9" name="cinherited9"></a>
        <div class="memitem">
        <div class="memproto">
        <table class="mlabels"><tr><td class="mlabels-left">
        <table class="memname"><tr><td class="memname">void ModelAPI_Entity::emptyFunction </td></tr></table>
        </td><td class="mlabels-right"><span class="mlabel inherited">inherited</span></td></tr></table>
        </div>
        <div class="memdoc"><p>""" + " ".join(["Copied", "from", "base."] * 20) + """</p></div>
        </div>
        </div></body></html>""",
    )
    html_file = tmp_path / "classModelAPI__Feature.html"
    html_file.write_text(html)
    processor = DocumentationProcessor(_memitem_config(tmp_path))
    chunks = processor.process_file(str(html_file), "test_module", "dev")

    symbol_names = {c.metadata["symbol_name"] for c in chunks}
    assert symbol_names == {
        "bool ModelAPI_Feature::execute",
        "bool ModelAPI_Feature::isValid",
    }


def _class_page_html():
    """A class page with a memberdecls summary plus declared and inherited memitems."""
    body = " ".join(["Feature", "function", "of", "this", "operation."] * 15)
    return """<html><head><title>ModelAPI_Feature Class Reference</title></head>
    <body><div class="contents">
    <div class="textblock"><p>""" + body + """</p></div>
    <table class="memberdecls">
    <tr class="heading"><td><h2 class="groupheader">Public Member Functions</h2></td></tr>
    <tr class="memitem:a1a2b3c4d5"><td class="memItemRight">bool execute (const std::string &amp;name)</td></tr>
    <tr class="memdesc:a1a2b3c4d5"><td class="mdescRight">Executes the feature and returns success.</td></tr>
    <tr class="memitem:ainherited0"><td class="memItemRight">void emptyFunction () const</td></tr>
    <tr class="memdesc:ainherited0"><td class="mdescRight">Empty function for virtualisation.</td></tr>
    </table>
    <a id="a1a2b3c4d5" name="a1a2b3c4d5"></a>
    <div class="memitem">
    <div class="memproto"><table class="memname"><tr>
    <td class="memname">bool ModelAPI_Feature::execute </td></tr></table></div>
    <div class="memdoc"><p>""" + " ".join(["Executes", "the", "feature."] * 20) + """</p></div>
    </div>
    <a id="ainherited0" name="ainherited0"></a>
    <div class="memitem">
    <div class="memproto">
    <table class="mlabels"><tr><td class="mlabels-left">
    <table class="memname"><tr><td class="memname">void ModelAPI_Entity::emptyFunction </td></tr></table>
    </td><td class="mlabels-right"><span class="mlabel inherited">inherited</span></td></tr></table>
    </div>
    <div class="memdoc"><p>""" + " ".join(["Copied", "from", "base."] * 20) + """</p></div>
    </div>
    </div></body></html>"""


def _summary_chunks(tmp_path):
    html_file = tmp_path / "classModelAPI__Feature.html"
    html_file.write_text(_class_page_html())
    processor = DocumentationProcessor(_memitem_config(tmp_path))
    chunks = processor.process_file(str(html_file), "test_module", "dev")
    return chunks, [c for c in chunks if not c.metadata.get("anchor_id")]


def test_class_page_emits_a_summary_chunk(tmp_path):
    """Without this, no chunk in the corpus lists a class's methods (task #92)."""
    _, summaries = _summary_chunks(tmp_path)
    assert len(summaries) == 1
    assert "Public Member Functions" in summaries[0].content


def test_summary_chunk_lists_declared_members(tmp_path):
    _, summaries = _summary_chunks(tmp_path)
    assert "execute" in summaries[0].content
    assert "Executes the feature and returns success." in summaries[0].content


def test_summary_chunk_excludes_inherited_members(tmp_path):
    _, summaries = _summary_chunks(tmp_path)
    assert "emptyFunction" not in summaries[0].content


def test_summary_chunk_url_has_no_fragment(tmp_path):
    """It describes the page, not a symbol, so it must not cite an anchor."""
    _, summaries = _summary_chunks(tmp_path)
    assert "#" not in summaries[0].url


def test_summary_chunk_section_id_is_page_scoped(tmp_path):
    _, summaries = _summary_chunks(tmp_path)
    section_id = summaries[0].metadata["section_id"]
    assert section_id.endswith("#__summary")
    assert "classModelAPI__Feature.html" in section_id


def test_memitem_chunks_still_emitted_alongside_summary(tmp_path):
    """The summary is additional context, not a replacement."""
    chunks, summaries = _summary_chunks(tmp_path)
    memitems = [c for c in chunks if c.metadata.get("anchor_id")]
    assert len(summaries) == 1
    assert {c.metadata["symbol_name"] for c in memitems} == {"bool ModelAPI_Feature::execute"}


def test_no_summary_chunk_when_page_has_no_memberdecls(tmp_path):
    html_file = tmp_path / "classModelAPI__Feature.html"
    html_file.write_text(_memitem_html())
    processor = DocumentationProcessor(_memitem_config(tmp_path))
    chunks = processor.process_file(str(html_file), "test_module", "dev")
    assert all(c.metadata.get("anchor_id") for c in chunks)


def test_every_summary_chunk_names_its_class(tmp_path):
    """Doxygen's member table omits the class prefix on each row, so without this
    the summary never contains the class name and neither BM25 nor the embedder
    can match "methods of ModelAPI_Feature" to it (task #92)."""
    _, summaries = _summary_chunks(tmp_path)
    assert summaries, "no summary chunk emitted"
    assert all("ModelAPI_Feature" in c.content for c in summaries)


def test_long_summary_names_its_class_in_every_chunk(tmp_path):
    """A big class splits into several summary chunks; each must stand alone."""
    rows = "".join(
        f'<tr class="memitem:a{i:04d}"><td>virtual void method{i} (const std::string &amp;argument{i})</td></tr>'
        f'<tr class="memdesc:a{i:04d}"><td>Performs operation number {i} on the given argument value.</td></tr>'
        for i in range(60)
    )
    details = "".join(
        f'<a id="a{i:04d}" name="a{i:04d}"></a><div class="memitem">'
        f'<div class="memproto"><table class="memname"><tr>'
        f'<td class="memname">void ModelAPI_Feature::method{i} </td></tr></table></div>'
        f'<div class="memdoc"><p>{" ".join(["Performs", "the", "operation."] * 20)}</p></div></div>'
        for i in range(60)
    )
    html = f"""<html><head><title>ModelAPI_Feature Class Reference</title></head>
    <body><div class="contents">
    <div class="textblock"><p>{" ".join(["Feature", "function."] * 30)}</p></div>
    <table class="memberdecls">
    <tr class="heading"><td><h2 class="groupheader">Public Member Functions</h2></td></tr>
    {rows}
    </table>
    {details}
    </div></body></html>"""
    html_file = tmp_path / "classModelAPI__Feature.html"
    html_file.write_text(html)
    processor = DocumentationProcessor(_memitem_config(tmp_path))
    chunks = processor.process_file(str(html_file), "test_module", "dev")
    summaries = [c for c in chunks if not c.metadata.get("anchor_id")]

    assert len(summaries) > 1, "expected the summary to split across chunks"
    assert all("ModelAPI_Feature" in c.content for c in summaries)


def _nested_config(tmp_path, root):
    cfg = _memitem_config(tmp_path)
    cfg["modules"] = {}
    return cfg


def _processor_with_module_root(tmp_path, root):
    """Processor whose module doc root is `root`, with a citation URL configured.

    add_module() is the only path that registers a module root and citation URL —
    __init__ ignores config["modules"] — so it must be called explicitly.
    """
    processor = DocumentationProcessor(_memitem_config(tmp_path))
    processor.add_module(
        name="SHAPER",
        description="",
        user_path=str(root),
        url_for_sources_citation="https://docs.example.org/SHAPER",
    )
    return processor


def test_pages_in_different_subdirs_get_different_urls(tmp_path):
    """SketchPlugin/pointFeature.html and ConstructionPlugin/pointFeature.html are
    distinct pages; a bare filename collapsed them onto one URL and one section_id
    prefix, so dedup silently dropped chunks (same failure as the #88 collision)."""
    root = tmp_path / "gui"
    body = " ".join(["Point", "feature", "creates", "a", "point."] * 20)
    for plugin, text in (("SketchPlugin", "sketch " + body), ("ConstructionPlugin", "construction " + body)):
        d = root / plugin
        d.mkdir(parents=True)
        (d / "pointFeature.html").write_text(
            f"<html><head><title>Point</title></head><body><div class='body'>"
            f"<h1>Point</h1><p>{text}</p></div></body></html>"
        )

    processor = _processor_with_module_root(tmp_path, root)
    sketch = processor.process_file(str(root / "SketchPlugin" / "pointFeature.html"), "SHAPER", "user")
    construction = processor.process_file(str(root / "ConstructionPlugin" / "pointFeature.html"), "SHAPER", "user")

    assert sketch and construction
    urls = {c.url.split("#")[0] for c in sketch} | {c.url.split("#")[0] for c in construction}
    assert len(urls) == 2, f"pages collapsed onto one URL: {urls}"

    sketch_sections = {c.metadata["section_id"] for c in sketch}
    construction_sections = {c.metadata["section_id"] for c in construction}
    assert not (sketch_sections & construction_sections), "section_id prefixes collide"


def test_url_keeps_the_subdirectory_path(tmp_path):
    root = tmp_path / "gui"
    d = root / "SketchPlugin"
    d.mkdir(parents=True)
    (d / "pointFeature.html").write_text(
        "<html><head><title>Point</title></head><body><div class='body'><h1>Point</h1><p>"
        + " ".join(["Point", "feature."] * 40) + "</p></div></body></html>"
    )
    processor = _processor_with_module_root(tmp_path, root)
    chunks = processor.process_file(str(d / "pointFeature.html"), "SHAPER", "user")

    assert chunks
    assert chunks[0].url.startswith(
        "https://docs.example.org/SHAPER/SketchPlugin/pointFeature.html"
    ), chunks[0].url


def test_url_falls_back_to_filename_outside_any_module_root(tmp_path):
    """Files processed outside a registered module root keep the old behaviour."""
    html_file = tmp_path / "classModelAPI__Feature.html"
    html_file.write_text(_memitem_html())
    processor = DocumentationProcessor(_memitem_config(tmp_path))
    chunks = processor.process_file(str(html_file), "test_module", "dev")

    assert chunks
    assert chunks[0].url.split("#")[0] == "classModelAPI__Feature.html"


def _page(path: Path, title: str = "Point"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"<html><head><title>{title}</title></head><body><div class='body'><h1>{title}</h1><p>"
        + " ".join(["Feature", "documentation", "text."] * 30) + "</p></div></body></html>"
    )


def test_citation_url_can_differ_per_doc_category(tmp_path):
    """SALOME publishes dev docs under /tui/ and user docs under /gui/. One URL per
    module cited every user page under the dev-doc space."""
    dev_root, user_root = tmp_path / "tui", tmp_path / "gui"
    _page(dev_root / "classFoo.html", "Foo Class Reference")
    _page(user_root / "SketchPlugin" / "pointFeature.html")

    processor = DocumentationProcessor(_memitem_config(tmp_path))
    processor.add_module(
        name="SHAPER",
        dev_path=str(dev_root),
        user_path=str(user_root),
        url_for_sources_citation={
            "dev": "https://docs.example.org/latest/tui/SHAPER",
            "user": "https://docs.example.org/latest/gui/SHAPER",
        },
    )

    dev = processor.process_file(str(dev_root / "classFoo.html"), "SHAPER", "dev")
    user = processor.process_file(str(user_root / "SketchPlugin" / "pointFeature.html"), "SHAPER", "user")

    assert dev[0].url.startswith("https://docs.example.org/latest/tui/SHAPER/classFoo.html")
    assert user[0].url.startswith(
        "https://docs.example.org/latest/gui/SHAPER/SketchPlugin/pointFeature.html"
    )


def test_citation_url_still_accepts_a_plain_string(tmp_path):
    """Existing configs pass one URL for the whole module; that must keep working."""
    user_root = tmp_path / "gui"
    _page(user_root / "SketchPlugin" / "pointFeature.html")

    processor = DocumentationProcessor(_memitem_config(tmp_path))
    processor.add_module(
        name="SHAPER",
        user_path=str(user_root),
        url_for_sources_citation="https://docs.example.org/latest/tui/SHAPER",
    )
    chunks = processor.process_file(str(user_root / "SketchPlugin" / "pointFeature.html"), "SHAPER", "user")

    assert chunks[0].url.startswith(
        "https://docs.example.org/latest/tui/SHAPER/SketchPlugin/pointFeature.html"
    )

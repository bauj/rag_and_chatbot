import json
import sys
from pathlib import Path

CHATBOT_DIR = Path(__file__).parent.parent
if str(CHATBOT_DIR) not in sys.path:
    sys.path.insert(0, str(CHATBOT_DIR))

from langchain_core.documents import Document

from core.bm25_index import _tokenize, _flatten_metadata, BM25Index, reciprocal_rank_fusion


def test_tokenize_lowercases_and_splits_on_non_word_chars():
    assert _tokenize("ModelAPI_Feature::execute") == ["modelapi_feature", "execute"]


def test_tokenize_empty_string_returns_empty_list():
    assert _tokenize("") == []


def test_flatten_metadata_maps_expected_keys():
    row = {
        "title": "ModelAPI_Feature Class Reference",
        "content": "bool ModelAPI_Feature::execute(...)",
        "url": "https://docs.example.org/classModelAPI__Feature.html",
        "doc_type": "class",
        "hierarchy": "bool ModelAPI_Feature::execute",
        "chunk_id": 0,
        "module": "SHAPER",
        "doc_category": "dev",
        "metadata": {
            "parent_doc_id": "https://docs.example.org",
            "chunk_position": "1/1",
            "quality_score": 0.9,
            "has_code": False,
            "section_id": "https://docs.example.org#a1a2b3",
            "section_text": "bool ModelAPI_Feature::execute(...)",
        },
    }
    flat = _flatten_metadata(row)
    assert flat["url"] == row["url"]
    assert flat["section_id"] == "https://docs.example.org#a1a2b3"
    assert flat["chunk_position"] == "1/1"
    assert flat["module"] == "SHAPER"
    assert flat["doc_category"] == "dev"


def _write_jsonl(tmp_path, rows):
    path = tmp_path / "test_docs.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _row(content, module="SHAPER", doc_category="dev", url="u1", section_id="s1",
         chunk_position="1/1", title="T", chunk_id=0):
    return {
        "title": title, "content": content, "url": url, "doc_type": "class",
        "hierarchy": "h", "chunk_id": chunk_id, "module": module, "doc_category": doc_category,
        "metadata": {
            "parent_doc_id": "", "chunk_position": chunk_position, "quality_score": 0.5,
            "has_code": False, "section_id": section_id, "section_text": content,
        },
    }


def test_bm25_search_returns_documents_ranked_by_relevance(tmp_path):
    rows = [
        _row("The ModelAPI_Feature execute method runs the feature.", url="u1", section_id="s1"),
        _row("Unrelated content about tutorials and installation.", url="u2", section_id="s2"),
    ]
    path = _write_jsonl(tmp_path, rows)
    index = BM25Index(str(path))
    results = index.search("ModelAPI_Feature execute", k=2)
    assert len(results) >= 1
    assert results[0].metadata["url"] == "u1"


def test_bm25_search_applies_module_filter(tmp_path):
    rows = [
        _row("execute feature", module="SHAPER", url="u1", section_id="s1"),
        _row("execute feature", module="GEOM", url="u2", section_id="s2"),
    ]
    path = _write_jsonl(tmp_path, rows)
    index = BM25Index(str(path))
    results = index.search("execute feature", k=5, module_filter="GEOM")
    urls = [r.metadata["url"] for r in results]
    assert urls == ["u2"]


def test_bm25_search_applies_doc_category_filter(tmp_path):
    rows = [
        _row("execute feature", doc_category="dev", url="u1", section_id="s1"),
        _row("execute feature", doc_category="user", url="u2", section_id="s2"),
    ]
    path = _write_jsonl(tmp_path, rows)
    index = BM25Index(str(path))
    results = index.search("execute feature", k=5, doc_category_filter="user")
    urls = [r.metadata["url"] for r in results]
    assert urls == ["u2"]


def test_bm25_search_respects_k_limit(tmp_path):
    rows = [_row(f"execute feature number {i}", url=f"u{i}", section_id=f"s{i}") for i in range(5)]
    path = _write_jsonl(tmp_path, rows)
    index = BM25Index(str(path))
    results = index.search("execute feature", k=2)
    assert len(results) == 2


def test_bm25_search_empty_corpus_returns_empty_list(tmp_path):
    path = _write_jsonl(tmp_path, [])
    index = BM25Index(str(path))
    assert index.search("anything", k=5) == []


def _doc(content, url, section_id, chunk_position="1/1"):
    return Document(page_content=content, metadata={
        "url": url, "section_id": section_id, "chunk_position": chunk_position,
    })


def test_rrf_ranks_doc_appearing_in_both_lists_highest():
    # "shared" is ranked #1 in list A and #2 in list B — should out-rank a doc
    # that only appears once, even at rank #1 of its own list.
    shared = _doc("shared content", url="u1", section_id="s1")
    only_a = _doc("only in a", url="u2", section_id="s2")
    only_b = _doc("only in b", url="u3", section_id="s3")

    list_a = [shared, only_a]
    list_b = [only_b, shared]

    result = reciprocal_rank_fusion([list_a, list_b], k=3)
    result_keys = [(d.metadata["url"]) for d in result]
    assert result_keys[0] == "u1"


def test_rrf_deduplicates_by_url_section_chunk_position():
    doc_a = _doc("version from list a", url="u1", section_id="s1")
    doc_b = _doc("version from list b", url="u1", section_id="s1")
    result = reciprocal_rank_fusion([[doc_a], [doc_b]], k=5)
    assert len(result) == 1


def test_rrf_truncates_to_k():
    docs = [_doc(f"content {i}", url=f"u{i}", section_id=f"s{i}") for i in range(5)]
    result = reciprocal_rank_fusion([docs], k=2)
    assert len(result) == 2


def test_rrf_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([[], []], k=5) == []


# ---------------------------------------------------------------------------
# Title/identifier channel (task: retrieval pool + title channel)
#
# A question naming a class ("What are the public methods of the
# ModelAPI_Feature class?") is an entity lookup: the discriminating token is
# diluted across ~1,800 chunk bodies (every subclass memitem mentions the
# parent class) but is the whole of the ~112 chunk titles that carry it — 16x
# sharper. search_titles ranks by title-only BM25 and collapses each page's
# many chunks down to its one page-level (first) chunk, so a title match
# doesn't flood the RRF pool with near-duplicate hits of the same page.
# ---------------------------------------------------------------------------

def test_search_titles_ranks_by_title_not_body_relevance(tmp_path):
    """The whole point of this channel: score the TITLE, not the diluted body."""
    rows = [
        _row("body mentions ModelAPI_Feature repeatedly ModelAPI_Feature ModelAPI_Feature",
             title="Other Class Reference", url="u1", section_id="s1"),
        _row("unrelated body text", title="ModelAPI_Feature Class Reference", url="u2", section_id="s2"),
        # A third, wholly unrelated doc — with only two docs, BM25's IDF formula
        # gives exactly 0 for any term appearing in half the corpus, which would
        # mask the effect this test exists to demonstrate.
        _row("installation guide", title="Unrelated Tutorial", url="u3", section_id="s3"),
    ]
    path = _write_jsonl(tmp_path, rows)
    index = BM25Index(str(path))
    results = index.search_titles("ModelAPI_Feature", k=5)
    assert results[0].metadata["url"] == "u2"


def test_search_titles_collapses_multiple_chunks_of_one_page_to_the_first(tmp_path):
    """All chunks of a page share its title; a raw title search must not flood
    the pool with every chunk of the same page — only the page's first
    (summary) chunk should survive."""
    rows = [
        _row("chunk 2 body", title="ModelAPI_Feature Class Reference",
             url="page1#frag2", section_id="s2", chunk_position="2/3", chunk_id=1),
        _row("chunk 1 body (the summary)", title="ModelAPI_Feature Class Reference",
             url="page1", section_id="s1", chunk_position="1/3", chunk_id=0),
        _row("chunk 3 body", title="ModelAPI_Feature Class Reference",
             url="page1#frag3", section_id="s3", chunk_position="3/3", chunk_id=2),
        _row("other page body", title="Other Class Reference",
             url="page2", section_id="o1", chunk_position="1/1", chunk_id=0),
    ]
    path = _write_jsonl(tmp_path, rows)
    index = BM25Index(str(path))
    results = index.search_titles("ModelAPI_Feature", k=5)
    page1_hits = [r for r in results if r.metadata["url"].split("#")[0] == "page1"]
    assert len(page1_hits) == 1
    assert page1_hits[0].page_content == "chunk 1 body (the summary)"


def test_search_titles_breaks_score_ties_deterministically(tmp_path):
    """Two pages with identical titles score identically; ordering must not
    depend on incidental dict/insertion order."""
    rows = [
        _row("body", title="Alpha Beta", url="pageB", section_id="b1"),
        _row("body", title="Alpha Beta", url="pageA", section_id="a1"),
    ]
    path = _write_jsonl(tmp_path, rows)
    index = BM25Index(str(path))
    results = index.search_titles("Alpha Beta", k=5)
    assert [r.metadata["url"] for r in results] == ["pageA", "pageB"]


def test_search_titles_applies_module_filter(tmp_path):
    rows = [
        _row("body", title="Alpha", module="SHAPER", url="u1", section_id="s1"),
        _row("body", title="Alpha", module="GEOM", url="u2", section_id="s2"),
    ]
    path = _write_jsonl(tmp_path, rows)
    index = BM25Index(str(path))
    results = index.search_titles("Alpha", k=5, module_filter="GEOM")
    assert [r.metadata["url"] for r in results] == ["u2"]


def test_search_titles_applies_doc_category_filter(tmp_path):
    rows = [
        _row("body", title="Alpha", doc_category="dev", url="u1", section_id="s1"),
        _row("body", title="Alpha", doc_category="user", url="u2", section_id="s2"),
    ]
    path = _write_jsonl(tmp_path, rows)
    index = BM25Index(str(path))
    results = index.search_titles("Alpha", k=5, doc_category_filter="user")
    assert [r.metadata["url"] for r in results] == ["u2"]


def test_search_titles_respects_k_limit(tmp_path):
    rows = [_row("body", title=f"Alpha Page {i}", url=f"u{i}", section_id=f"s{i}") for i in range(5)]
    path = _write_jsonl(tmp_path, rows)
    index = BM25Index(str(path))
    results = index.search_titles("Alpha", k=2)
    assert len(results) == 2


def test_search_titles_empty_corpus_returns_empty_list(tmp_path):
    path = _write_jsonl(tmp_path, [])
    index = BM25Index(str(path))
    assert index.search_titles("anything", k=5) == []


# ---------------------------------------------------------------------------
# No-match rows must not be returned (task #230)
#
# get_scores() scores EVERY row, so on a query sharing no token with most of
# the corpus the tail is a run of score-0 rows ordered only by the tie-break
# (JSONL order for search(), URL order for search_titles()) — unrelated hits
# that RRF then weights like real ones.
# A row is a hit only if it contains at least one query token. That is
# deliberately not "score > 0": a term present in exactly half the corpus
# has IDF 0, and a tiny corpus can have negative-floored IDFs, so a genuine
# match can score <= 0.
# ---------------------------------------------------------------------------

def test_bm25_search_returns_nothing_when_no_row_contains_a_query_token(tmp_path):
    rows = [
        _row("execute feature", url="u1", section_id="s1"),
        _row("installation guide", url="u2", section_id="s2"),
        _row("tutorial overview", url="u3", section_id="s3"),
    ]
    path = _write_jsonl(tmp_path, rows)
    index = BM25Index(str(path))
    assert index.search("comment créer une esquisse", k=3) == []


def test_bm25_search_drops_rows_without_a_query_token_even_under_k(tmp_path):
    rows = [
        _row("installation guide", url="u1", section_id="s1"),
        _row("execute feature", url="u2", section_id="s2"),
        _row("tutorial overview", url="u3", section_id="s3"),
    ]
    path = _write_jsonl(tmp_path, rows)
    index = BM25Index(str(path))
    results = index.search("execute", k=3)
    assert [r.metadata["url"] for r in results] == ["u2"]


def test_bm25_search_keeps_a_match_whose_term_has_zero_idf(tmp_path):
    """'feature' is in exactly 2 of 4 rows, so its IDF is exactly 0 and both
    matching rows score 0 — they are still matches and must be returned."""
    rows = [
        _row("feature one", url="u1", section_id="s1"),
        _row("feature two", url="u2", section_id="s2"),
        _row("installation guide", url="u3", section_id="s3"),
        _row("tutorial overview", url="u4", section_id="s4"),
    ]
    path = _write_jsonl(tmp_path, rows)
    index = BM25Index(str(path))
    results = index.search("feature", k=4)
    assert sorted(r.metadata["url"] for r in results) == ["u1", "u2"]


def test_search_titles_returns_nothing_when_no_title_contains_a_query_token(tmp_path):
    rows = [
        _row("body", title="Alpha Class Reference", url="u1", section_id="s1"),
        _row("body", title="Beta Class Reference", url="u2", section_id="s2"),
        _row("body", title="Installation Guide", url="u3", section_id="s3"),
    ]
    path = _write_jsonl(tmp_path, rows)
    index = BM25Index(str(path))
    assert index.search_titles("comment créer une esquisse", k=3) == []


def test_search_titles_drops_pages_without_a_query_token_even_under_k(tmp_path):
    rows = [
        _row("body", title="Alpha Class Reference", url="u1", section_id="s1"),
        _row("body", title="Beta Page", url="u2", section_id="s2"),
        _row("body", title="Installation Guide", url="u3", section_id="s3"),
    ]
    path = _write_jsonl(tmp_path, rows)
    index = BM25Index(str(path))
    results = index.search_titles("Alpha", k=3)
    assert [r.metadata["url"] for r in results] == ["u1"]

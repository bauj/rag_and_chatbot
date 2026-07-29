"""
BM25 sparse retrieval for hybrid search.

Loads the same JSONL corpus produced by extraction/process_docs.py
(one dataclasses.asdict(DocumentChunk) per line) and provides keyword-based
search with the same module/doc_category filtering semantics as the Chroma
vector retriever, so results from both can be fused (see reciprocal_rank_fusion
below and its use in rag_chatbot.py's _hybrid_retrieve).
"""

import json
import re
from typing import Dict, List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-word characters."""
    return re.findall(r"\w+", text.lower())


def _page_key(row: dict) -> str:
    """
    Identify the page a chunk belongs to, for collapsing a title search's
    per-chunk hits down to one Document per page (see BM25Index.search_titles).
    Prefers the chunk's URL with any '#fragment' anchor stripped — Doxygen and
    Sphinx both anchor same-page sections that way — and falls back to
    metadata.file for rows that carry no url.
    """
    url = row.get('url', '')
    if url:
        return url.split('#')[0]
    return row.get('metadata', {}).get('file', '')


def _first_chunk_rank(row: dict) -> tuple:
    """
    Sort key that picks a page's first (summary) chunk out of a group of
    chunks sharing one page_key. chunk_position's numerator ("1/4" -> 1) is
    the primary signal, since it directly encodes a chunk's ordinal position
    on the page; chunk_id is a secondary tiebreak for rows without usable
    chunk_position metadata. Lower sorts first.
    """
    position = row.get('metadata', {}).get('chunk_position', '')
    numerator = None
    if isinstance(position, str) and '/' in position:
        head = position.split('/', 1)[0].strip()
        if head.isdigit():
            numerator = int(head)

    chunk_id = row.get('chunk_id')
    try:
        chunk_id = int(chunk_id)
    except (TypeError, ValueError):
        chunk_id = None

    return (
        numerator if numerator is not None else float('inf'),
        chunk_id if chunk_id is not None else float('inf'),
    )


def _flatten_metadata(row: dict) -> dict:
    """
    Map a parsed JSONL row to the same flattened metadata shape that
    extraction/process_docs.py's create_chromadb() writes into Chroma,
    so BM25 and vector search results share compatible metadata keys
    for fusion (see _fusion_key).
    """
    meta = row.get('metadata', {})
    return {
        'title': row.get('title', ''),
        'url': row.get('url', ''),
        'doc_type': row.get('doc_type', ''),
        'hierarchy': row.get('hierarchy', ''),
        'chunk_id': row.get('chunk_id', 0),
        'module': row.get('module', ''),
        'doc_category': row.get('doc_category', ''),
        'parent_doc_id': meta.get('parent_doc_id', ''),
        'chunk_position': meta.get('chunk_position', ''),
        'quality_score': meta.get('quality_score', 0.0),
        'has_code': meta.get('has_code', False),
        'section_id': meta.get('section_id', ''),
        'section_text': meta.get('section_text', ''),
    }


class BM25Index:
    """Keyword (BM25) search over the extraction pipeline's JSONL corpus."""

    def __init__(self, jsonl_path: str):
        self.rows: List[dict] = []
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

        tokenized_corpus = [_tokenize(row.get('content', '')) for row in self.rows]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

        # Second index over titles only, for search_titles(). Cheap to build alongside
        # the body index (same corpus, already loaded) and kept even when the title
        # channel is disabled by config — the flag only gates whether _hybrid_retrieve
        # queries it, not whether it exists.
        tokenized_titles = [_tokenize(row.get('title', '')) for row in self.rows]
        self._title_bm25 = BM25Okapi(tokenized_titles) if tokenized_titles else None

    def search(
        self,
        query: str,
        k: int,
        module_filter: Optional[str] = None,
        doc_category_filter: Optional[str] = None,
    ) -> List[Document]:
        """Return up to k Documents ranked by BM25 score, honoring optional filters."""
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenize(query))

        candidates = []
        for idx, score in enumerate(scores):
            row = self.rows[idx]
            if module_filter and row.get('module') != module_filter:
                continue
            if doc_category_filter and row.get('doc_category') != doc_category_filter:
                continue
            candidates.append((score, idx))

        candidates.sort(key=lambda pair: pair[0], reverse=True)

        return [
            Document(page_content=self.rows[idx].get('content', ''), metadata=_flatten_metadata(self.rows[idx]))
            for _score, idx in candidates[:k]
        ]

    def search_titles(
        self,
        query: str,
        k: int,
        module_filter: Optional[str] = None,
        doc_category_filter: Optional[str] = None,
    ) -> List[Document]:
        """
        Rank pages by BM25 score over their title only (not body content), for
        entity-lookup style queries ("what are the methods of the ModelAPI_Feature
        class?") where the identifying token is the whole of a page's title but is
        diluted among ~100+ tokens — and, in a corpus with Doxygen inherited-member
        copies, thousands of *other pages'* bodies — when scored against content.

        Every chunk of a page shares that page's title, so a raw per-chunk title
        search would return many near-duplicate hits of the same page and flood
        whatever pool this feeds. Collapse to one Document per page (identified by
        _page_key) before applying k, keeping the page's first/summary chunk
        (lowest _first_chunk_rank) as the representative. Ties in score are broken
        by page key so results are reproducible rather than depending on dict
        iteration order.

        Honors the same module/doc_category filters as search(), applied before
        collapsing so a filtered-out chunk cannot still nominate its page.
        """
        if self._title_bm25 is None:
            return []

        scores = self._title_bm25.get_scores(_tokenize(query))

        best_idx_by_page: Dict[str, int] = {}
        for idx, row in enumerate(self.rows):
            if module_filter and row.get('module') != module_filter:
                continue
            if doc_category_filter and row.get('doc_category') != doc_category_filter:
                continue
            page = _page_key(row)
            current = best_idx_by_page.get(page)
            if current is None or _first_chunk_rank(row) < _first_chunk_rank(self.rows[current]):
                best_idx_by_page[page] = idx

        candidates = [(scores[idx], idx) for idx in best_idx_by_page.values()]
        candidates.sort(key=lambda pair: (-pair[0], self.rows[pair[1]].get('url', '')))

        return [
            Document(page_content=self.rows[idx].get('content', ''), metadata=_flatten_metadata(self.rows[idx]))
            for _score, idx in candidates[:k]
        ]


def _fusion_key(doc: Document) -> tuple:
    """Identity a chunk shares across vector and BM25 result sets."""
    m = doc.metadata
    return (m.get('url', ''), m.get('section_id', ''), m.get('chunk_position', ''))


def reciprocal_rank_fusion(ranked_lists: List[List[Document]], k: int, rrf_k: int = 60) -> List[Document]:
    """
    Merge multiple ranked Document lists into one via Reciprocal Rank Fusion:
    score(doc) = sum(1 / (rrf_k + rank + 1)) across every list it appears in.
    Deduplicates by _fusion_key, keeping the first-seen Document per key.
    """
    scores: Dict[tuple, float] = {}
    doc_by_key: Dict[tuple, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = _fusion_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            doc_by_key.setdefault(key, doc)

    ordered_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    return [doc_by_key[key] for key in ordered_keys[:k]]

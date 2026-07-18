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

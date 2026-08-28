"""
Shared HTML parsing utilities used by:
  - extraction/process_docs.py  (at extraction time)
  - chatbot/core/agentic_chatbot.py  (at query time)
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from langchain_chroma import Chroma
from bs4 import BeautifulSoup, NavigableString, Tag
from markdownify import markdownify as _markdownify


def html_to_markdown(tag: Tag) -> str:
    """
    Convert an HTML subtree to Markdown, preserving <pre>/<code> blocks as
    fenced code instead of flattening them to prose (unlike tag.get_text()).
    """
    md = _markdownify(str(tag), heading_style='ATX', bullets='-')
    md = re.sub(r'\n{3,}', '\n\n', md).strip()
    return md


def _code_block_to_markdown(tag: Tag) -> str:
    """
    Render a code block as a fenced Markdown block.

    Handles both plain <pre> (Sphinx/generic HTML) and Doxygen's
    <div class="fragment"><div class="line">...</div>...</div> structure,
    where each source line is its own child div.
    """
    classes = tag.get('class') or []
    if tag.name == 'div' and 'fragment' in classes:
        lines = [line.get_text() for line in tag.find_all('div', class_='line')]
        code = '\n'.join(lines)
    else:
        code = tag.get_text()
    code = code.strip('\n')
    if not code.strip():
        return ''
    return f"```\n{code}\n```"


def _split_into_sections(element: Tag) -> List[tuple]:
    """
    Split a BS4 element into (heading, text) sections at h2/h3 boundaries.

    Returns list of (heading_text: str | None, section_text: str).
    If no h2/h3 tags are present, returns a single (None, full_text) entry.
    """
    sections = []
    current_heading = None
    current_parts: List[str] = []

    def process_node(node):
        nonlocal current_heading
        if isinstance(node, Tag):
            if node.name in ('h2', 'h3'):
                # Save whatever we've accumulated so far
                text = '\n'.join(p for p in current_parts if p)
                if text:
                    sections.append((current_heading, text))
                current_heading = node.get_text(strip=True)
                current_parts.clear()
            elif node.name == 'pre' or (node.name == 'div' and 'fragment' in (node.get('class') or [])):
                code_md = _code_block_to_markdown(node)
                if code_md:
                    current_parts.append(code_md)
            else:
                for child in node.children:
                    process_node(child)
        elif isinstance(node, NavigableString):
            text = node.strip()
            if text:
                current_parts.append(text)

    for child in element.children:
        process_node(child)

    # Flush last section
    text = '\n'.join(p for p in current_parts if p)
    if text:
        sections.append((current_heading, text))

    if not sections:
        fallback = html_to_markdown(element)
        return [(None, fallback)]

    return sections


def extract_sections_with_soup(soup: BeautifulSoup, doc_category: str) -> List[tuple]:
    """
    Find the main content element and split it into sections.

    Mirrors the element-selection logic of extract_content_doxygen /
    extract_content_sphinx so we operate on the same subtree.
    Returns list of (heading_text, section_text) — see _split_into_sections.
    """
    if doc_category == 'dev':
        element = soup.find('div', class_='contents') or soup.find('body')
    else:
        element = (
            soup.find('div', role='main') or
            soup.find('div', class_='document') or
            soup.find('div', class_='body') or
            soup.find('div', class_='section') or
            soup.find('article') or
            soup.find('body')
        )
    if not element:
        return [(None, '')]
    for tag in element.find_all(['script', 'style', 'noscript']):
        tag.decompose()
    return _split_into_sections(element)


def has_memitems(soup: BeautifulSoup) -> bool:
    """True if the page contains Doxygen div.memitem blocks (per-symbol docs)."""
    return soup.find('div', class_='memitem') is not None


def extract_memitems_with_soup(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """
    Extract Doxygen memitem blocks: one entry per documented symbol.

    Doxygen renders each documented member as:
        <a id="a1b2c3" name="a1b2c3"></a>
        <div class="memitem">
          <div class="memproto"> ... signature ... </div>
          <div class="memdoc"> ... description ... </div>
        </div>

    Members inherited from a base class are skipped. When a project builds its docs
    with Doxygen's INLINE_INHERITED_MEMB, each inherited member's documentation is
    copied verbatim onto every subclass page — in the SHAPER docs used as the test
    corpus that accounted for 57% of all chunks, one method reaching 181 copies. The
    declaring class's own page always carries the same documentation, so nothing is
    lost by skipping the copies. Doxygen marks them:

        <span class="mlabel inherited">inherited</span>

    Match that class rather than the rendered text: "inherited" also appears in
    ordinary prose descriptions.

    Returns a list of dicts with keys: symbol_name, signature, description,
    anchor_id. anchor_id is '' when no preceding <a id="..."> is found.
    """
    items: List[Dict[str, str]] = []
    for memitem in soup.find_all('div', class_='memitem'):
        memproto = memitem.find('div', class_='memproto')
        if memproto is None:
            continue
        if memproto.find('span', class_='inherited') is not None:
            continue
        memdoc = memitem.find('div', class_='memdoc')

        signature = memproto.get_text(separator=' ', strip=True)
        signature = re.sub(r'\s+', ' ', signature).strip()

        description = memdoc.get_text(separator='\n', strip=True) if memdoc else ''

        memname_cell = memproto.find('td', class_='memname')
        if memname_cell is not None:
            symbol_name = re.sub(r'\s+', ' ', memname_cell.get_text(strip=True)).strip()
        else:
            symbol_name = signature[:80]

        anchor_id = ''
        anchor = memitem.find_previous_sibling('a', id=True)
        if anchor is not None:
            anchor_id = anchor.get('id', '')

        items.append({
            'symbol_name': symbol_name,
            'signature': signature,
            'description': description,
            'anchor_id': anchor_id,
        })
    return items


def _member_name_for_anchor(row: Tag, anchor: str) -> str:
    """
    The member's own name from a memberdecls row.

    Take the link whose href ends in this row's anchor, NOT the first link in the
    row: the leading link is often the return type, so
    'virtual std::shared_ptr< ModelAPI_Document > document () const' would yield
    'ModelAPI_Document' instead of 'document'.
    """
    for link in row.find_all('a'):
        if (link.get('href') or '').endswith('#' + anchor):
            return link.get_text(strip=True)
    return ''


def extract_class_summary(soup: BeautifulSoup, declared_anchors: Set[str],
                          class_name: str = '') -> str:
    """
    Build a page-level summary of a Doxygen class: description + member listing.

    Doxygen renders each class's API as one or more `table.memberdecls`, grouped by
    a heading ("Public Member Functions", "Static Public Member Functions", ...),
    with two rows per symbol:

        <tr class="memitem:a1b2c3"> ... signature ... </tr>
        <tr class="memdesc:a1b2c3"> ... brief description ... </tr>

    This is the only page-level view of a class's API. Without it the corpus holds
    a class as unrelated per-symbol chunks, and aggregate questions ("what are the
    public methods of X") have nothing to match.

    INLINE_INHERITED_MEMB merges inherited members into these tables with NO marker
    on the row, so rows are kept only when their anchor appears in declared_anchors
    — pass the anchors from extract_memitems_with_soup(), which already skips
    inherited copies. A heading whose rows all drop out is omitted too.

    When `class_name` is given, each member line is prefixed with `Class::member`.
    Doxygen's table lists BARE member names, so without this every class summary
    reads alike and they compete with each other for "methods of <class>" queries —
    the class name would appear once per chunk while the query term needs to
    outweigh a thousand near-identical pages. The per-symbol detail chunks carry
    the qualified name inline already, which is why symbol-level questions resolve
    and page-level ones do not.

    Returns '' when the page has no memberdecls tables or nothing survives filtering.
    """
    tables = soup.find_all('table', class_='memberdecls')
    if not tables:
        return ''

    parts: List[str] = []
    textblock = soup.find('div', class_='textblock')
    if textblock is not None:
        description = re.sub(r'\s+', ' ', textblock.get_text(' ', strip=True)).strip()
        if description:
            parts.append(description)

    for table in tables:
        heading_tag = table.find('h2')
        rows: Dict[str, Dict[str, str]] = {}
        order: List[str] = []
        for row in table.find_all('tr'):
            classes = ' '.join(row.get('class') or [])
            match = re.search(r'mem(item|desc):(\w+)', classes)
            if not match:
                continue
            kind, anchor = match.group(1), match.group(2)
            if anchor not in declared_anchors:
                continue
            if anchor not in rows:
                rows[anchor] = {}
                order.append(anchor)
            rows[anchor][kind] = re.sub(r'\s+', ' ', row.get_text(' ', strip=True)).strip()
            if kind == 'item' and class_name:
                rows[anchor]['member'] = _member_name_for_anchor(row, anchor)

        if not order:
            continue

        lines = []
        if heading_tag is not None:
            lines.append(heading_tag.get_text(strip=True))
        for anchor in order:
            signature = rows[anchor].get('item', '')
            brief = rows[anchor].get('desc', '')
            line = f"{signature}  {brief}".strip() if brief else signature
            member = rows[anchor].get('member', '')
            if class_name and member:
                line = f"{class_name}::{member}  {line}".strip()
            if line:
                lines.append(line)
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def get_page_title(filepath: str) -> str:
    """
    Extract page title from an HTML file.
    Priority: h1 > <title> > filename stem.
    Raises FileNotFoundError if file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"HTML file not found: {filepath}")

    with open(path, encoding='utf-8', errors='replace') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Try h1 first
    h1 = soup.find('h1')
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    # Try <title>
    title_tag = soup.find('title')
    if title_tag and title_tag.get_text(strip=True):
        return title_tag.get_text(strip=True)

    # Fallback: filename without extension
    return path.stem


def parse_page(filepath: str, doc_category: str, max_chars: int = 8000) -> str:
    """
    Read and parse an HTML page into a plain-text string suitable for LLM context.

    - Strips navigation, headers/footers, script/style tags
    - Extracts meaningful text content
    - Truncates to max_chars with a truncation marker

    Raises FileNotFoundError if file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"HTML file not found: {filepath}")

    with open(path, encoding='utf-8', errors='replace') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Remove noise elements
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()
    # Remove Doxygen navigation divs
    for tag in soup.find_all(id=['nav-tree', 'nav-path', 'top', 'side-nav']):
        tag.decompose()

    text = soup.get_text(separator='\n', strip=True)
    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"

    return text


def save_page_index(entries: List[Dict[str, Any]], output_path: str) -> None:
    """
    Write page index entries to a JSON file.
    Creates parent directories if needed.

    Each entry has: filepath, filename, title, module, doc_category.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _tokenize(text: str) -> Set[str]:
    return {t.lower() for t in re.split(r'[\s_\-]+', text) if len(t) >= 3}


def _chunk_fusion_key(doc: Any) -> tuple:
    """Identity a chunk shares across vector and BM25 result sets. Mirrors
    chatbot/core/bm25_index._fusion_key exactly — reimplemented locally so
    extraction/ stays independent of chatbot/core (it keeps working
    standalone, e.g. from process_docs.py)."""
    m = doc.metadata
    return (m.get('url', ''), m.get('section_id', ''), m.get('chunk_position', ''))


def _rrf_fuse_chunks(ranked_lists: List[List[Any]], rrf_k: int = 60) -> List[Any]:
    """
    Reciprocal rank fusion over chunk-level Document lists — same algorithm
    and dedup-by-fusion-key as chatbot/core/bm25_index.reciprocal_rank_fusion,
    reimplemented locally for the same independence reason as
    _chunk_fusion_key above.

    Chunk-level (not filename-level) so a page's score reflects every one of
    its chunks that scored well, matching how RAG mode's own hybrid_retrieve
    fuses — and so a later dedup pass sees every candidate chunk per page,
    not just whichever one a channel happened to rank first.
    """
    scores: Dict[tuple, float] = {}
    doc_by_key: Dict[tuple, Any] = {}
    order: List[tuple] = []
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = _chunk_fusion_key(doc)
            if key not in doc_by_key:
                order.append(key)
                doc_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
    return [doc_by_key[key] for key in sorted(order, key=lambda k: scores[k], reverse=True)]


def _declaring_page_names(hierarchy: str) -> set:
    """
    Doxygen page names that would hold the declaration of a qualified symbol.

    'std::shared_ptr< ModelAPI_Result > ModelAPI_Feature::lastResult' yields
    {'classModelAPI__Feature.html', 'structModelAPI__Feature.html',
     'interfaceModelAPI__Feature.html'}. Mirrors
    chatbot/core/rag_chatbot.py's DocumentationChatbot._declaring_page_names
    exactly — reimplemented locally for the same independence reason as
    _chunk_fusion_key above.

    Returns an empty set when no qualified name can be read out of hierarchy.
    """
    match = re.search(r'([A-Za-z_]\w*)::~?[A-Za-z_]\w*\s*$', hierarchy or '')
    if not match:
        return set()
    escaped = match.group(1).replace('_', '__')
    return {f"{kind}{escaped}.html" for kind in ('class', 'struct', 'interface')}


def _dedup_symbol_copies(docs: List[Any]) -> List[Any]:
    """
    Collapse duplicate copies of the same documented symbol (Doxygen's
    INLINE_INHERITED_MEMB copies one inherited member's docs onto every
    subclass page — left alone, a single inherited method can fill a whole
    result set and crowd out the page the question was actually about).

    Mirrors chatbot/core/rag_chatbot.py's DocumentationChatbot._dedup_symbol_copies
    exactly — same anchor_id key, same declaring-class-wins tie-break, same
    content-hash fallback for chunks with no anchor — reimplemented locally
    for the same independence reason as _chunk_fusion_key above. Must run
    before any collapse to one candidate per page: a page's chunk that
    happens to rank first in a channel is not necessarily the chunk carrying
    the shared anchor_id, so deduping after that collapse would miss most
    duplicates. Input order is preserved.
    """
    best: Dict[str, int] = {}
    result: List[Any] = []

    for doc in docs:
        anchor = doc.metadata.get('anchor_id', '')
        key = f"a:{anchor}" if anchor else f"c:{hashlib.sha1(doc.page_content.encode('utf-8')).hexdigest()}"

        page = doc.metadata.get('url', '').split('#')[0].rsplit('/', 1)[-1]
        is_declaring = page in _declaring_page_names(doc.metadata.get('hierarchy', ''))

        if key not in best:
            best[key] = len(result)
            result.append(doc)
        elif is_declaring:
            result[best[key]] = doc

    return result


def search_pages(
    vectorstore: Chroma,
    page_index: List[Dict[str, Any]],
    query: str,
    k: int = 50,
    exclude_filepaths: Optional[Set[str]] = None,
    bm25_index: Optional[Any] = None,
    title_boost_enabled: bool = False,
    rerank_fn: Optional[Callable[[str, List[Any]], List[float]]] = None,
    select_sections_fn: Optional[Callable[[str, List[Any], int, int], List[Any]]] = None,
    top_n: int = 5,
    char_budget: int = 15000,
) -> List[Dict[str, Any]]:
    """
    Hybrid search over the RAG-mode retrieval channels, mapped back to
    page_index entries so read_page(filepath=...) keeps working unchanged.

    Fuses up to three ranked CHUNK-level channels via reciprocal rank fusion:
    dense (always), BM25 body search and BM25 title search (both only when
    bm25_index is given — a duck-typed object exposing .search(query, k) and
    .search_titles(query, k), i.e. chatbot/core/bm25_index.BM25Index, passed
    in rather than imported so this module doesn't depend on chatbot/core).
    title_boost_enabled gates the title channel same as rag_chatbot.py's
    _hybrid_retrieve — see [[project-rag-entity-lookup-lesson]]/task #93:
    dense-only retrieval misses named-entity lookups, which the agentic
    page-search tool is just as exposed to as the main RAG chain.

    Mirrors RAG mode's own _hybrid_retrieve + _dedup_symbol_copies pipeline:
    fuse at chunk level, THEN dedup duplicate symbol copies (Doxygen's
    INLINE_INHERITED_MEMB defect — see _dedup_symbol_copies), THEN collapse to
    one candidate per page. Doing it in this order — not collapsing to pages
    first — means dedup sees every chunk of every page, not just whichever
    chunk a channel happened to rank first; collapsing first can miss a
    duplicate entirely if that page's first-ranked chunk isn't the one
    carrying the shared anchor_id.

    rerank_fn, if given, is a duck-typed callable(query, docs) -> scores —
    i.e. chatbot/core/rag_chatbot.py's DocumentationChatbot.score_against_query,
    injected the same way bm25_index is — that reorders the final ≤k page
    candidates by score instead of fusion rank, using each page's chosen
    representative chunk (the one that survived dedup and collapse below).
    Only used by the page-level branch.

    select_sections_fn, if given, switches to section-level retrieval
    (agentic mode): a duck-typed callable(query, chunks, top_n, char_budget)
    -> List[Document] — i.e. DocumentationChatbot.expand_sections — that
    reranks the fused + symbol-deduped chunk pool, keeps one representative
    per section, caps at top_n and rebuilds each survivor's full section
    text. Each returned Document is re-joined to its page_index entry; the
    result dicts carry the entry plus 'section_id' and 'section_text'. When
    None, the page-level branch runs unchanged.

    Chroma chunk metadata may hold a web URL instead of a local filepath
    (extraction config-dependent), so the join to page_index is done via
    filename — the raw, unmodified trailing segment of the URL — rather
    than the url/filepath fields directly.
    """
    exclude_filepaths = exclude_filepaths or set()

    # Join key = trailing filename, PLUS a <parent-dir>/<filename> qualified key.
    # Filenames alone are NOT unique across the corpus (e.g. ConstructionPlugin/
    # pointFeature.html and SketchPlugin/pointFeature.html both end "pointFeature
    # .html"); keying only on the bare name silently maps a chunk hit to whichever
    # colliding page the dict kept, handing the agent the wrong page. Chunk URLs
    # carry the parent dir, so prefer the qualified key and fall back to the bare
    # name only when the qualified one doesn't match.
    def _qualified(path: str) -> str:
        parts = [p for p in path.split('#')[0].split('?')[0].split('/') if p]
        return '/'.join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else '')

    by_key: Dict[str, Any] = {}
    for e in page_index:
        by_key.setdefault(e['filename'], e)
        by_key[_qualified(e['filepath'])] = e

    fetch_k = k * 4  # over-fetch; dedup + collapse-to-page + join will shrink it

    channels = [vectorstore.similarity_search(query, k=fetch_k)]
    if bm25_index is not None:
        channels.append(bm25_index.search(query, fetch_k))
        if title_boost_enabled:
            channels.append(bm25_index.search_titles(query, fetch_k))

    fused_chunks = channels[0] if len(channels) == 1 else _rrf_fuse_chunks(channels)
    deduped_chunks = _dedup_symbol_copies(fused_chunks)

    if select_sections_fn is not None:
        # Section-level branch (agentic mode): hand the fused + symbol-deduped
        # chunk pool to the injected DocumentationChatbot.expand_sections, which
        # reranks, keeps one representative per section_id, caps at top_n and
        # rebuilds each survivor's full section text. Re-join each returned
        # section to its page_index entry via the same qualified-filename
        # lookup the page branch uses.
        section_docs = select_sections_fn(query, deduped_chunks, top_n, char_budget)
        sections: List[Dict[str, Any]] = []
        seen_section_paths: Set[str] = set()
        for doc in section_docs:
            url = doc.metadata.get('url', '')
            if not url:
                continue
            qualified = _qualified(url)
            bare = qualified.split('/')[-1]
            key = qualified if qualified in by_key else (bare if bare in by_key else None)
            if key is None:
                continue
            entry = by_key[key]
            filepath = entry['filepath']
            if filepath in seen_section_paths or filepath in exclude_filepaths:
                continue
            seen_section_paths.add(filepath)
            sections.append({
                **entry,
                'section_id': doc.metadata.get('section_id', ''),
                'section_text': doc.page_content,
            })
        return sections

    seen: Set[str] = set()
    result_keys: List[str] = []
    key_to_doc: Dict[str, Any] = {}
    for doc in deduped_chunks:
        url = doc.metadata.get('url', '')
        if not url:
            continue
        qualified = _qualified(url)
        bare = qualified.split('/')[-1]
        key = qualified if qualified in by_key else (bare if bare in by_key else None)
        if key is None:
            continue  # chunk has no matching page_index entry — skip rather than guess

        filepath = by_key[key]['filepath']
        if filepath in seen or filepath in exclude_filepaths:
            continue
        seen.add(filepath)
        result_keys.append(key)
        key_to_doc[key] = doc

        if len(result_keys) >= k:
            break

    if rerank_fn is not None and result_keys:
        scored_docs = [key_to_doc[key] for key in result_keys]
        scores = rerank_fn(query, scored_docs)
        result_keys = [key for _score, key in
                       sorted(zip(scores, result_keys), key=lambda x: float(x[0]), reverse=True)]

    return [by_key[key] for key in result_keys]

"""
Shared HTML parsing utilities used by:
  - extraction/process_docs.py  (at extraction time)
  - chatbot/core/agentic_chatbot.py  (at query time)
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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


def search_pages(
    page_index: List[Dict[str, Any]],
    query: str,
    exclude_filepaths: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search the page index for entries matching the query.

    Two-stage token matching:
    1. Filename tokens: for each query token, find pages whose filename contains it (case-insensitive)
    2. Title tokens: for each remaining query token, find pages whose title contains it

    Deduplicates by filepath. Preserves insertion order (filename matches first).
    exclude_filepaths: set of filepath strings to skip.
    """
    if exclude_filepaths is None:
        exclude_filepaths = set()

    tokens = [t.lower() for t in re.split(r'[\s_\-]+', query) if len(t) >= 3]
    if not tokens:
        return []

    seen: Set[str] = set(exclude_filepaths)
    result = []

    # Stage 1: filename matches
    for token in tokens:
        for entry in page_index:
            fp = entry['filepath']
            if fp in seen:
                continue
            if token in entry['filename'].lower():
                seen.add(fp)
                result.append(entry)

    # Stage 2: title matches
    for token in tokens:
        for entry in page_index:
            fp = entry['filepath']
            if fp in seen:
                continue
            if token in entry['title'].lower():
                seen.add(fp)
                result.append(entry)

    return result

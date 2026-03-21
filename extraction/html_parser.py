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
        fallback = element.get_text(separator='\n', strip=True)
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

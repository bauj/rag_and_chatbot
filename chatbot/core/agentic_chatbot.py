"""
Agentic documentation chatbot — answers questions by directly reading HTML pages.

Uses a fixed 2-round pipeline:
  Round 1: search page index → LLM picks pages → read+parse → LLM answers
  Round 2 (if NEED_MORE_INFO): refined search → read more pages → LLM answers again

Returns the same {answer, sources, filters, error} dict as DocumentationChatbot.ask().
"""

import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .config import ChatbotConfig

# html_parser lives in extraction/ which conftest adds to sys.path in tests.
# At runtime (chatbot/chatbot.py), extraction/ may not be on sys.path, so we
# add it dynamically here.
_EXTRACTION_DIR = Path(__file__).parent.parent.parent / "extraction"
if str(_EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(_EXTRACTION_DIR))

from html_parser import parse_page, search_pages  # noqa: E402


class AgenticChatbot:
    """
    Answers questions by searching a page index and reading HTML files on-the-fly.

    Alternative to DocumentationChatbot — no ChromaDB required at query time.
    Requires config.agentic to be set.
    """

    def __init__(self, config: ChatbotConfig):
        if config.agentic is None:
            raise ValueError(
                "AgenticChatbot requires config.agentic to be set. "
                "Add an 'agentic' block to your config.json."
            )
        self.config = config
        self._agentic_cfg = config.agentic

        # Load and validate page index
        index_path = Path(self._agentic_cfg.page_index_path)
        if not index_path.is_absolute():
            index_path = Path(__file__).parent.parent / self._agentic_cfg.page_index_path
        index_path = index_path.resolve()

        if not index_path.exists():
            raise FileNotFoundError(f"Page index not found: {index_path}")

        try:
            with open(index_path, encoding='utf-8') as f:
                self._page_index: List[dict] = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Page index is not valid JSON: {e}")

        self.llm = self._init_llm()

    def _init_llm(self) -> ChatOpenAI:
        llm_cfg = self.config.llm
        if llm_cfg.ssl_cert_file:
            import os
            cert_path = Path(llm_cfg.ssl_cert_file).expanduser().resolve()
            if not cert_path.exists():
                raise FileNotFoundError(f"SSL certificate file not found: {cert_path}")
            os.environ['SSL_CERT_FILE'] = str(cert_path)
            os.environ['REQUESTS_CA_BUNDLE'] = str(cert_path)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Parameters \{'max_tokens'\} should be specified explicitly",
                category=UserWarning,
            )
            return ChatOpenAI(
                model=llm_cfg.model,
                base_url=llm_cfg.base_url,
                api_key=llm_cfg.api_key,
                temperature=self.config.temperature,
                max_completion_tokens=None,
                streaming=False,
                model_kwargs={"max_tokens": self.config.max_tokens},
            )

    def _select_pages(self, candidates: List[dict], question: str, max_pages: int) -> List[dict]:
        """
        Ask the LLM to pick the most relevant pages from candidates.
        Returns selected entries, falling back to first max_pages if LLM output is malformed.
        """
        if not candidates:
            return []

        candidate_list = "\n".join(
            f"- {e['filename']} ({e['title']}, {e['module']}/{e['doc_category']})"
            for e in candidates
        )
        prompt = (
            f"Given the question: {question!r}\n\n"
            f"Select the most relevant documentation pages (up to {max_pages}) from this list:\n"
            f"{candidate_list}\n\n"
            f"Respond with a JSON array of filenames only, e.g. "
            f'["classModelAPI__Feature.html"]. '
            f"Return only the JSON array, no explanation."
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
            # Extract JSON array (LLM may wrap it in markdown code fences)
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON array found")
            selected_names = json.loads(match.group())
            if not isinstance(selected_names, list):
                raise ValueError("Not a list")
        except Exception:
            # Fallback: use first max_pages candidates
            return candidates[:max_pages]

        # Map names back to index entries (validate they exist in candidates)
        name_to_entry = {e['filename']: e for e in candidates}
        result = []
        for name in selected_names:
            if name in name_to_entry and len(result) < max_pages:
                result.append(name_to_entry[name])
        if not result:
            return candidates[:max_pages]
        return result

    def _read_pages(self, entries: List[dict], max_chars: Optional[int] = None) -> List[dict]:
        """
        Read and parse each page. Returns entries enriched with 'content' key.
        Skips pages that can't be read (FileNotFoundError) with a warning.
        """
        limit = max_chars if max_chars is not None else self._agentic_cfg.max_chars_per_page
        result = []
        for entry in entries:
            try:
                content = parse_page(
                    entry['filepath'],
                    entry['doc_category'],
                    max_chars=limit,
                )
                result.append({**entry, 'content': content})
            except FileNotFoundError:
                print(f"Warning: page not found, skipping: {entry['filepath']}", file=sys.stderr)
        return result

    def _build_context(self, read_entries: List[dict]) -> str:
        """Format read pages into a context string for the LLM."""
        parts = []
        for e in read_entries:
            header = f"[{e['module']}/{e['doc_category']}] {e['title']}"
            parts.append(f"--- {header} ---\n{e.get('content', '')}")
        return "\n\n".join(parts)

    def _answer_from_context(self, question: str, context: str, missing_info: Optional[str] = None) -> str:
        """
        Ask the LLM to answer the question from the given context.
        If missing_info is provided (Round 2), include it to focus the answer.
        Instructs the LLM to append NEED_MORE_INFO marker if pages are insufficient.
        """
        system = (
            f"You are an expert assistant for {self.config.project_name} documentation. "
            "Answer the user's question strictly based on the documentation provided. "
            "If the documentation is insufficient to answer the question, append on its own "
            "line at the very end of your response: NEED_MORE_INFO: <brief description of what is missing>"
        )
        user_parts = [f"Documentation:\n{context}\n\nQuestion: {question}"]
        if missing_info:
            user_parts.append(f"\n\nNote: a previous search identified this gap: {missing_info}")
        user_content = "".join(user_parts)

        response = self.llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user_content),
        ])
        return response.content

    @staticmethod
    def _extract_need_more_info(response: str) -> Optional[str]:
        """
        Check if the response ends with NEED_MORE_INFO marker.
        Returns the description after the colon, or None if not present.
        """
        last_line = response.rstrip().split("\n")[-1]
        if last_line.startswith("NEED_MORE_INFO:"):
            return last_line[len("NEED_MORE_INFO:"):].strip()
        return None

    @staticmethod
    def _strip_marker(response: str) -> str:
        """Remove the NEED_MORE_INFO line from the end of a response."""
        lines = response.rstrip().split("\n")
        if lines and lines[-1].startswith("NEED_MORE_INFO:"):
            lines = lines[:-1]
        return "\n".join(lines).rstrip()

    def ask(self, question: str,
            max_chars_per_page: Optional[int] = None,
            max_pages_per_round: Optional[int] = None,
            max_pages_round2: Optional[int] = None,
            max_tokens: Optional[int] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Answer a question by searching the page index and reading HTML files.

        Args:
            max_chars_per_page: Override config value at runtime.
            max_pages_per_round: Override config value at runtime.
            max_pages_round2: Override config value at runtime.
            max_tokens: Override LLM max_tokens at runtime (not yet wired; reserved).
            **kwargs: Accepted for interface compatibility; ignored.

        Returns:
            {answer, sources, filters, error}
        """
        cfg = self._agentic_cfg
        _max_chars = max_chars_per_page if max_chars_per_page is not None else cfg.max_chars_per_page
        _max_pages_r1 = max_pages_per_round if max_pages_per_round is not None else cfg.max_pages_per_round
        _max_pages_r2 = max_pages_round2 if max_pages_round2 is not None else cfg.max_pages_round2

        def _error_response(err):
            return {
                "answer": None,
                "sources": [],
                "filters": {"mode": "agentic", "rounds_used": 0},
                "error": err,
            }

        # --- Round 1 ---
        candidates = search_pages(self._page_index, question)
        if not candidates:
            # Fall back to all pages when token matching yields no results
            candidates = [e for e in self._page_index]
        if not candidates:
            return _error_response("No relevant pages found for query.")

        selected = self._select_pages(candidates, question, _max_pages_r1)
        read_entries = self._read_pages(selected, max_chars=_max_chars)
        context = self._build_context(read_entries)
        read_filepaths = {e['filepath'] for e in read_entries}

        try:
            response = self._answer_from_context(question, context)
        except Exception as e:
            return _error_response(str(e))

        missing_info = self._extract_need_more_info(response)

        # --- Round 2 (if needed) ---
        rounds_used = 1
        if missing_info:
            r2_candidates = search_pages(
                self._page_index, missing_info, exclude_filepaths=read_filepaths
            )
            if r2_candidates:
                r2_selected = self._select_pages(r2_candidates, missing_info, _max_pages_r2)
                r2_read = self._read_pages(r2_selected, max_chars=_max_chars)
                read_entries.extend(r2_read)
                context2 = self._build_context(read_entries)
                try:
                    response = self._answer_from_context(question, context2, missing_info=missing_info)
                except Exception as e:
                    return _error_response(str(e))
                rounds_used = 2

        answer = self._strip_marker(response)

        sources = [
            {
                "filepath": e["filepath"],
                "filename": e["filename"],
                "title": e["title"],
                "module": e["module"],
                "doc_category": e["doc_category"],
                "content": e.get("content", ""),
            }
            for e in read_entries
        ]

        return {
            "answer": answer,
            "sources": sources,
            "filters": {"mode": "agentic", "rounds_used": rounds_used},
            "error": None,
        }

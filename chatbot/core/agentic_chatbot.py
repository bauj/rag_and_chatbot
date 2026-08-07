"""
Agentic documentation chatbot — smolagents CodeAgent.

A CodeAgent decides for itself how many search/read cycles to run over a
page index, bounded by config.agentic.max_steps. Requires config.agentic
and the smolagents package.

Returns the same {answer, sources, filters, error} dict as DocumentationChatbot.ask().
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ChatbotConfig

# Matches code the model wrote instead of prose when forced to answer at max_steps
# (smolagents' default code_block_tags is literally "<code>"/"</code>") — see
# _handle_max_steps_reached in smolagents: the forced-synthesis prompt asks for a
# plain-language answer, but the model has just spent the whole transcript writing
# code and often keeps doing so.
_LEAKED_CODE_PATTERN = re.compile(
    r"<code>|```|Calling tools:|search_pages\(|read_page\(", re.IGNORECASE
)

# html_parser lives in extraction/ which conftest adds to sys.path in tests.
# At runtime (chatbot/chatbot.py), extraction/ may not be on sys.path, so we
# add it dynamically here.
_EXTRACTION_DIR = Path(__file__).parent.parent.parent / "extraction"
if str(_EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(_EXTRACTION_DIR))

from html_parser import parse_page, search_pages  # noqa: E402


class AgenticChatbot:
    """
    Answers questions using a smolagents CodeAgent with search_pages/read_page tools.

    Requires config.agentic to be set and the smolagents package to be installed.
    """

    def __init__(self, config: ChatbotConfig):
        if config.agentic is None:
            raise ValueError(
                "AgenticChatbot requires config.agentic to be set. "
                "Add an 'agentic' block to your config.json."
            )

        try:
            from smolagents import Tool, CodeAgent, OpenAIServerModel
        except ImportError as e:
            raise ImportError(
                "smolagents is required for agentic mode.\n"
                "Install it: pip install smolagents"
            ) from e

        self.config = config
        self._agentic_cfg = config.agentic
        self._CodeAgent = CodeAgent
        self._OpenAIServerModel = OpenAIServerModel
        self._SearchPagesToolCls, self._ReadPageToolCls = _build_tool_classes(Tool)
        self._prompt_templates = _build_prompt_templates()

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

        self._entry_by_filepath = {e['filepath']: e for e in self._page_index}

    def _build_model_kwargs(self, temperature: Optional[float], max_tokens: Optional[int]) -> dict:
        """
        Assemble the extra kwargs OpenAIServerModel forwards to the underlying
        completion call, plus SSL cert wiring for internal/self-signed endpoints
        (same pattern the old fixed-script AgenticChatbot used via httpx.Client).
        """
        llm_cfg = self.config.llm
        kwargs: Dict[str, Any] = {
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        if llm_cfg.ssl_cert_file:
            import httpx
            cert_path = Path(llm_cfg.ssl_cert_file).expanduser().resolve()
            if not cert_path.exists():
                raise FileNotFoundError(f"SSL certificate file not found: {cert_path}")
            kwargs["client_kwargs"] = {"http_client": httpx.Client(verify=str(cert_path))}
        return kwargs

    def ask(self, question: str,
            max_steps: Optional[int] = None,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Answer a question by letting a CodeAgent search the page index and read pages.

        Args:
            max_steps: Override config value at runtime.
            max_tokens: Override config.max_tokens at runtime.
            temperature: Override config.temperature at runtime.
            **kwargs: Accepted for interface compatibility; ignored.

        Returns:
            {answer, sources, filters, error}
        """
        cfg = self._agentic_cfg
        _max_steps = max_steps if max_steps is not None else cfg.max_steps

        read_entries: List[dict] = []
        search_tool = self._SearchPagesToolCls(self._page_index)
        read_tool = self._ReadPageToolCls(self._entry_by_filepath, cfg.max_chars_per_page, read_entries)

        model = self._OpenAIServerModel(
            model_id=self.config.llm.model,
            api_base=self.config.llm.base_url,
            api_key=self.config.llm.api_key,
            **self._build_model_kwargs(temperature, max_tokens),
        )

        agent = self._CodeAgent(
            tools=[search_tool, read_tool],
            model=model,
            max_steps=_max_steps,
            additional_authorized_imports=[],
            prompt_templates=self._prompt_templates,
        )

        task = (
            f"You are an expert assistant for {self.config.project_name} documentation. "
            "Use the search_pages tool to find candidate documentation pages, then the "
            "read_page tool to read their content. You MUST call read_page at least once "
            "before answering — never answer from prior knowledge alone. Answer the "
            "question strictly based on what you read.\n\n"
            f"Question: {question}"
        )

        try:
            raw_answer = agent.run(task)
        except Exception as e:
            return {
                "answer": None,
                "sources": [],
                "filters": {"mode": "agentic", "steps_used": 0, "grounded": False},
                "error": str(e),
            }

        answer = str(raw_answer)

        seen = set()
        sources = []
        for entry in read_entries:
            if entry["filepath"] in seen:
                continue
            seen.add(entry["filepath"])
            sources.append({
                "filepath": entry["filepath"],
                "filename": entry["filename"],
                "title": entry["title"],
                "module": entry["module"],
                "doc_category": entry["doc_category"],
                "content": entry.get("content", ""),
            })

        steps_used = len(agent.memory.steps) if hasattr(agent, "memory") else None

        degraded = bool(_LEAKED_CODE_PATTERN.search(answer))
        if degraded:
            answer = _fallback_answer(sources)

        return {
            "answer": answer,
            "sources": sources,
            "filters": {
                "mode": "agentic",
                "steps_used": steps_used,
                "grounded": bool(read_entries),
                "degraded_answer": degraded,
            },
            "error": None,
        }


def _fallback_answer(sources: List[dict]) -> str:
    """Honest stand-in when the model returned leaked code instead of prose."""
    if not sources:
        return (
            "I ran out of steps before reading any relevant documentation page "
            "and could not produce an answer. Try rephrasing the question."
        )
    titles = ", ".join(dict.fromkeys(s["title"] for s in sources))
    return (
        "I read the following page(s) but ran out of steps before synthesizing a "
        f"complete answer: {titles}. Try rephrasing the question or increasing "
        "agentic.max_steps."
    )


def _build_prompt_templates() -> dict:
    """
    smolagents' default forced-final-answer prompt ("provide an answer instead")
    is weak against a transcript that is 100% code — the model often keeps writing
    <code> blocks / tool calls instead of switching to prose. Strengthen just the
    final_answer section; everything else stays the CodeAgent default.
    """
    import importlib.resources
    import yaml

    templates = yaml.safe_load(
        importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
    )
    templates["final_answer"]["pre_messages"] = (
        "An agent tried to answer a user query but it got stuck and failed to do so. "
        "You are tasked with providing an answer instead, based on its memory below. "
        "Respond with plain natural-language prose ONLY — do not write Python code, "
        "do not use <code> tags, do not call any tool. If the memory below does not "
        "contain enough information to answer, say so plainly instead of guessing."
    )
    templates["final_answer"]["post_messages"] = (
        "Based on the above, provide a plain-language answer (no code, no tool calls) "
        "to the following user task:\n{{task}}"
    )
    return templates


def _build_tool_classes(Tool):
    """
    Build the SearchPagesTool/ReadPageTool classes, deferred until `Tool` is
    known to be importable — keeps this module importable even when smolagents
    isn't installed (the ImportError guard lives in __init__, not at module load).
    """

    class SearchPagesTool(Tool):
        name = "search_pages"
        description = (
            "Search the documentation page index for pages matching a query. "
            "Returns candidate pages with their filepath, title, module, and doc_category."
        )
        inputs = {"query": {"type": "string", "description": "Search terms describing what to look for."}}
        output_type = "string"

        def __init__(self, page_index: List[dict]):
            super().__init__()
            self._page_index = page_index

        def forward(self, query: str) -> str:
            candidates = search_pages(self._page_index, query)
            if not candidates:
                return "No matching pages found."
            return "\n".join(
                f"- {e['filepath']} | {e['title']} | {e['module']}/{e['doc_category']}"
                for e in candidates
            )

    class ReadPageTool(Tool):
        name = "read_page"
        description = (
            "Read and return the text content of a documentation page. The filepath "
            "must be one returned by search_pages — other filepaths are rejected."
        )
        inputs = {
            "filepath": {"type": "string", "description": "Exact filepath as returned by search_pages."},
            "doc_category": {"type": "string", "description": "The page's doc_category (dev or user), as returned by search_pages."},
        }
        output_type = "string"

        def __init__(self, entry_by_filepath: Dict[str, dict], max_chars: int, read_entries: List[dict]):
            super().__init__()
            self._entry_by_filepath = entry_by_filepath
            self._max_chars = max_chars
            self._read_entries = read_entries

        def forward(self, filepath: str, doc_category: str) -> str:
            entry = self._entry_by_filepath.get(filepath)
            if entry is None:
                return (
                    f"Error: {filepath!r} is not a known documentation page. "
                    "Only use filepaths returned by search_pages."
                )
            try:
                content = parse_page(filepath, doc_category, max_chars=self._max_chars)
            except FileNotFoundError:
                return f"Error: page file not found on disk: {filepath}"
            self._read_entries.append({**entry, "content": content})
            return content

    return SearchPagesTool, ReadPageTool

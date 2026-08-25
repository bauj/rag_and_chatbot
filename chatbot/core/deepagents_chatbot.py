"""
Agentic documentation chatbot — LangChain "deepagents" harness.

THROWAWAY SPIKE (branch: spike/deepagents-shaper). Compares deepagents'
tool-calling loop against smolagents' code-execution loop (AgenticChatbot)
on the same search_pages/read_page tools, same page_index, same config.agentic
block. Not meant to be merged as-is — see notes/ for the eval writeup.

Returns the same {answer, sources, filters, error} dict as AgenticChatbot.ask()
so it drops into the same evaluator harness via --mode deepagents.
"""

import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from .bm25_index import BM25Index
from .config import ChatbotConfig

_EXTRACTION_DIR = Path(__file__).parent.parent.parent / "extraction"
if str(_EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(_EXTRACTION_DIR))

from html_parser import parse_page, search_pages  # noqa: E402


def _sum_token_usage(messages) -> Optional[Dict[str, int]]:
    """Sum usage_metadata across every AIMessage in the run's message log
    (LangChain's standard {input_tokens, output_tokens, total_tokens} shape,
    populated by ChatOpenAI per model call). None if never reported."""
    input_tokens = output_tokens = 0
    seen_any = False
    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if not usage:
            continue
        seen_any = True
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)
    if not seen_any:
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


_SYSTEM_PROMPT = (
    "You are an expert assistant for {project_name} documentation. "
    "Use the search_pages_tool to find candidate documentation pages, then "
    "read_page_tool to fetch one. read_page_tool does NOT return the page text — "
    "it returns a page_id. You MUST then call grep_page_tool with that EXACT "
    "page_id (not the filepath you searched with) and a specific search term (a "
    "method/class name, or a few keywords from the question) to find the relevant "
    "excerpt. Do not use the generic read_file or grep tools on these pages — "
    "grep_page_tool is the one built for this and returns proper surrounding "
    "context. Only call read_file on the page_id.txt path as a last resort if "
    "grep_page_tool finds nothing after trying a couple of different terms. "
    "You MUST read at least one page's content before answering — never answer "
    "from prior knowledge alone. Answer the question strictly based on what you "
    "read. Identify each distinct operation required and search for each one "
    "separately. If a search returns no results, you MUST retry with at least "
    "one alternative term before proceeding. Never invent API calls, imports, or "
    "libraries that did not appear in retrieved documentation — if you cannot "
    "find grounding, say so explicitly rather than guessing. Answer in plain "
    "prose; do not run shell commands."
)


class DeepAgentsChatbot:
    """
    Answers questions using a deepagents tool-calling agent with search_pages/
    read_page tools, mirroring AgenticChatbot's contract for apples-to-apples
    eval comparison against smolagents.
    """

    def __init__(self, config: ChatbotConfig, vectorstore: Chroma, bm25_index: Optional[BM25Index] = None):
        if config.agentic is None:
            raise ValueError(
                "DeepAgentsChatbot requires config.agentic to be set. "
                "Add an 'agentic' block to your config.json."
            )

        try:
            from deepagents import create_deep_agent
            from deepagents.backends import FilesystemBackend
            from langchain_core.tools import tool
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "deepagents is required for deepagents mode.\n"
                "Install it: pip install deepagents"
            ) from e

        self.config = config
        self._agentic_cfg = config.agentic
        self._create_deep_agent = create_deep_agent
        self._FilesystemBackend = FilesystemBackend
        self._tool_decorator = tool
        self._ChatOpenAI = ChatOpenAI
        self._vectorstore = vectorstore
        self._bm25_index = bm25_index

        index_path = Path(self._agentic_cfg.page_index_path)
        if not index_path.is_absolute():
            index_path = Path(__file__).parent.parent / self._agentic_cfg.page_index_path
        index_path = index_path.resolve()

        if not index_path.exists():
            raise FileNotFoundError(f"Page index not found: {index_path}")

        import json
        try:
            with open(index_path, encoding='utf-8') as f:
                self._page_index: List[dict] = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Page index is not valid JSON: {e}")

        self._entry_by_filepath = {e['filepath']: e for e in self._page_index}

    def _build_model(self, temperature: Optional[float], max_tokens: Optional[int]):
        llm_cfg = self.config.llm
        kwargs: Dict[str, Any] = {
            "model": llm_cfg.model,
            "base_url": llm_cfg.base_url,
            "api_key": llm_cfg.api_key,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        if llm_cfg.ssl_cert_file:
            import httpx
            cert_path = Path(llm_cfg.ssl_cert_file).expanduser().resolve()
            if not cert_path.exists():
                raise FileNotFoundError(f"SSL certificate file not found: {cert_path}")
            kwargs["http_client"] = httpx.Client(verify=str(cert_path))
        return self._ChatOpenAI(**kwargs)

    def _build_tools(self, max_chars_per_page: int, read_entries: List[dict], backend):
        title_boost_enabled = self.config.title_boost_enabled
        vectorstore = self._vectorstore
        page_index = self._page_index
        bm25_index = self._bm25_index
        entry_by_filepath = self._entry_by_filepath
        pages_by_id: Dict[str, str] = {}

        @self._tool_decorator
        def search_pages_tool(query: str) -> str:
            """Search the documentation page index for pages matching a query.
            Returns candidate pages with their filepath, title, module, and doc_category."""
            candidates = search_pages(
                vectorstore, page_index, query,
                bm25_index=bm25_index,
                title_boost_enabled=title_boost_enabled,
            )
            if not candidates:
                return "No matches for " + query + ". Try a broader or alternative term, or search a related feature category."
            return "\n".join(
                f"- {e['filepath']} | {e['title']} | {e['module']}/{e['doc_category']}"
                for e in candidates
            )

        @self._tool_decorator
        def read_page_tool(filepath: str, doc_category: str) -> str:
            """Fetch a documentation page. The filepath must be one returned by
            search_pages_tool — other filepaths are rejected. Does NOT return the
            page text — returns a page_id. Pass that EXACT page_id (not the
            filepath you searched with) to grep_page_tool with a specific search
            term to find the relevant excerpt, rather than reading the whole page."""
            entry = entry_by_filepath.get(filepath)
            if entry is None:
                return (
                    f"Error: {filepath!r} is not a known documentation page. "
                    "Only use filepaths returned by search_pages_tool."
                )
            try:
                # Full, untruncated text to search — max_chars_per_page only
                # bounds what ends up in sources/grading, not what's searchable.
                # parse_page compares len(text) > max_chars, so a plain None would
                # TypeError; use a sentinel large enough no real page hits it.
                content = parse_page(filepath, doc_category, max_chars=10**9)
            except FileNotFoundError:
                return f"Error: page file not found on disk: {filepath}"
            read_entries.append({**entry, "content": content[:max_chars_per_page]})

            page_id = re.sub(r"[^\w.-]", "_", f"{doc_category}_{entry['filename']}")
            pages_by_id[page_id] = content
            backend.write(page_id + ".txt", content)  # fallback: glob/read_file can still browse it
            return (
                f'page_id="{page_id}" ({len(content)} chars). Next call: '
                f'grep_page_tool(page_id="{page_id}", pattern=<specific term>).'
            )

        @self._tool_decorator
        def grep_page_tool(page_id: str, pattern: str, context_lines: int = 15) -> str:
            """Search a page fetched via read_page_tool for a literal substring
            (case-insensitive) and return matching lines with surrounding context
            — use this instead of reading the whole page. page_id must be exactly
            the value read_page_tool returned, not the original filepath."""
            content = pages_by_id.get(page_id)
            if content is None:
                return (
                    f"Error: no page with page_id={page_id!r}. Use the exact "
                    "page_id string read_page_tool returned, not the filepath."
                )
            lines = content.split("\n")
            pattern_lower = pattern.lower()
            hit_indices = [i for i, line in enumerate(lines) if pattern_lower in line.lower()]
            if not hit_indices:
                return f"No matches for {pattern!r} in {page_id}. Try a different or broader term."

            excerpts = []
            for i in hit_indices[:5]:  # cap so a very common term doesn't dump the whole page anyway
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                excerpts.append("\n".join(lines[start:end]))
            return "\n\n---\n\n".join(excerpts)

        return [search_pages_tool, read_page_tool, grep_page_tool]

    @staticmethod
    def _run_streaming(agent, question: str, recursion_limit: int) -> dict:
        """
        Live step trace to stderr — stream_mode="updates" yields one chunk per
        graph node (model call or tool call), so each search_pages/read_page
        call and each model turn prints as it happens, mirroring smolagents'
        default console step-by-step output.
        """
        # stream_mode="updates" yields only the messages each node *added*
        # (langgraph's add_messages reducer), so appending them in order
        # reconstructs the same message log agent.invoke() would return.
        all_messages: List[Any] = []
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": f"Question: {question}"}]},
            config={"recursion_limit": recursion_limit},
            stream_mode="updates",
        ):
            for node_name, node_update in chunk.items():
                msgs = node_update.get("messages", []) if isinstance(node_update, dict) else []
                all_messages.extend(msgs)
                for msg in msgs:
                    tool_calls = getattr(msg, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            print(f"[deepagents] {node_name}: call {tc['name']}({tc['args']})", file=sys.stderr)
                    elif getattr(msg, "name", None):
                        content = str(getattr(msg, "content", ""))
                        preview = content[:300] + ("..." if len(content) > 300 else "")
                        print(f"[deepagents] {node_name}: {msg.name} -> {preview}", file=sys.stderr)
                    elif getattr(msg, "content", None):
                        content = str(msg.content)
                        preview = content[:300] + ("..." if len(content) > 300 else "")
                        print(f"[deepagents] {node_name}: model says: {preview}", file=sys.stderr)

        return {"messages": all_messages}

    def ask(self, question: str,
            max_steps: Optional[int] = None,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            max_chars_per_page: Optional[int] = None,
            debug: Optional[bool] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Answer a question by letting a deepagents tool-calling agent search the
        page index and read pages. Returns {answer, sources, filters, error}.
        """
        cfg = self._agentic_cfg
        _max_steps = max_steps if max_steps is not None else cfg.max_steps
        _max_chars_per_page = max_chars_per_page if max_chars_per_page is not None else cfg.max_chars_per_page
        _debug = debug if debug is not None else getattr(cfg, "debug", False)

        read_entries: List[dict] = []
        model = self._build_model(temperature, max_tokens)

        # Fresh scratch dir per call: read_page writes pages here for grep to
        # search instead of dumping full text into the transcript. Per-call
        # (not instance-level) so concurrent ask() calls (evaluator's
        # ThreadPoolExecutor) don't collide, and torn down after — no state
        # carries between questions, same as the rest of this class.
        scratch_dir = tempfile.mkdtemp(prefix="deepagents_pages_")
        try:
            backend = self._FilesystemBackend(root_dir=scratch_dir, virtual_mode=True)
            tools = self._build_tools(_max_chars_per_page, read_entries, backend)

            agent = self._create_deep_agent(
                model=model,
                tools=tools,
                system_prompt=_SYSTEM_PROMPT.format(project_name=self.config.project_name),
                backend=backend,
            )

            # deepagents' recursion_limit counts model+tool nodes, roughly 2 per
            # tool-calling round-trip; approximate smolagents' max_steps budget.
            recursion_limit = 2 * _max_steps + 2

            try:
                if _debug:
                    result = self._run_streaming(agent, question, recursion_limit)
                else:
                    result = agent.invoke(
                        {"messages": [{"role": "user", "content": f"Question: {question}"}]},
                        config={"recursion_limit": recursion_limit},
                    )
                answer = result["messages"][-1].content
                steps_used = len(result["messages"])
            except Exception as e:
                return {
                    "answer": None,
                    "sources": [],
                    "filters": {"mode": "deepagents", "steps_used": 0, "grounded": False},
                    "error": str(e),
                }
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

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

        return {
            "answer": answer,
            "sources": sources,
            "filters": {
                "mode": "deepagents",
                "steps_used": steps_used,
                "grounded": bool(read_entries),
                "token_usage": _sum_token_usage(result["messages"]),
            },
            "error": None,
        }

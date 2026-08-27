"""
Agentic documentation chatbot — LangChain "deepagents" harness.

A deepagents tool-calling agent (LangGraph loop) decides for itself how many
search/read cycles to run over a page index, bounded by config.agentic.max_steps
(mapped to LangGraph's recursion_limit). Requires config.agentic and the
deepagents package.

Returns the same {answer, sources, filters, error} dict as DocumentationChatbot.ask().

Replaced the earlier smolagents CodeAgent engine on 2026-08-27: deepagents keeps
answer quality within noise while compacting context between turns (~2.7x fewer
tokens), and its subagent/context-management model is the direction the mode is
going. See docs/superpowers/specs/2026-08-27-replace-smolagents-with-deepagents-design.md.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langgraph.errors import GraphRecursionError

from .bm25_index import BM25Index
from .config import ChatbotConfig
from .debug_trace_writer import DebugTraceWriter
from .grounding_judge import check_grounding, llm_builtin_classifier

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


def _build_sources(read_entries: List[dict]) -> List[dict]:
    """Dedup the pages the agent read (by filepath), preserving read order and
    carrying the parsed page content through for the evaluator's groundedness
    grading (#114)."""
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
    return sources


def _fallback_answer(sources: List[dict], reason: str = "steps") -> str:
    """Honest stand-in when the model's answer couldn't be trusted as-is."""
    if not sources:
        return (
            "I ran out of steps before reading any relevant documentation page "
            "and could not produce an answer. Try rephrasing the question."
        )
    titles = ", ".join(dict.fromkeys(s["title"] for s in sources))

    if reason == "ungrounded":
        return (
            "I read the following page(s) but the code I generated didn't match "
            f"the documented API, so I'm not returning it: {titles}. Try rephrasing "
            "the question, or ask again — this can happen intermittently."
        )

    return (
        "I read the following page(s) but ran out of steps before synthesizing a "
        f"complete answer: {titles}. Try rephrasing the question or increasing "
        "agentic.max_steps."
    )


_SYSTEM_PROMPT = (
    "You are an expert assistant for {project_name} documentation. "
    "Use the search_pages_tool to find candidate documentation pages, then the "
    "read_page_tool to read their content. You MUST call read_page_tool at least "
    "once before answering — never answer from prior knowledge alone. Answer the "
    "question strictly based on what you read. Identify each distinct operation "
    "required and search for each one separately. If a search returns no results, "
    "you MUST retry with at least one alternative term before proceeding. Never "
    "invent API calls, imports, or libraries that did not appear in retrieved "
    "documentation — if you cannot find grounding, say so explicitly rather than "
    "guessing. Answer in plain prose; do not run shell commands."
)


class AgenticChatbot:
    """
    Answers questions using a deepagents tool-calling agent with search_pages/
    read_page tools.

    Requires config.agentic to be set and the deepagents package to be installed.
    """

    def __init__(self, config: ChatbotConfig, vectorstore: Chroma, bm25_index: Optional[BM25Index] = None,
                 reranker_score_fn: Optional[Any] = None):
        if config.agentic is None:
            raise ValueError(
                "AgenticChatbot requires config.agentic to be set. "
                "Add an 'agentic' block to your config.json."
            )

        try:
            from deepagents import create_deep_agent
            from langchain_core.tools import tool
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "deepagents is required for agentic mode.\n"
                "Install it: pip install deepagents"
            ) from e

        self.config = config
        self._agentic_cfg = config.agentic
        self._create_deep_agent = create_deep_agent
        self._tool_decorator = tool
        self._ChatOpenAI = ChatOpenAI
        self._vectorstore = vectorstore
        self._bm25_index = bm25_index
        self._reranker_score_fn = reranker_score_fn

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

        self._entry_by_filename = {e['filename']: e for e in self._page_index}
        self._debug_writer = DebugTraceWriter(
            debug_dir=getattr(self._agentic_cfg, "debug_dir", "debug_traces")
        )

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

    def _build_tools(self, max_chars_per_page: int, read_entries: List[dict]):
        title_boost_enabled = self.config.title_boost_enabled
        vectorstore = self._vectorstore
        page_index = self._page_index
        bm25_index = self._bm25_index
        entry_by_filename = self._entry_by_filename
        rerank_fn = self._reranker_score_fn

        @self._tool_decorator
        def search_pages_tool(query: str) -> str:
            """Search the documentation page index for pages matching a query.
            Returns candidate pages with their filename, title, module, and doc_category."""
            candidates = search_pages(
                vectorstore, page_index, query,
                bm25_index=bm25_index,
                title_boost_enabled=title_boost_enabled,
                rerank_fn=rerank_fn,
                k=15,
            )
            if not candidates:
                return "No matches for " + query + ". Try a broader or alternative term, or search a related feature category."
            return "\n".join(
                f"- {e['filename']} | {e['title']} | {e['module']}/{e['doc_category']}"
                for e in candidates
            )

        @self._tool_decorator
        def read_page_tool(filename: str, doc_category: str) -> str:
            """Read and return the text content of a documentation page. The
            filename must be one returned by search_pages_tool — other
            filenames are rejected."""
            entry = entry_by_filename.get(filename)
            if entry is None:
                return (
                    f"Error: {filename!r} is not a known documentation page. "
                    "Only use filenames returned by search_pages_tool."
                )
            try:
                content = parse_page(entry['filepath'], doc_category, max_chars=max_chars_per_page)
            except FileNotFoundError:
                return f"Error: page file not found on disk: {entry['filepath']}"
            read_entries.append({**entry, "content": content})
            return content

        return [search_pages_tool, read_page_tool]

    @staticmethod
    def _run_streaming(agent, input_messages: List[dict], recursion_limit: int) -> dict:
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
            {"messages": input_messages},
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
        MAX_GROUNDING_RETRIES = 1

        cfg = self._agentic_cfg
        _max_steps = max_steps if max_steps is not None else cfg.max_steps
        _max_chars_per_page = max_chars_per_page if max_chars_per_page is not None else cfg.max_chars_per_page
        _debug = debug if debug is not None else getattr(cfg, "debug", False)

        read_entries: List[dict] = []
        model = self._build_model(temperature, max_tokens)
        tools = self._build_tools(_max_chars_per_page, read_entries)

        agent = self._create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=_SYSTEM_PROMPT.format(project_name=self.config.project_name),
        )

        # deepagents' recursion_limit counts model+tool nodes, roughly 2 per
        # tool-calling round-trip; approximate smolagents' max_steps budget.
        recursion_limit = 2 * _max_steps + 2

        def _invoke(input_messages: List[dict]) -> dict:
            if _debug:
                return self._run_streaming(agent, input_messages, recursion_limit)
            return agent.invoke(
                {"messages": input_messages},
                config={"recursion_limit": recursion_limit},
            )

        def _observations_text() -> str:
            return "\n".join(e.get("content", "") for e in read_entries)

        try:
            result = _invoke([{"role": "user", "content": f"Question: {question}"}])
            answer = result["messages"][-1].content

            classify_builtin = llm_builtin_classifier(lambda messages: model.invoke(messages))
            ungrounded_calls = check_grounding(answer, _observations_text(), llm_classify_fn=classify_builtin)
            retries_used = 0
            while ungrounded_calls and retries_used < MAX_GROUNDING_RETRIES:
                retries_used += 1
                complaints = "; ".join(f"{c['call']} ({c['reason']})" for c in ungrounded_calls)
                retry_task = (
                    f"Your previous answer has issues with these calls: {complaints}. "
                    f"For any call not found in the documentation, search/read again to find "
                    f"the correct name or signature — do not invent a replacement. For any "
                    f"call flagged as having multiple documented signatures, re-read its "
                    f"documentation and confirm which variant you're using and what each "
                    f"argument actually means — do not assume from a bare code example "
                    f"alone. Fix ONLY these calls and provide the corrected answer."
                )
                retry_messages = result["messages"] + [{"role": "user", "content": retry_task}]
                try:
                    result = _invoke(retry_messages)
                except Exception:
                    break  # keep the last answer/ungrounded_calls, fall through to degraded handling

                answer = result["messages"][-1].content
                ungrounded_calls = check_grounding(answer, _observations_text(), llm_classify_fn=classify_builtin)

            steps_used = len(result["messages"])
        except GraphRecursionError:
            # LangGraph raises this (a hard crash, answer=None) when the step
            # budget is exhausted — smolagents forced a best-effort synthesis
            # here instead. read_entries is populated as a side effect by
            # read_page_tool's closure during the run, so pages read before the
            # budget ran out survive the crash; degrade to a fallback answer.
            sources = _build_sources(read_entries)
            return {
                "answer": _fallback_answer(sources, reason="steps"),
                "sources": sources,
                "filters": {
                    "mode": "agentic",
                    "steps_used": recursion_limit,
                    "grounded": bool(read_entries),
                    "degraded_answer": True,
                    "ungrounded_calls": [],
                    "grounding_retries_used": 0,
                    "token_usage": None,
                },
                "error": None,
            }
        except Exception as e:
            return {
                "answer": None,
                "sources": [],
                "filters": {"mode": "agentic", "steps_used": 0, "grounded": False},
                "error": str(e),
            }

        # An ambiguous_overload flag can never clear on its own (see
        # grounding_judge.check_grounding docstring) — only a still-unknown
        # name after the retry is real evidence the answer isn't grounded.
        blocking_calls = [c for c in ungrounded_calls if c.get("kind") != "ambiguous_overload"]
        degraded = bool(blocking_calls)

        sources = _build_sources(read_entries)

        if degraded:
            answer = _fallback_answer(sources, reason="ungrounded")

        filters = {
            "mode": "agentic",
            "steps_used": steps_used,
            "grounded": bool(read_entries),
            "degraded_answer": degraded,
            "ungrounded_calls": ungrounded_calls,
            "grounding_retries_used": retries_used,
            "token_usage": _sum_token_usage(result["messages"]),
        }

        if _debug:
            self._debug_writer.dump(result["messages"], question, extra={
                "steps_used": steps_used,
                "degraded_answer": degraded,
                "grounded": bool(read_entries),
                "ungrounded_calls": ungrounded_calls,
                "grounding_retries_used": retries_used,
            })

        return {
            "answer": answer,
            "sources": sources,
            "filters": filters,
            "error": None,
        }

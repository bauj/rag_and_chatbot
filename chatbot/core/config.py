"""
Configuration management for Documentation RAG Chatbot.
Supports nested JSON config blocks for llm, embedding, and reranker.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:8080/v1"
    model: str = "mistral"
    api_key: str = "dummy"
    ssl_cert_file: Optional[str] = None


@dataclass
class EmbeddingConfig:
    model: str = "all-MiniLM-L6-v2"
    type: str = "local"          # "local" (sentence-transformers) or "api" (OpenAI-compatible)
    base_url: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class RerankerConfig:
    model: str = "BAAI/bge-reranker-v2-m3"
    type: str = "local"          # "local" only for now


@dataclass
class AgenticConfig:
    page_index_path: str           # required — path to page_index.json (relative or absolute)
    max_chars_per_page: int = 8000
    max_pages_per_round: int = 3
    max_pages_round2: int = 2
    max_steps: int = 6             # CodeAgent step budget for agentic-smol mode


@dataclass
class ChatbotConfig:
    """Configuration for Documentation RAG Chatbot with nested JSON support."""

    project_name: str = "docs"
    chromadb_path: str = ""      # defaults to ../{project_name}_docs_extracted/chromadb

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: Optional[RerankerConfig] = None   # None = reranking disabled
    agentic: Optional[AgenticConfig] = None    # None = agentic mode unavailable

    # Retrieval parameters
    k_standard: int = 40
    k_deep_dive: int = 60
    # Per-channel retrieval depth (how many candidates the vector retriever and BM25 each
    # fetch before fusion), decoupled from k_standard/k_deep_dive (the reranker pool size —
    # what RRF's fused output is truncated to). None means "old behaviour": per-channel
    # depth equals the pool size. Reciprocal rank fusion merges two k-length lists and
    # truncates back to k, so each channel effectively contributes only ~k/2 of its
    # candidates to the pool — a doc ranked outside that in only one channel never
    # reaches reranking even if a deeper look would have found it. Setting k_retrieve
    # deeper than the pool (e.g. 80 vs a 40 pool) fixes that without growing reranker
    # cost, which tracks the pool size, not this value.
    k_retrieve: Optional[int] = None
    deep_dive_batch_size: int = 10
    top_n_after_rerank: int = 15    # docs to keep after cross-encoder reranking
    # Total characters parent-section expansion may add across all surviving docs.
    # Expansion rebuilds each doc's full section from the corpus (uncapped, unlike the
    # extractor's 5,000-char metadata copy), so a single 54,000-char page could otherwise
    # crowd the context window. Sections that no longer fit fall back to the capped copy.
    expansion_char_budget: int = 60000
    bm25_enabled: bool = False      # opt-in BM25 hybrid retrieval (fused with vector search via RRF)
    # opt-in third RRF list: BM25 over chunk TITLES only, collapsed to one hit per page.
    # A question naming a class/identifier is an entity lookup, not a semantic search — the
    # identifier is one token among the ~100+ of a chunk body but the whole of a page's
    # title, so this channel is far more precise for "what does class X do" style questions.
    # The title index lives inside BM25Index, so this requires bm25_enabled's index to be
    # loaded; it has no effect on its own.
    title_boost_enabled: bool = False
    hyde_enabled: bool = False      # opt-in HyDE: embed an LLM-written hypothetical passage instead of the raw question
    smol_enabled: bool = False      # opt-in agentic-smol mode (smolagents CodeAgent multi-hop). Requires agentic block too.

    # LLM generation parameters
    temperature: float = 0.0
    max_tokens: int = 2000

    # Response style presets (used by web UI); ClassVar excludes this from dataclass instance fields.
    # Temperature only — search depth and deep dive have their own explicit controls, and
    # the presets used to declare 'k'/'deep_dive' values the UI never read.
    RESPONSE_STYLES: ClassVar[dict] = {
        "Precise (Recommended)": {"temperature": 0.0},
        "Balanced":              {"temperature": 0.2},
        "Comprehensive":         {"temperature": 0.1},
    }
    # Sentinel style meaning "don't override — use config.temperature".
    CONFIG_DEFAULT_STYLE: ClassVar[str] = "Config default"

    def __post_init__(self):
        if " " in self.project_name:
            raise ValueError(
                f"project_name must not contain spaces: {self.project_name!r}. "
                "Use underscores instead (e.g. 'my_project')."
            )
        if not self.chromadb_path:
            self.chromadb_path = f"../extraction/{self.project_name}_docs_extracted/chromadb"

    @property
    def collection_name(self) -> str:
        return f"{self.project_name}_documentation"

    @property
    def bm25_jsonl_path(self) -> str:
        return str(Path(self.chromadb_path).parent / f"{self.project_name}_docs.jsonl")

    @classmethod
    def load(cls, config_file: Optional[str] = None) -> "ChatbotConfig":
        """Load config: explicit file > config.json in cwd > defaults."""
        if config_file:
            return cls._from_json(config_file)
        default = Path("config.json")
        if default.exists():
            return cls._from_json(str(default))
        return cls()

    @classmethod
    def _from_json(cls, path: str) -> "ChatbotConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        try:
            with open(config_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

        # Strip top-level comment keys
        data = {k: v for k, v in data.items() if not k.startswith("_")}

        # Known top-level keys (scalars + nested blocks)
        scalar_keys = {"project_name", "chromadb_path", "k_standard", "k_deep_dive", "k_retrieve",
                       "deep_dive_batch_size", "top_n_after_rerank", "expansion_char_budget", "temperature", "max_tokens",
                       "bm25_enabled", "title_boost_enabled", "hyde_enabled", "smol_enabled"}
        nested_keys = {"llm", "embedding", "reranker", "agentic"}
        valid_keys = scalar_keys | nested_keys
        invalid = set(data.keys()) - valid_keys
        if invalid:
            raise ValueError(f"Invalid config keys: {invalid}")

        def strip_comments(d: dict) -> dict:
            return {k: v for k, v in d.items() if not k.startswith("_")}

        def _make(cls_, d: dict, block_name: str):
            clean = {k: v for k, v in d.items() if not k.startswith("_")}
            try:
                return cls_(**clean)
            except TypeError as e:
                raise ValueError(f"Invalid keys in '{block_name}' config block: {e}")

        kwargs = {k: v for k, v in data.items() if k in scalar_keys}

        if "llm" in data:
            kwargs["llm"] = _make(LLMConfig, data["llm"], "llm")
        if "embedding" in data:
            kwargs["embedding"] = _make(EmbeddingConfig, data["embedding"], "embedding")
        if "reranker" in data:
            kwargs["reranker"] = _make(RerankerConfig, data["reranker"], "reranker") if data["reranker"] else None
        if "agentic" in data:
            kwargs["agentic"] = _make(AgenticConfig, data["agentic"], "agentic") if data["agentic"] else None

        return cls(**kwargs)

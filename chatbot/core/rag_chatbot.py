"""
Core business logic for Documentation RAG Chatbot
Pure logic with no UI concerns - returns data structures only
"""

import os
import warnings
os.environ['POSTHOG_DISABLED'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'false'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Optional, Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.callbacks import UsageMetadataCallbackHandler

from .config import ChatbotConfig
from .bm25_index import BM25Index, reciprocal_rank_fusion


def _sum_usage_metadata(usage_metadata: Dict[str, dict]) -> Optional[Dict[str, int]]:
    """Sum a UsageMetadataCallbackHandler's per-model usage dict into the same
    {input_tokens, output_tokens, total_tokens} shape agentic/deepagents modes
    report, so eval results are comparable across modes. None if no LLM call
    reported usage (e.g. an endpoint that doesn't return it)."""
    if not usage_metadata:
        return None
    input_tokens = output_tokens = 0
    for usage in usage_metadata.values():
        input_tokens += usage.get("input_tokens", 0) or 0
        output_tokens += usage.get("output_tokens", 0) or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


class DocumentationChatbot:
    """
    Core RAG chatbot for documentation

    Handles vectorstore, LLM, and retrieval logic without UI concerns.
    Returns structured data that UI layers can format as needed.
    """

    def __init__(self, config: ChatbotConfig, skip_reranker: bool = False):
        """
        skip_reranker: skip loading the configured reranker model. no-rag mode
        never touches self.reranker, so loading it (e.g. a ColBERT
        MultiVectorEncoder) would be pure per-question startup cost for no benefit.
        """
        self.config = config
        print("DEBUG : load vector store ...")
        self.vectorstore = self._load_vectorstore()
        print("DEBUG : detect modules ...")
        self.available_modules = self._detect_modules()
        print("DEBUG : initialize LLM ...")
        self.llm = self._initialize_llm()
        print("DEBUG : create prompt ...")
        self.base_prompt = self._create_prompt()
        self.reranker = None if skip_reranker else self._load_reranker()
        self.bm25_index = self._load_bm25_index()
        self.hyde_llm = self._initialize_llm(temperature=0.0) if config.hyde_enabled else None
        self.no_rag_prompt = self._create_no_rag_prompt()

    def _load_vectorstore(self) -> Chroma:
        """Load ChromaDB vector store"""
        db_path = Path(self.config.chromadb_path)
        if not db_path.is_absolute():
            db_path = Path(__file__).parent.parent / self.config.chromadb_path
        db_path = db_path.resolve()

        if not db_path.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at: {db_path}\n"
                f"Run the processor first: cd ../extraction && python process_docs.py --config config.json"
            )

        emb_cfg = self.config.embedding
        if emb_cfg.type == "local":
            embeddings = HuggingFaceEmbeddings(model_name=emb_cfg.model)
        else:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(
                model=emb_cfg.model,
                base_url=emb_cfg.base_url,
                api_key=emb_cfg.api_key,
            )

        vectorstore = Chroma(
            persist_directory=str(db_path),
            embedding_function=embeddings,
            collection_name=self.config.collection_name,
        )
        return vectorstore

    def _detect_modules(self) -> List[str]:
        """Detect which modules are in the database"""
        try:
            modules = set()
            batch_size = 5000
            offset = 0
            while True:
                batch = self.vectorstore._collection.get(
                    include=['metadatas'],
                    limit=batch_size,
                    offset=offset,
                )
                for meta in batch['metadatas']:
                    if 'module' in meta:
                        modules.add(meta['module'])
                if len(batch['metadatas']) < batch_size:
                    break
                offset += batch_size
            return sorted(list(modules))
        except Exception as e:
            raise RuntimeError(
                f"Failed to detect modules from ChromaDB: {e}\n"
                f"Make sure the database was populated by the extraction pipeline."
            )

    def _initialize_llm(self, temperature: Optional[float] = None,
                        max_tokens: Optional[int] = None) -> ChatOpenAI:
        """Initialize LLM with OpenAI-compatible API"""
        llm_cfg = self.config.llm
        if llm_cfg.ssl_cert_file:
            cert_path = Path(llm_cfg.ssl_cert_file).expanduser().resolve()
            if not cert_path.exists():
                raise FileNotFoundError(f"SSL certificate file not found: {cert_path}")
            os.environ['SSL_CERT_FILE'] = str(cert_path)
            os.environ['REQUESTS_CA_BUNDLE'] = str(cert_path)

        effective_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        # model_kwargs bypasses langchain-openai's max_tokens→max_completion_tokens renaming,
        # which breaks non-OpenAI endpoints. Suppress the resulting "should be specified
        # explicitly" warning since this is intentional.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Parameters \{'max_tokens'\} should be specified explicitly",
                category=UserWarning,
            )
            llm = ChatOpenAI(
                model=llm_cfg.model,
                base_url=llm_cfg.base_url,
                api_key=llm_cfg.api_key,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_completion_tokens=None,  # prevent langchain-openai from sending this field
                streaming=True,
                model_kwargs={"max_tokens": effective_max_tokens},
            )
        return llm

    def _load_reranker(self):
        """
        Load the configured reranker. Returns None when disabled.

        reranker.type picks the scoring method: "cross_encoder" (default) loads
        sentence_transformers.CrossEncoder; "late_interaction" loads
        sentence_transformers.MultiVectorEncoder for ColBERT-style MaxSim scoring,
        which requires sentence-transformers >= 6.0 (MultiVectorEncoder was added
        in that release). score_against_query() picks the matching scoring call
        based on this same type.
        """
        if self.config.reranker is None:
            return None

        model = self.config.reranker.model

        if self.config.reranker.type == "late_interaction":
            try:
                from sentence_transformers import MultiVectorEncoder
            except ImportError:
                raise ImportError(
                    "reranker.type 'late_interaction' requires sentence-transformers >= 6.0 "
                    "(MultiVectorEncoder was added in that release).\n"
                    "Upgrade it: pip install -U sentence-transformers"
                )
            print(f"DEBUG : load late-interaction reranker ({model}) ...")
            return MultiVectorEncoder(model)

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for reranking.\n"
                "Install it: pip install sentence-transformers"
            )
        print(f"DEBUG : load reranker ({model}) ...")
        return CrossEncoder(model)

    def _load_bm25_index(self) -> Optional[BM25Index]:
        """Load the BM25 keyword index if bm25_enabled and its JSONL corpus exists."""
        if not self.config.bm25_enabled:
            return None

        jsonl_path = Path(self.config.bm25_jsonl_path)
        if not jsonl_path.is_absolute():
            jsonl_path = Path(__file__).parent.parent / jsonl_path
        jsonl_path = jsonl_path.resolve()

        if not jsonl_path.exists():
            print(f"Warning: bm25_enabled is True but {jsonl_path} was not found — hybrid retrieval disabled")
            return None

        print(f"DEBUG : load BM25 index ({jsonl_path}) ...")
        return BM25Index(str(jsonl_path))

    def _generate_hyde_passage(self, question: str,
                                usage_callback: Optional[UsageMetadataCallbackHandler] = None) -> str:
        """
        Generate a hypothetical documentation passage answering `question` (HyDE).

        Used only to steer the dense vector search — BM25 and reranking still
        score against the real question. Falls back to the raw question on any
        LLM failure so retrieval degrades to plain vector search instead of erroring.
        """
        prompt = PromptTemplate.from_template(
            """Write a short hypothetical passage (2-4 sentences) from {project_name}'s technical documentation that would answer the following question. Write it as if it were real documentation, not an explanation.

Question: {question}

Passage:"""
        )
        try:
            chain = prompt | self.hyde_llm | StrOutputParser()
            config = {"callbacks": [usage_callback]} if usage_callback else None
            return chain.invoke({"project_name": self.config.project_name, "question": question}, config=config)
        except Exception as e:
            print(f"Warning: HyDE passage generation failed ({e}) — falling back to raw question")
            return question

    # Title channel size: how many page-level hits search_titles() contributes to
    # fusion. Kept far shallower than retrieve_k (10 vs. e.g. 80) because the title
    # channel is high-precision/low-recall by construction — few pages share an
    # exact identifier, so there is little to gain from searching it deep, and a
    # short list keeps an unrelated but token-overlapping title from crowding the
    # fused pool. See _hybrid_retrieve's docstring for the full rationale.
    _TITLE_CHANNEL_K = 10

    def _hybrid_retrieve(self, query: str, vector_docs: List, k: int,
                         module_filter: Optional[str], doc_category_filter: Optional[str],
                         bm25_enabled: bool = True, retrieve_k: Optional[int] = None,
                         title_boost_enabled: bool = False) -> List:
        """
        Fuse the vector retriever's results with BM25 keyword search and, optionally,
        a title/identifier search, via Reciprocal Rank Fusion (RRF).

        Two k's, two jobs:
          - retrieve_k: how deep BM25 (and, via the caller's search_kwargs, the
            vector retriever) searches before fusion. Defaults to k when unset,
            reproducing the historical behaviour where one number did both jobs.
          - k: the fused pool size handed onward to reranking — the pre-existing
            k_standard/k override semantics, unchanged.
          Splitting them matters because reciprocal_rank_fusion(lists, k) merges
          k-length lists and truncates the result back to k: a doc appearing in
          only one channel effectively gets ~k/2 of headroom before it is cut.
          Measured on the SHAPER corpus: a target chunk sat at dense rank 46-47
          and BM25 rank 44 — inside both channels' top-60/80 — and still missed
          the fused pool when retrieval depth equalled the pool size. Searching
          deeper per channel while keeping k fixed fixes that without growing
          reranker cost, which tracks k, not retrieve_k.

        Title channel (gated by title_boost_enabled): a question naming a class
        ("What are the public methods of the ModelAPI_Feature class?") is an
        entity lookup, not a semantic search. The discriminating identifier is
        one token among ~150 in a chunk's body — diluted further in corpora with
        Doxygen inherited-member copies, where every subclass page's memitems
        repeat the parent class's name (measured: 1,796 chunk bodies vs. 112
        chunk titles contain the same identifier, 16x sharper) — but it is the
        *entire* title of the page that declares it. RRF only looks at rank, not
        score, so even a short, high-precision title list can out-rank a
        body/dense channel where the same page is buried at rank 40+.
        BM25Index.search_titles() collapses each page's chunks to one hit before
        this ever sees them.

        Returns vector_docs unchanged when no BM25 index is loaded, or when
        neither bm25_enabled nor title_boost_enabled contribute anything to fuse
        (preserves the original list, including its identity, for callers that
        compare by ==).
        """
        if self.bm25_index is None:
            return vector_docs

        effective_retrieve_k = retrieve_k if retrieve_k is not None else k

        ranked_lists = [vector_docs]

        if bm25_enabled:
            ranked_lists.append(self.bm25_index.search(
                query, k=effective_retrieve_k, module_filter=module_filter, doc_category_filter=doc_category_filter,
            ))

        if title_boost_enabled:
            ranked_lists.append(self.bm25_index.search_titles(
                query, k=self._TITLE_CHANNEL_K, module_filter=module_filter, doc_category_filter=doc_category_filter,
            ))

        if len(ranked_lists) == 1:
            return vector_docs

        return reciprocal_rank_fusion(ranked_lists, k=k)

    @staticmethod
    def _declaring_page_names(hierarchy: str) -> set:
        """
        Doxygen page names that would hold the declaration of a qualified symbol.

        'std::shared_ptr< ModelAPI_Result > ModelAPI_Feature::lastResult' yields
        {'classModelAPI__Feature.html', 'structModelAPI__Feature.html',
         'interfaceModelAPI__Feature.html'} — Doxygen escapes '_' as '__' in filenames
        and prefixes the file by the entity kind, which the metadata does not record.

        Returns an empty set when no qualified name can be read out of hierarchy.
        """
        match = re.search(r'([A-Za-z_]\w*)::~?[A-Za-z_]\w*\s*$', hierarchy or '')
        if not match:
            return set()
        escaped = match.group(1).replace('_', '__')
        return {f"{kind}{escaped}.html" for kind in ('class', 'struct', 'interface')}

    def _dedup_symbol_copies(self, docs: List) -> List:
        """
        Collapse duplicate copies of the same documented symbol.

        When docs are built with Doxygen's INLINE_INHERITED_MEMB, every inherited
        member's documentation is copied verbatim onto every subclass page — in the
        SHAPER corpus used for testing, one symbol occupied 180 chunks. Left alone, a
        single inherited method fills the whole result set and crowds out the class
        page the question was actually about. Extraction can strip these copies, but
        this guard also covers corpora extracted before that, and any other source of
        byte-identical duplicates.

        Copies are keyed by anchor_id, which Doxygen reuses across every page carrying
        the member, including the declaring class's own page. Chunks with no anchor
        (section-level chunks) fall back to their exact content.

        The declaring class's own copy wins when it is present, so citations point at
        the canonical documentation rather than an arbitrary subclass. Otherwise the
        first-retrieved copy wins: copies within a group are identical or near-identical,
        so reranker scores tie and retrieval rank is the meaningful tiebreak.

        Input order is preserved.
        """
        best: Dict[str, int] = {}
        result: List = []

        for doc in docs:
            anchor = doc.metadata.get('anchor_id', '')
            key = f"a:{anchor}" if anchor else f"c:{hashlib.sha1(doc.page_content.encode('utf-8')).hexdigest()}"

            page = doc.metadata.get('url', '').split('#')[0].rsplit('/', 1)[-1]
            is_declaring = page in self._declaring_page_names(doc.metadata.get('hierarchy', ''))

            if key not in best:
                best[key] = len(result)
                result.append(doc)
            elif is_declaring:
                # Replace in place: keeps the group at its best retrieval rank.
                result[best[key]] = doc

        return result

    def _corpus_rows(self) -> List[dict]:
        """
        Every chunk of the corpus, as written by the extraction pipeline.

        Borrowed from the BM25 index when one is loaded, since that is the same JSONL
        parsed already. Expansion is not a retrieval concern, though — a bm25_enabled
        =False run still has to expand its survivors — so the file is read directly
        when there is no index to borrow from. A missing file yields no rows, which
        degrades expansion to "leave the chunk as retrieved" rather than raising.
        """
        index = getattr(self, 'bm25_index', None)
        if index is not None:
            return getattr(index, 'rows', [])

        path = Path(self.config.bm25_jsonl_path)
        if not path.is_absolute():
            path = (Path(__file__).parent.parent / path).resolve()
        if not path.exists():
            return []

        rows = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _section_rows(self) -> dict:
        """
        Map section_id -> its chunks, ordered by chunk_position, built once from the
        corpus. Empty when no corpus is available.
        """
        cached = getattr(self, '_section_rows_cache', None)
        if cached is not None:
            return cached

        rows: dict = {}
        for row in self._corpus_rows():
            section_id = row.get('metadata', {}).get('section_id', '')
            if section_id:
                rows.setdefault(section_id, []).append(row)

        def position(row):
            pos = row.get('metadata', {}).get('chunk_position', '')
            head = pos.split('/')[0]
            return int(head) if head.isdigit() else 0

        for fragments in rows.values():
            fragments.sort(key=position)

        self._section_rows_cache = rows
        return rows

    # Longest suffix/prefix match used to rejoin fragments. The extractor's chunker
    # overlaps consecutive chunks (200 chars by default, or the token equivalent), so
    # a naive concatenation would repeat that window.
    _MAX_FRAGMENT_OVERLAP = 500

    # Below this many chars, agentic section retrieval hands back a whole page
    # instead of the single matched h2/h3 section. Small Sphinx user-doc feature
    # pages split into sections whose individual text drops half of a multi-part
    # answer; large Doxygen class pages stay section-level (#156/#157).
    _SMALL_PAGE_CHARS = 8000

    def _full_section_text(self, section_id: str) -> str:
        """
        Rebuild a section's complete text from its chunks.

        A page-level class summary longer than the chunk size is split into many chunks
        sharing one section_id; section dedup keeps one of them and expansion to
        metadata['section_text'] is capped at 5,000 chars by the extractor, so half the
        class summaries can never reach the LLM in full (task #107). The chunks
        themselves are already in memory for BM25, so the section can be reassembled
        from them — uncapped, and with no re-extraction.

        Returns '' when the section is unknown, which callers treat as "fall back to
        metadata['section_text']" (old ChromaDB indexes, or bm25 disabled).
        """
        fragments = self._section_rows().get(section_id, [])
        if not fragments:
            return ''

        text = fragments[0].get('content', '')
        for row in fragments[1:]:
            piece = row.get('content', '')
            # Every summary fragment repeats the page title as its first line.
            header = f"{row.get('title', '')}\n\n"
            if row.get('title') and piece.startswith(header):
                piece = piece[len(header):]

            window = min(len(text), len(piece), self._MAX_FRAGMENT_OVERLAP)
            overlap = 0
            for size in range(window, 0, -1):
                if text.endswith(piece[:size]):
                    overlap = size
                    break
            text += piece[overlap:]

        return text.strip()

    def _page_section_ids(self) -> dict:
        """page key (the section_id prefix) -> [section_id, ...] in corpus order.

        Built once from the corpus, same source as _section_rows.
        """
        cached = getattr(self, '_page_section_ids_cache', None)
        if cached is not None:
            return cached

        pages: dict = {}
        for row in self._corpus_rows():
            section_id = row.get('metadata', {}).get('section_id', '')
            if not section_id:
                continue
            page_key = section_id.split('#', 1)[0]
            ids = pages.setdefault(page_key, [])
            if section_id not in ids:
                ids.append(section_id)

        self._page_section_ids_cache = pages
        return pages

    def _full_page_text(self, page_key: str) -> str:
        """Rebuild a whole page from its sections' chunks, sections in corpus
        order. '' when the page is unknown (old index, or bm25 disabled)."""
        if not page_key:
            return ''
        parts = [self._full_section_text(section_id)
                 for section_id in self._page_section_ids().get(page_key, [])]
        return "\n\n".join(p for p in parts if p).strip()

    def _expand_to_section(self, doc, budget: int):
        """
        Swap a chunk for its full section text, honouring a remaining char budget.

        Returns (doc, chars_spent). Prefers the corpus reconstruction and falls back to
        the extractor's capped metadata copy — which is also what a section too large
        for the remaining budget gets, so one 54,000-char page cannot crowd out the
        other survivors.
        """
        from langchain_core.documents import Document as LCDocument

        capped = doc.metadata.get('section_text', '')
        full = self._full_section_text(doc.metadata.get('section_id', ''))

        text = full if full and len(full) <= budget else capped
        if not text:
            return doc, 0

        return LCDocument(page_content=text, metadata=doc.metadata), len(text)

    def score_against_query(self, query: str, docs: List) -> List[float]:
        """
        Score each doc's relevance to the query using the configured reranker,
        best-match score highest. Dispatches on config.reranker.type: cross-encoder
        .predict() (default) or ColBERT-style late-interaction (MaxSim).

        Public: also used by extraction/html_parser.py's search_pages() (agentic/
        deepagents mode's page-search tool) via dependency injection, the same
        duck-typed-callable pattern already used for bm25_index — keeps
        extraction/ independent of chatbot/core.
        """
        if self.config.reranker is not None and self.config.reranker.type == "late_interaction":
            # encode_query/encode_document are separate calls (not interchangeable:
            # these models use different prefixes/length caps per side), per
            # MultiVectorEncoder's API.
            query_emb = self.reranker.encode_query([query])
            doc_embs = self.reranker.encode_document([doc.page_content for doc in docs])
            return list(self.reranker.similarity(query_emb, doc_embs)[0])

        pairs = [(query, doc.page_content) for doc in docs]
        return list(self.reranker.predict(pairs))

    def _pick_section_representatives(self, docs: List) -> List:
        """
        Keep one chunk per section_id, the first in the given order.

        Expansion swaps each survivor for its whole section, so two chunks of one
        section would send that section's text twice. "First" is only as meaningful
        as the ordering handed in — cross-encoder score when reranking is on, RRF
        fusion rank when it is off.

        Docs with an empty section_id (old ChromaDB databases written before section
        metadata existed) all pass through. Intentional backward compatibility.
        """
        result = []
        seen: set = set()

        for doc in docs:
            section_id = doc.metadata.get('section_id', '')
            if section_id:
                if section_id in seen:
                    continue
                seen.add(section_id)
            result.append(doc)

        return result

    def _expand_survivors(self, docs: List, expansion_char_budget: Optional[int] = None) -> List:
        """Swap each doc for its full section, spending one shared char budget."""
        result = []
        budget = expansion_char_budget if expansion_char_budget is not None else self.config.expansion_char_budget

        for doc in docs:
            doc, spent = self._expand_to_section(doc, budget)
            budget -= spent
            result.append(doc)

        return result

    def _select_context(self, query: str, docs: List,
                        reranker_enabled: bool = True,
                        top_n: Optional[int] = None,
                        expansion_char_budget: Optional[int] = None) -> List:
        """
        Turn a retrieval pool into the documents the LLM actually reads.

        Four independent steps, only one of which the cross-encoder owns:
          1. Collapse copies of the same documented symbol (corpus-defect guard).
          2. Order the pool — by cross-encoder score, or by the incoming RRF rank
             when reranking is off or unconfigured.
          3. Keep one representative per section, then cap at top_n
             (or config.top_n_after_rerank).
          4. Expand each survivor to its full section text.

        Disabling the reranker used to skip steps 3 and 4 as a side effect of an
        early return, so it changed ranking, context size and chunk granularity at
        once. It now varies exactly the ordering.
        """
        limit = top_n if top_n is not None else self.config.top_n_after_rerank

        docs = self._dedup_symbol_copies(docs)

        if self.reranker is not None and reranker_enabled:
            scores = self.score_against_query(query, docs)
            docs = [d for _score, d in
                    sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)]

        docs = self._pick_section_representatives(docs)[:limit]

        return self._expand_survivors(docs, expansion_char_budget=expansion_char_budget)

    def expand_sections(self, query: str, chunks: List, top_n: int = 5,
                        char_budget: int = 15000) -> List:
        """Rank a chunk pool and return up to top_n relevant sections as fully
        reconstructed Documents.

        Injected into extraction/html_parser.search_pages for agentic
        section-level retrieval, the same duck-typed way as score_against_query
        and bm25_index, so extraction/ stays independent of chatbot/core.

        For a small cohesive page (<= _SMALL_PAGE_CHARS whole) the matched
        section is replaced by the whole page, deduped so the page is emitted
        once. A multi-part question — "name the two algorithms AND give the TUI
        signature" — has its two halves in different h2/h3 sections of such a
        page, and a single section starves groundedness (#157). Large Doxygen
        class pages stay section-level (the point of #156). One shared
        char_budget across all survivors, same as _expand_survivors.
        """
        from langchain_core.documents import Document as LCDocument

        docs = self._dedup_symbol_copies(chunks)

        if self.reranker is not None:
            scores = self.score_against_query(query, docs)
            docs = [d for _score, d in
                    sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)]

        reps = self._pick_section_representatives(docs)

        result: List = []
        budget = char_budget
        seen_pages: set = set()
        for doc in reps:
            if len(result) >= top_n:
                break
            section_id = doc.metadata.get('section_id', '')
            page_key = section_id.split('#', 1)[0] if section_id else ''
            full_page = self._full_page_text(page_key)

            if full_page and len(full_page) <= self._SMALL_PAGE_CHARS:
                if page_key in seen_pages:
                    continue  # whole page already emitted for a sibling section
                seen_pages.add(page_key)
                text = full_page if len(full_page) <= budget else doc.metadata.get('section_text', '')
                if not text:
                    continue
                result.append(LCDocument(page_content=text, metadata=doc.metadata))
                budget -= len(text)
            else:
                expanded, spent = self._expand_to_section(doc, budget)
                budget -= spent
                result.append(expanded)

        return result

    def _create_prompt(self) -> PromptTemplate:
        """Create the base prompt template"""
        module_lines = "\n".join(
            f"- {m}" for m in self.available_modules
        )
        template = f"""You are an expert assistant for {self.config.project_name} documentation.

Available modules:
{module_lines}

Each module has:
- Dev docs: API reference (classes, methods, namespaces)
- User docs: Tutorials, guides, usage examples
- Methodology docs: Conceptual explanations, theory

CRITICAL INSTRUCTIONS:
1. You MUST base your answer ONLY on the documentation provided below
2. DO NOT use any external knowledge or information from your training data
3. If the documentation does not contain enough information to answer the question, explicitly say "I don't have enough information in the documentation to answer this question"
4. ALWAYS cite which module and document type you're referencing
5. If multiple modules are relevant, explain which provides which functionality
6. If the question is not related to {self.config.project_name} documentation, politely decline to answer

Documentation:
{{context}}

Question: {{question}}

Answer (based strictly on the documentation above):"""
        return PromptTemplate.from_template(template)

    def _create_no_rag_prompt(self) -> PromptTemplate:
        """Prompt for the no-retrieval baseline — no context block, no module list,
        so the answer can only come from the LLM's own training knowledge."""
        template = f"""You are an expert assistant for {self.config.project_name}.

CRITICAL INSTRUCTIONS:
1. Answer using your own knowledge of {self.config.project_name} from your training data.
2. If you are not confident in the answer, explicitly say "I don't have enough information to answer this question confidently."
3. If the question is not related to {self.config.project_name}, politely decline to answer.

Question: {{question}}

Answer:"""
        return PromptTemplate.from_template(template)

    def ask_no_rag(self, question: str,
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Ask a question with no retrieval — isolates how much of a normal RAG
        answer's correctness comes from retrieval vs. the LLM's own training
        knowledge of the corpus's subject matter.

        Returns the same {answer, sources, filters, error} shape as ask(),
        with sources always empty and filters always {"mode": "no-rag"}.
        """
        llm = self.llm if temperature is None and max_tokens is None \
            else self._initialize_llm(temperature=temperature, max_tokens=max_tokens)
        chain = self.no_rag_prompt | llm | StrOutputParser()
        try:
            answer = chain.invoke({"question": question})
            return {"answer": answer, "sources": [], "filters": {"mode": "no-rag"}, "error": None}
        except Exception as e:
            return {"answer": None, "sources": [], "filters": {"mode": "no-rag"}, "error": str(e)}

    def _create_chain(self,
                     module_filter: Optional[str] = None,
                     doc_type_filter: Optional[str] = None,
                     deep_dive: bool = False,
                     k: Optional[int] = None,
                     temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None,
                     reranker_enabled: bool = True,
                     top_n: Optional[int] = None,
                     hyde_enabled: Optional[bool] = None,
                     bm25_enabled: Optional[bool] = None,
                     title_boost_enabled: Optional[bool] = None,
                     k_retrieve: Optional[int] = None,
                     deep_dive_batch_size: Optional[int] = None,
                     expansion_char_budget: Optional[int] = None,
                     usage_callback: Optional[UsageMetadataCallbackHandler] = None):
        """
        Create RAG chain with optional filtering

        Args:
            module_filter: Filter by module (or None)
            doc_type_filter: Filter by doc type (dev, user, methodology, or None)
            deep_dive: Use deep dive mode (more docs + summarization)
            k: Override number of chunks to retrieve (optional)
            temperature: Override LLM temperature (optional)
            max_tokens: Override LLM max_tokens (optional)
            reranker_enabled: Runtime toggle for cross-encoder reranking
            top_n: Override docs kept after reranking
            hyde_enabled: Runtime toggle for HyDE (None = whatever was configured)
            bm25_enabled: Runtime toggle for BM25 hybrid retrieval (None = whatever was configured)
            title_boost_enabled: Runtime toggle for the title/identifier RRF channel
                (None = whatever was configured)
            k_retrieve: Override per-channel retrieval depth before RRF fusion (optional,
                ignored in deep dive mode — see below)
            deep_dive_batch_size: Override deep dive's summarization batch size (optional)
            expansion_char_budget: Override the shared char budget for section expansion (optional)

        Returns:
            Tuple of (chain, retriever, source_docs_holder). source_docs_holder is a
            per-call list that the standard chain fills with the docs actually sent
            to the LLM; it stays empty for the deep-dive chain.
        """
        # Set k based on priority: runtime override > deep_dive mode > config default
        if k is None:
            k = self.config.k_deep_dive if deep_dive else self.config.k_standard

        # retrieve_k decouples per-channel search depth from k (the reranker pool).
        # Deep dive deliberately keeps using k unchanged — it bypasses this whole
        # rerank/BM25/title pipeline and retrieves+summarizes directly, so there is
        # no fusion step here for a deeper per-channel search to feed.
        effective_k_retrieve = k_retrieve if k_retrieve is not None else self.config.k_retrieve
        retrieve_k = k
        if not deep_dive and effective_k_retrieve is not None:
            retrieve_k = effective_k_retrieve
        search_kwargs = {"k": retrieve_k}

        effective_hyde = hyde_enabled if hyde_enabled is not None else (self.hyde_llm is not None)
        effective_hyde = effective_hyde and self.hyde_llm is not None

        effective_bm25 = bm25_enabled if bm25_enabled is not None else (self.bm25_index is not None)
        effective_bm25 = effective_bm25 and self.bm25_index is not None

        # Unlike bm25/hyde, no dedicated object's nullity encodes "config enabled this
        # at startup" — the title index lives inside bm25_index regardless of this flag.
        # So the startup gate is config.title_boost_enabled itself, not an object check.
        effective_title_boost = title_boost_enabled if title_boost_enabled is not None else True
        effective_title_boost = effective_title_boost and self.config.title_boost_enabled and self.bm25_index is not None

        if deep_dive:
            # Deep dive uses its own retrieve-and-summarize path, so the whole
            # rerank/HyDE/BM25 pipeline is bypassed. Say so instead of failing silently.
            skipped = []
            if self.reranker is not None and reranker_enabled:
                skipped.append("reranking")
            if effective_hyde:
                skipped.append("HyDE")
            if effective_bm25:
                skipped.append("BM25 hybrid retrieval")
            if effective_title_boost:
                skipped.append("title/identifier channel")
            if skipped:
                print(f"Note: deep_dive=True — {', '.join(skipped)} skipped in deep dive mode.")

        # Create LLM with runtime overrides
        llm = self._initialize_llm(temperature=temperature, max_tokens=max_tokens)

        # Build filters
        filters = []
        if module_filter and module_filter != "All":
            filters.append({"module": module_filter})
        if doc_type_filter and doc_type_filter != "All":
            filters.append({"doc_category": doc_type_filter.lower()})

        if len(filters) == 1:
            search_kwargs["filter"] = filters[0]
        elif len(filters) > 1:
            search_kwargs["filter"] = {"$and": filters}

        # Create retriever
        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        # Per-call holder for the docs the standard chain actually sends to the LLM.
        source_docs_holder: List = []

        if deep_dive:
            # Deep dive: retrieve more docs and summarize in batches
            def summarize_batch(docs):
                summaries = []
                batch_size = deep_dive_batch_size if deep_dive_batch_size is not None else self.config.deep_dive_batch_size

                for i in range(0, len(docs), batch_size):
                    batch = docs[i:i + batch_size]
                    batch_text = "\n\n".join(doc.page_content for doc in batch)

                    summary_prompt = PromptTemplate.from_template(
                        """Summarize the following documentation excerpts for answering a technical question.
                        Focus on key technical details, APIs, and usage patterns.
                        Be concise but preserve all important technical information.

                        Documentation:
                        {context}

                        Summary:"""
                    )

                    summary_chain = (
                        {"context": RunnableLambda(lambda x: x)}
                        | summary_prompt
                        | llm
                        | StrOutputParser()
                    )

                    summary_config = {"callbacks": [usage_callback]} if usage_callback else None
                    summary = summary_chain.invoke(batch_text, config=summary_config)
                    summaries.append(summary)

                return "\n\n".join(summaries)

            # Deep dive chain
            chain = (
                {
                    "context": retriever | RunnableLambda(summarize_batch),
                    "question": RunnablePassthrough()
                }
                | self.base_prompt
                | llm
                | StrOutputParser()
            )
        else:
            # Standard chain — retrieve, rerank, format.
            # Results are stashed in source_docs_holder (a per-call closure list, NOT
            # instance state) so ask() can reuse them for source attribution without a
            # second retrieval + reranking pass. Per-call keeps concurrent requests —
            # e.g. two Gradio users at once — from overwriting each other's sources.
            def retrieve_and_format(question: str) -> str:
                vector_query = self._generate_hyde_passage(question, usage_callback=usage_callback) if effective_hyde else question
                raw_docs = retriever.invoke(vector_query)
                raw_docs = self._hybrid_retrieve(
                    question, raw_docs, k=k,
                    module_filter=module_filter if module_filter and module_filter != "All" else None,
                    doc_category_filter=doc_type_filter.lower() if doc_type_filter and doc_type_filter != "All" else None,
                    bm25_enabled=effective_bm25,
                    retrieve_k=retrieve_k,
                    title_boost_enabled=effective_title_boost,
                )
                reranked = self._select_context(
                    question, raw_docs,
                    reranker_enabled=reranker_enabled,
                    top_n=top_n,
                    expansion_char_budget=expansion_char_budget,
                )
                source_docs_holder[:] = reranked  # per-call cache for source attribution
                return "\n\n".join(doc.page_content for doc in reranked)

            chain = (
                {
                    "context": RunnableLambda(retrieve_and_format),
                    "question": RunnablePassthrough()
                }
                | self.base_prompt
                | llm
                | StrOutputParser()
            )

        return chain, retriever, source_docs_holder

    def ask(self,
            question: str,
            module: Optional[str] = None,
            doc_type: Optional[str] = None,
            deep_dive: bool = False,
            k: Optional[int] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            reranker_enabled: bool = True,
            top_n: Optional[int] = None,
            hyde_enabled: Optional[bool] = None,
            bm25_enabled: Optional[bool] = None,
            title_boost_enabled: Optional[bool] = None,
            k_retrieve: Optional[int] = None,
            deep_dive_batch_size: Optional[int] = None,
            expansion_char_budget: Optional[int] = None) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources

        Args:
            question: The question to ask
            module: Optional module filter
            doc_type: Optional doc type filter (dev, user)
            deep_dive: Use deep dive mode for comprehensive analysis
            k: Override number of chunks to retrieve (higher priority than config/deep_dive)
            temperature: Override LLM temperature (higher priority than config)
            max_tokens: Override LLM max_tokens (higher priority than config)
            reranker_enabled: Runtime toggle for cross-encoder reranking
            top_n: Override docs kept after reranking
            hyde_enabled: Runtime toggle for HyDE (None = whatever was configured)
            bm25_enabled: Runtime toggle for BM25 hybrid retrieval (None = whatever was configured)
            k_retrieve: Override per-channel retrieval depth before RRF fusion (higher
                priority than config; ignored in deep dive mode)
            deep_dive_batch_size: Override deep dive's summarization batch size (higher
                priority than config)
            expansion_char_budget: Override the shared char budget for section expansion
                (higher priority than config)
            title_boost_enabled: Runtime toggle for the title/identifier RRF channel
                (None = whatever was configured)

        Returns:
            Dict with:
                - answer: str - The generated answer
                - sources: List[Dict] - Source documents with metadata
                - filters: Dict - Applied filters
                - error: Optional[str] - Error message if failed
        """
        # Validate filters
        if module and module not in self.available_modules and module != "All":
            return {
                "answer": None,
                "sources": [],
                "filters": {},
                "error": f"Unknown module: {module}. Available: {', '.join(self.available_modules)}"
            }

        if doc_type and doc_type.lower() not in ['dev', 'user', 'methodology', 'all']:
            return {
                "answer": None,
                "sources": [],
                "filters": {},
                "error": f"Unknown doc type: {doc_type}. Available: dev, user, methodology"
            }

        # Track applied filters
        filters = {'mode': 'rag'}
        if module:
            filters['module'] = module
        if doc_type:
            filters['doc_type'] = doc_type
        if deep_dive:
            filters['deep_dive'] = True

        # Report which retrieval stages actually ran, so the UIs can stop
        # advertising toggles that deep dive silently bypasses.
        want_rerank = reranker_enabled and self.reranker is not None
        want_hyde = (hyde_enabled if hyde_enabled is not None else True) and self.hyde_llm is not None
        want_bm25 = (bm25_enabled if bm25_enabled is not None else True) and self.bm25_index is not None
        # No dedicated object's nullity encodes "title boost was enabled at startup"
        # (the title index lives inside bm25_index regardless of this flag), so the
        # startup gate is config.title_boost_enabled itself — see _create_chain.
        want_title_boost = (
            (title_boost_enabled if title_boost_enabled is not None else True)
            and self.config.title_boost_enabled
            and self.bm25_index is not None
        )
        filters['reranker'] = want_rerank and not deep_dive
        filters['hyde'] = want_hyde and not deep_dive
        filters['bm25'] = want_bm25 and not deep_dive
        filters['title_boost'] = want_title_boost and not deep_dive
        if deep_dive and (want_rerank or want_hyde or want_bm25 or want_title_boost):
            filters['bypassed_by_deep_dive'] = [
                name for name, wanted in
                (('reranker', want_rerank), ('hyde', want_hyde), ('bm25', want_bm25),
                 ('title_boost', want_title_boost))
                if wanted
            ]

        try:
            # Create chain and get answer (pass runtime overrides)
            usage_callback = UsageMetadataCallbackHandler()
            chain, retriever, source_docs_holder = self._create_chain(
                module, doc_type, deep_dive,
                k=k, temperature=temperature, max_tokens=max_tokens,
                reranker_enabled=reranker_enabled, top_n=top_n,
                hyde_enabled=hyde_enabled, bm25_enabled=bm25_enabled,
                title_boost_enabled=title_boost_enabled,
                k_retrieve=k_retrieve, deep_dive_batch_size=deep_dive_batch_size,
                expansion_char_budget=expansion_char_budget,
                usage_callback=usage_callback,
            )
            answer = chain.invoke(question, config={"callbacks": [usage_callback]})
            # Reuse docs already retrieved+reranked inside the standard chain.
            # Deep-dive chains leave the holder empty, so fall back to a
            # separate retrieval call for that path.
            if source_docs_holder:
                source_docs = list(source_docs_holder)
            else:
                raw_source_docs = retriever.invoke(question)
                source_docs = self._select_context(
                    question, raw_source_docs,
                    reranker_enabled=reranker_enabled,
                    top_n=top_n,
                    expansion_char_budget=expansion_char_budget,
                )

            # Format sources
            sources = []
            for doc in source_docs:
                sources.append({
                    'title': doc.metadata.get('title', 'Unknown'),
                    'module': doc.metadata.get('module', 'Unknown'),
                    'doc_category': doc.metadata.get('doc_category', 'Unknown'),
                    'doc_type': doc.metadata.get('doc_type', 'Unknown'),
                    'url': doc.metadata.get('url', ''),
                    'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                    # Untruncated text the LLM actually saw. 'content' above stays a
                    # short preview for display; automated consumers (groundedness
                    # judging in particular) need the whole chunk, since scoring an
                    # answer against a 200-character excerpt flags correct statements
                    # as hallucinations.
                    'full_content': doc.page_content,
                })

            # How much text the LLM actually read. Retrieval settings move this
            # silently — before #106, turning the reranker off swapped 15 expanded
            # sections for the whole ~40-doc pool unexpanded — so every eval result
            # file should carry the number rather than leave it to be inferred.
            filters['context_docs'] = len(source_docs)
            filters['context_chars'] = sum(len(d.page_content) for d in source_docs)
            filters['token_usage'] = _sum_usage_metadata(usage_callback.usage_metadata)

            return {
                "answer": answer,
                "sources": sources,
                "filters": filters,
                "error": None
            }

        except Exception as e:
            return {
                "answer": None,
                "sources": [],
                "filters": filters,
                "error": str(e)
            }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dict with stats by module and doc type
        """
        stats = {}
        total_chunks = 0
        batch_size = 5000
        offset = 0
        while True:
            batch = self.vectorstore._collection.get(
                include=['metadatas'],
                limit=batch_size,
                offset=offset,
            )
            for meta in batch['metadatas']:
                module = meta.get('module', 'Unknown')
                doc_cat = meta.get('doc_category', 'Unknown')
                if module not in stats:
                    stats[module] = {'dev': 0, 'user': 0, 'methodology': 0, 'total': 0}
                if doc_cat in ['dev', 'user', 'methodology']:
                    stats[module][doc_cat] += 1
                stats[module]['total'] += 1
            total_chunks += len(batch['metadatas'])
            if len(batch['metadatas']) < batch_size:
                break
            offset += batch_size

        return {
            'total_chunks': total_chunks,
            'modules': stats,
            'available_modules': self.available_modules
        }

    def get_available_modules(self) -> List[str]:
        """Get list of available modules"""
        return self.available_modules.copy()

    def get_chunk_count(self) -> int:
        """Get total number of documentation chunks"""
        return self.vectorstore._collection.count()

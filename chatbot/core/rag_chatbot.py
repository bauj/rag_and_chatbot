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

from .config import ChatbotConfig
from .bm25_index import BM25Index, reciprocal_rank_fusion


class DocumentationChatbot:
    """
    Core RAG chatbot for documentation

    Handles vectorstore, LLM, and retrieval logic without UI concerns.
    Returns structured data that UI layers can format as needed.
    """

    def __init__(self, config: ChatbotConfig):
        self.config = config
        print("DEBUG : load vector store ...")
        self.vectorstore = self._load_vectorstore()
        print("DEBUG : detect modules ...")
        self.available_modules = self._detect_modules()
        print("DEBUG : initialize LLM ...")
        self.llm = self._initialize_llm()
        print("DEBUG : create prompt ...")
        self.base_prompt = self._create_prompt()
        self.reranker = self._load_reranker()
        self.bm25_index = self._load_bm25_index()
        self.hyde_llm = self._initialize_llm(temperature=0.0) if config.hyde_enabled else None

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
        """Load cross-encoder reranker if configured. Returns None when disabled."""
        if self.config.reranker is None:
            return None
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for reranking.\n"
                "Install it: pip install sentence-transformers"
            )
        model = self.config.reranker.model
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

    def _generate_hyde_passage(self, question: str) -> str:
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
            return chain.invoke({"project_name": self.config.project_name, "question": question})
        except Exception as e:
            print(f"Warning: HyDE passage generation failed ({e}) — falling back to raw question")
            return question

    def _hybrid_retrieve(self, query: str, vector_docs: List, k: int,
                         module_filter: Optional[str], doc_category_filter: Optional[str],
                         bm25_enabled: bool = True) -> List:
        """
        Fuse BM25 keyword search results with the vector retriever's results via RRF.
        Returns vector_docs unchanged when no BM25 index is loaded or when
        bm25_enabled is False (runtime toggle from the UI).
        """
        if self.bm25_index is None or not bm25_enabled:
            return vector_docs

        bm25_docs = self.bm25_index.search(
            query, k=k, module_filter=module_filter, doc_category_filter=doc_category_filter,
        )
        return reciprocal_rank_fusion([vector_docs, bm25_docs], k=k)

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

    def _rerank_and_expand(self, query: str, docs: List,
                           reranker_enabled: bool = True,
                           top_n: Optional[int] = None) -> List:
        """
        Deduplicate symbol copies, rerank with cross-encoder, expand to section context.

        Symbol-level deduplication always runs — duplicates flood the context whether
        or not reranking is on. If no reranker is configured or reranker_enabled is
        False, the deduplicated docs are returned as-is. Otherwise:
          1. Scores all (query, page_content) pairs.
          2. Sorts by score descending.
          3. Deduplicates by section_id (keeps highest-scored chunk per section).
          4. Expands each surviving doc to its full section_text if available.
          5. Returns at most top_n (or config.top_n_after_rerank) docs.

        Note: docs with an empty section_id (e.g. old ChromaDB databases without
        section metadata) bypass section deduplication — all such docs pass through.
        This is intentional backward-compatibility behaviour.
        """
        docs = self._dedup_symbol_copies(docs)

        if self.reranker is None or not reranker_enabled:
            return docs

        limit = top_n if top_n is not None else self.config.top_n_after_rerank

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)

        scored = sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)

        result = []
        seen_sections: set = set()

        for _score, doc in scored:
            if len(result) >= limit:
                break
            section_id = doc.metadata.get('section_id', '')
            if section_id and section_id in seen_sections:
                continue
            if section_id:
                seen_sections.add(section_id)

            section_text = doc.metadata.get('section_text', '')
            if section_text:
                from langchain_core.documents import Document as LCDocument
                doc = LCDocument(page_content=section_text, metadata=doc.metadata)

            result.append(doc)

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
                     bm25_enabled: Optional[bool] = None):
        """
        Create RAG chain with optional filtering

        Args:
            module_filter: Filter by module (or None)
            doc_type_filter: Filter by doc type (dev, user, or None)
            deep_dive: Use deep dive mode (more docs + summarization)
            k: Override number of chunks to retrieve (optional)
            temperature: Override LLM temperature (optional)
            max_tokens: Override LLM max_tokens (optional)
            reranker_enabled: Runtime toggle for cross-encoder reranking
            top_n: Override docs kept after reranking
            hyde_enabled: Runtime toggle for HyDE (None = whatever was configured)
            bm25_enabled: Runtime toggle for BM25 hybrid retrieval (None = whatever was configured)

        Returns:
            Tuple of (chain, retriever, source_docs_holder). source_docs_holder is a
            per-call list that the standard chain fills with the docs actually sent
            to the LLM; it stays empty for the deep-dive chain.
        """
        # Set k based on priority: runtime override > deep_dive mode > config default
        if k is None:
            k = self.config.k_deep_dive if deep_dive else self.config.k_standard
        search_kwargs = {"k": k}

        effective_hyde = hyde_enabled if hyde_enabled is not None else (self.hyde_llm is not None)
        effective_hyde = effective_hyde and self.hyde_llm is not None

        effective_bm25 = bm25_enabled if bm25_enabled is not None else (self.bm25_index is not None)
        effective_bm25 = effective_bm25 and self.bm25_index is not None

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
                batch_size = self.config.deep_dive_batch_size

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

                    summary = summary_chain.invoke(batch_text)
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
                vector_query = self._generate_hyde_passage(question) if effective_hyde else question
                raw_docs = retriever.invoke(vector_query)
                raw_docs = self._hybrid_retrieve(
                    question, raw_docs, k=k,
                    module_filter=module_filter if module_filter and module_filter != "All" else None,
                    doc_category_filter=doc_type_filter.lower() if doc_type_filter and doc_type_filter != "All" else None,
                    bm25_enabled=effective_bm25,
                )
                reranked = self._rerank_and_expand(
                    question, raw_docs,
                    reranker_enabled=reranker_enabled,
                    top_n=top_n,
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
            bm25_enabled: Optional[bool] = None) -> Dict[str, Any]:
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

        if doc_type and doc_type.lower() not in ['dev', 'user', 'all']:
            return {
                "answer": None,
                "sources": [],
                "filters": {},
                "error": f"Unknown doc type: {doc_type}. Available: dev, user"
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
        filters['reranker'] = want_rerank and not deep_dive
        filters['hyde'] = want_hyde and not deep_dive
        filters['bm25'] = want_bm25 and not deep_dive
        if deep_dive and (want_rerank or want_hyde or want_bm25):
            filters['bypassed_by_deep_dive'] = [
                name for name, wanted in
                (('reranker', want_rerank), ('hyde', want_hyde), ('bm25', want_bm25))
                if wanted
            ]

        try:
            # Create chain and get answer (pass runtime overrides)
            chain, retriever, source_docs_holder = self._create_chain(
                module, doc_type, deep_dive,
                k=k, temperature=temperature, max_tokens=max_tokens,
                reranker_enabled=reranker_enabled, top_n=top_n,
                hyde_enabled=hyde_enabled, bm25_enabled=bm25_enabled,
            )
            answer = chain.invoke(question)
            # Reuse docs already retrieved+reranked inside the standard chain.
            # Deep-dive chains leave the holder empty, so fall back to a
            # separate retrieval call for that path.
            if source_docs_holder:
                source_docs = list(source_docs_holder)
            else:
                raw_source_docs = retriever.invoke(question)
                source_docs = self._rerank_and_expand(
                    question, raw_source_docs,
                    reranker_enabled=reranker_enabled,
                    top_n=top_n,
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
                    stats[module] = {'dev': 0, 'user': 0, 'total': 0}
                if doc_cat in ['dev', 'user']:
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

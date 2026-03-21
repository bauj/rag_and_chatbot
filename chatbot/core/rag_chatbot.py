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

from pathlib import Path
from typing import List, Dict, Optional, Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from .config import ChatbotConfig


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
            all_data = self.vectorstore._collection.get(include=['metadatas'])
            modules = set()
            for meta in all_data['metadatas']:
                if 'module' in meta:
                    modules.add(meta['module'])
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

    def _rerank_and_expand(self, query: str, docs: List) -> List:
        """
        Rerank docs with cross-encoder, expand to section context, deduplicate.

        If no reranker is configured, returns docs unchanged.
        Otherwise:
          1. Scores all (query, page_content) pairs.
          2. Sorts by score descending.
          3. Deduplicates by section_id (keeps highest-scored chunk per section).
          4. Expands each surviving doc to its full section_text if available.
          5. Returns at most config.top_n_after_rerank docs.

        Note: docs with an empty section_id (e.g. old ChromaDB databases without
        section metadata) bypass deduplication — all such docs pass through. This
        is intentional backward-compatibility behaviour.
        """
        if self.reranker is None:
            return docs

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)

        scored = sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)

        result = []
        seen_sections: set = set()

        for _score, doc in scored:
            if len(result) >= self.config.top_n_after_rerank:
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
                     max_tokens: Optional[int] = None):
        """
        Create RAG chain with optional filtering

        Args:
            module_filter: Filter by module (or None)
            doc_type_filter: Filter by doc type (dev, user, or None)
            deep_dive: Use deep dive mode (more docs + summarization)
            k: Override number of chunks to retrieve (optional)
            temperature: Override LLM temperature (optional)
            max_tokens: Override LLM max_tokens (optional)

        Returns:
            Tuple of (chain, retriever)
        """
        # Set k based on priority: runtime override > deep_dive mode > config default
        if k is None:
            k = self.config.k_deep_dive if deep_dive else self.config.k_standard
        search_kwargs = {"k": k}

        if deep_dive and self.reranker is not None:
            print("Note: deep_dive=True — reranking is skipped in deep dive mode.")

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
            # Results are cached on self._last_reranked_docs so ask() can reuse
            # them for source attribution without a second retrieval + reranking pass.
            def retrieve_and_format(question: str) -> str:
                raw_docs = retriever.invoke(question)
                reranked = self._rerank_and_expand(question, raw_docs)
                self._last_reranked_docs = reranked  # cache for source attribution
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

        return chain, retriever

    def ask(self,
            question: str,
            module: Optional[str] = None,
            doc_type: Optional[str] = None,
            deep_dive: bool = False,
            k: Optional[int] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None) -> Dict[str, Any]:
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
        filters = {}
        if module:
            filters['module'] = module
        if doc_type:
            filters['doc_type'] = doc_type
        if deep_dive:
            filters['deep_dive'] = True

        try:
            # Create chain and get answer (pass runtime overrides)
            chain, retriever = self._create_chain(
                module, doc_type, deep_dive,
                k=k, temperature=temperature, max_tokens=max_tokens
            )
            answer = chain.invoke(question)
            # Reuse docs already retrieved+reranked inside the standard chain.
            # Deep-dive chains don't populate the cache, so fall back to a
            # separate retrieval call for that path.
            if not deep_dive and hasattr(self, '_last_reranked_docs'):
                source_docs = self._last_reranked_docs
            else:
                raw_source_docs = retriever.invoke(question)
                source_docs = self._rerank_and_expand(question, raw_source_docs)

            # Format sources
            sources = []
            for doc in source_docs:
                sources.append({
                    'title': doc.metadata.get('title', 'Unknown'),
                    'module': doc.metadata.get('module', 'Unknown'),
                    'doc_category': doc.metadata.get('doc_category', 'Unknown'),
                    'doc_type': doc.metadata.get('doc_type', 'Unknown'),
                    'url': doc.metadata.get('url', ''),
                    'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
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
        all_docs = self.vectorstore._collection.get(include=['metadatas'])

        stats = {}
        for meta in all_docs['metadatas']:
            module = meta.get('module', 'Unknown')
            doc_cat = meta.get('doc_category', 'Unknown')

            if module not in stats:
                stats[module] = {'dev': 0, 'user': 0, 'total': 0}

            if doc_cat in ['dev', 'user']:
                stats[module][doc_cat] += 1
            stats[module]['total'] += 1

        return {
            'total_chunks': len(all_docs['ids']),
            'modules': stats,
            'available_modules': self.available_modules
        }

    def get_available_modules(self) -> List[str]:
        """Get list of available modules"""
        return self.available_modules.copy()

    def get_chunk_count(self) -> int:
        """Get total number of documentation chunks"""
        return self.vectorstore._collection.count()

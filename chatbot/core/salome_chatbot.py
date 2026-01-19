"""
Core business logic for SALOME Documentation RAG Chatbot
Pure logic with no UI concerns - returns data structures only
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from .config import ChatbotConfig


class SALOMEChatbot:
    """
    Core RAG chatbot for SALOME documentation

    Handles vectorstore, LLM, and retrieval logic without UI concerns.
    Returns structured data that UI layers can format as needed.
    """

    def __init__(self, config: ChatbotConfig):
        """
        Initialize chatbot with configuration

        Args:
            config: ChatbotConfig instance

        Raises:
            FileNotFoundError: If ChromaDB not found
            Exception: If LLM connection fails
        """
        self.config = config

        # Set SSL certificate path if provided
        if self.config.ssl_cert_file:
            cert_path = Path(self.config.ssl_cert_file).expanduser().resolve()
            if cert_path.exists():
                os.environ['SSL_CERT_FILE'] = str(cert_path)
            else:
                raise FileNotFoundError(f"SSL certificate file not found: {cert_path}")

        # Load vector database
        self.vectorstore = self._load_vectorstore()

        # Detect available modules
        self.available_modules = self._detect_modules()

        # Initialize LLM
        self.llm = self._initialize_llm()

        # Create base prompt template
        self.base_prompt = self._create_prompt()

    def _load_vectorstore(self) -> Chroma:
        """Load ChromaDB vector store"""
        # Resolve path (handle both absolute and relative paths)
        db_path = Path(self.config.chromadb_path)
        if not db_path.is_absolute():
            # Relative to chatbot directory
            db_path = Path(__file__).parent.parent / self.config.chromadb_path

        db_path = db_path.resolve()

        if not db_path.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at: {db_path}\n"
                f"Run the processor first: cd ../extraction && "
                f"python process_multi_module_salome_docs.py"
            )

        embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model
        )

        vectorstore = Chroma(
            persist_directory=str(db_path),
            embedding_function=embeddings,
            collection_name="salome_documentation"
        )

        return vectorstore

    def _detect_modules(self) -> List[str]:
        """Detect which modules are in the database"""
        try:
            # Get all documents (metadata only - lightweight)
            all_data = self.vectorstore._collection.get(include=['metadatas'])
            modules = set()
            for meta in all_data['metadatas']:
                if 'module' in meta:
                    modules.add(meta['module'])
            return sorted(list(modules))
        except Exception:
            # Fallback to defaults if detection fails
            return ['SHAPER', 'SMESH', 'GUI']

    def _initialize_llm(self, temperature: Optional[float] = None,
                        max_tokens: Optional[int] = None) -> ChatOpenAI:
        """
        Initialize LLM with OpenAI-compatible API

        Args:
            temperature: Override config temperature (optional)
            max_tokens: Override config max_tokens (optional)
        """
        llm = ChatOpenAI(
            model=self.config.model_name,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            streaming=True,
        )
        return llm

    def _create_prompt(self) -> PromptTemplate:
        """Create the base prompt template"""
        template = """You are an expert assistant for SALOME platform documentation.

SALOME modules:
- SHAPER: CAD modeling and geometry creation
- SMESH: Mesh generation and manipulation
- GUI: Graphical user interface components

Each module has:
- Dev docs: API reference (classes, methods, namespaces)
- User docs: Tutorials, guides, usage examples

CRITICAL INSTRUCTIONS:
1. You MUST base your answer ONLY on the documentation provided below
2. DO NOT use any external knowledge or information from your training data
3. If the documentation does not contain enough information to answer the question, explicitly say "I don't have enough information in the documentation to answer this question"
4. ALWAYS cite which module and document type you're referencing
5. If multiple modules are relevant, explain which provides which functionality
6. If the question is not related to SALOME platform, politely decline to answer

Documentation:
{context}

Question: {question}

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
            module_filter: Filter by module (SHAPER, SMESH, GUI, or None)
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
            # Standard chain
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            chain = (
                {
                    "context": retriever | format_docs,
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
            module: Optional module filter (SHAPER, SMESH, GUI)
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
            source_docs = retriever.invoke(question)

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

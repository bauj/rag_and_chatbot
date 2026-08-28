#!/usr/bin/env python3
"""
Unified entry point for Documentation RAG Chatbot
Routes to terminal or web interface based on CLI arguments
"""

import sys
import json
import argparse
from pathlib import Path

from core import ChatbotConfig, DocumentationChatbot, RerankerConfig


def _emit_json(result, stream):
    """
    Write a single ask() result to `stream` as JSON and nothing else.

    Only whitelisted keys are emitted so that a future field on the result dict
    can't accidentally leak something private (e.g. raw config) to a caller that
    is parsing stdout. Sources keep their full page content: consumers such as
    the evaluator judge groundedness against these, and the 200-character
    preview used by the interactive UIs would make correct answers look
    hallucinated.
    """
    payload = {
        "answer": result.get("answer"),
        "error": result.get("error"),
        "filters": result.get("filters", {}),
        "sources": [
            {
                "title": s.get("title"),
                "module": s.get("module"),
                "doc_category": s.get("doc_category"),
                "doc_type": s.get("doc_type"),
                "url": s.get("url"),
                # rag mode's untruncated chunk lives in full_content; agentic
                # mode's parsed page text is already in content.
                "content": s.get("full_content") or s.get("content"),
            }
            for s in result.get("sources", [])
        ],
    }
    json.dump(payload, stream, ensure_ascii=False)
    stream.write("\n")
    stream.flush()


def main():
    """Main entry point with unified CLI"""

    parser = argparse.ArgumentParser(
        description='Documentation RAG Chatbot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chatbot.py
  python chatbot.py --question "How does the API work?"
  python chatbot.py --module MY_MODULE --type dev
  python chatbot.py --web
  python chatbot.py --config my_config.json
        """
    )

    # Interface selection
    parser.add_argument(
        '--web',
        action='store_true',
        help='Launch web interface (default: terminal)'
    )

    # Configuration
    parser.add_argument(
        '--config',
        help='Path to JSON config file (optional)'
    )
    parser.add_argument(
        '--chromadb',
        help='Path to ChromaDB (overrides config)'
    )
    parser.add_argument(
        '--base-url',
        help='LLM API base URL (overrides config)'
    )
    parser.add_argument(
        '--model',
        help='Model name (overrides config)'
    )
    parser.add_argument(
        '--api-key',
        dest='api_key',
        help='LLM API key (overrides config)'
    )
    parser.add_argument(
        '--reranker-model',
        dest='reranker_model',
        help='Reranker model (overrides config; enables reranking even if '
             'disabled in config, rag mode only, terminal single-question mode)'
    )
    parser.add_argument(
        '--reranker-type',
        dest='reranker_type',
        choices=['cross_encoder', 'late_interaction'],
        help='Reranker scoring method: late_interaction (default) or cross_encoder '
             '(ColBERT-style MaxSim via sentence_transformers.MultiVectorEncoder, requires '
             'sentence-transformers >= 6.0) (overrides config; enables reranking even if '
             'disabled in config, rag mode only, terminal single-question mode)'
    )

    # Query options (terminal mode)
    parser.add_argument(
        '--question',
        help='Ask single question and exit (terminal only)'
    )
    parser.add_argument(
        '--json-output',
        action='store_true',
        help='Return answer and sources as JSON (for evaluation mode)'
    )
    parser.add_argument(
        '--module',
        help='Filter by module name (must exist in the database)'
    )
    parser.add_argument(
        '--type',
        dest='doc_type',
        choices=['dev', 'user', 'methodology'],
        help='Filter by doc type (terminal single-question mode)'
    )
    parser.add_argument(
        '--deep-dive',
        action='store_true',
        help='Use deep dive mode (terminal single-question mode)'
    )
    parser.add_argument(
        '--k',
        type=int,
        help='Override number of chunks to retrieve for THIS call regardless of deep_dive '
             '(rag mode only, terminal single-question mode)'
    )
    parser.add_argument(
        '--k-standard',
        type=int,
        dest='k_standard',
        help='Override config.k_standard (the pool size used when deep_dive is off) — unlike '
             '--k, this only takes effect on calls where deep_dive is off, so both --k-standard '
             'and --k-deep-dive can be set at once without conflicting (rag mode only)'
    )
    parser.add_argument(
        '--k-deep-dive',
        type=int,
        dest='k_deep_dive',
        help='Override config.k_deep_dive (the pool size used when deep_dive is on) — see '
             '--k-standard (rag mode only)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        dest='top_n',
        help='Override number of docs kept after reranking (rag mode only, terminal single-question mode)'
    )
    parser.add_argument(
        '--no-rerank',
        action='store_true',
        help='Disable reranking for this query (rag mode only, terminal single-question mode)'
    )
    parser.add_argument(
        '--no-hyde',
        action='store_true',
        help='Disable HyDE for this query, even if enabled in config (rag mode only, terminal single-question mode)'
    )
    parser.add_argument(
        '--no-bm25',
        action='store_true',
        help='Disable BM25 hybrid retrieval for this query, even if enabled in config '
             '(rag mode only, terminal single-question mode)'
    )
    parser.add_argument(
        '--no-title-boost',
        action='store_true',
        help='Disable the title/identifier RRF channel for this query, even if enabled in config '
             '(rag mode only, terminal single-question mode)'
    )
    parser.add_argument(
        '--k-retrieve',
        type=int,
        dest='k_retrieve',
        help='Override per-channel retrieval depth before RRF fusion (rag mode only, ignored in '
             'deep dive mode, terminal single-question mode)'
    )
    parser.add_argument(
        '--deep-dive-batch-size',
        type=int,
        dest='deep_dive_batch_size',
        help='Override deep dive\'s summarization batch size (rag mode only, only applies with '
             '--deep-dive, terminal single-question mode)'
    )
    parser.add_argument(
        '--expansion-char-budget',
        type=int,
        dest='expansion_char_budget',
        help='Override the shared char budget for section expansion (rag mode only, terminal '
             'single-question mode)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        help='Override LLM temperature (terminal single-question mode)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        dest='max_tokens',
        help='Override LLM max_tokens (terminal single-question mode)'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        dest='max_steps',
        help='Override the agent step budget (agentic mode only, terminal single-question mode)'
    )
    parser.add_argument(
        '--max-chars-per-page',
        type=int,
        dest='max_chars_per_page',
        help='Override number of characters read per page (agentic mode only, terminal single-question mode)'
    )

    parser.add_argument(
        '--mode',
        choices=['rag', 'agentic', 'no-rag'],
        default='rag',
        help='Chatbot mode: rag (default), agentic (deepagents harness, '
             'multi-hop page browsing), or no-rag (no retrieval — answers from '
             "the LLM's own training knowledge, a baseline to isolate RAG's "
             'contribution; terminal single-question mode only)'
    )

    # Web options
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Web interface port (default: 7860)'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create public URL via Gradio (web mode only)'
    )

    args = parser.parse_args()

    if args.json_output:
        if args.web:
            parser.error("--json-output cannot be combined with --web")
        if not args.question:
            parser.error("--json-output requires --question")

    if args.mode == 'rag' and (args.max_steps is not None or args.max_chars_per_page is not None):
        print("Warning: --max-steps and --max-chars-per-page are not "
              "supported in rag mode and will be ignored.", file=sys.stderr)

    if args.mode == 'no-rag':
        if args.web:
            parser.error("--mode no-rag is not supported in web mode")
        if not args.question:
            parser.error("--mode no-rag requires --question (terminal single-question mode only)")
        if (args.module or args.doc_type or args.deep_dive or args.k is not None
                or args.top_n is not None or args.no_rerank or args.no_hyde or args.no_bm25
                or args.no_title_boost or args.k_retrieve is not None
                or args.deep_dive_batch_size is not None or args.expansion_char_budget is not None
                or args.reranker_model is not None or args.reranker_type is not None
                or args.k_standard is not None or args.k_deep_dive is not None
                or args.max_steps is not None or args.max_chars_per_page is not None):
            print("Warning: retrieval/reranker/agentic flags are not supported in "
                  "no-rag mode and will be ignored.", file=sys.stderr)

    # In JSON mode stdout must contain the JSON document and nothing else, so
    # point sys.stdout at stderr for the whole run. Every existing print() —
    # including the "DEBUG :" lines emitted while loading the vector store,
    # reranker and LLM — then lands on stderr untouched. The real stdout is
    # kept aside and used only by _emit_json at the very end.
    real_stdout = sys.stdout
    if args.json_output:
        sys.stdout = sys.stderr

    # Load configuration
    try:
        print(f"Initializing {ChatbotConfig.load(args.config).project_name} Documentation Chatbot...")

        # Start with config file or defaults
        config = ChatbotConfig.load(args.config)

        # Override with CLI arguments
        if args.chromadb:
            config.chromadb_path = args.chromadb
        if args.base_url:
            config.llm.base_url = args.base_url
        if args.model:
            config.llm.model = args.model
        if args.api_key:
            config.llm.api_key = args.api_key
        if args.reranker_model or args.reranker_type:
            # Lets a trial opt into reranking purely via CLI even if config.json has
            # it disabled (config.reranker is None) — same spirit as the boolean
            # runtime toggles (--no-rerank etc.) already supported for the rest of
            # the retrieval pipeline.
            if config.reranker is None:
                kwargs = {}
                if args.reranker_model:
                    kwargs['model'] = args.reranker_model
                if args.reranker_type:
                    kwargs['type'] = args.reranker_type
                config.reranker = RerankerConfig(**kwargs)
            else:
                if args.reranker_model:
                    config.reranker.model = args.reranker_model
                if args.reranker_type:
                    config.reranker.type = args.reranker_type
        if args.k_standard is not None:
            config.k_standard = args.k_standard
        if args.k_deep_dive is not None:
            config.k_deep_dive = args.k_deep_dive

        # Initialize core chatbot. no-rag mode never touches the reranker, so
        # skip loading it there — pure per-question subprocess startup cost
        # otherwise. Agentic mode DOES use it now: its search_sections_tool
        # reranks and reconstructs sections via chatbot.expand_sections
        # (loaded once here, reused per search call).
        print("Loading documentation database...")
        chatbot = DocumentationChatbot(config, skip_reranker=(args.mode == 'no-rag'))
        reranker_score_fn = chatbot.score_against_query if chatbot.reranker is not None else None

        print(f"  Loaded {chatbot.get_chunk_count()} documentation chunks")
        print(f"  Modules: {', '.join(chatbot.available_modules)}")
        print(f"Using model: {config.llm.model}")
        print(f"  Endpoint: {config.llm.base_url}")
        print("Chatbot ready!\n")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("  1. ChromaDB exists (run: python process_docs.py --config config.json)")
        print("  2. Path is correct in config or --chromadb argument")
        sys.exit(1)
    except Exception as e:
        print(f"\nFailed to initialize: {e}")
        print("\nMake sure:")
        print("  1. ChromaDB exists and is accessible")
        print("  2. LLM endpoint is reachable at the configured base_url")
        print("  3. Config file is valid JSON (if using --config)")
        sys.exit(1)

    # Initialize agentic chatbot if requested (or auto-enabled by a config
    # 'agentic' block).
    agentic_chatbot = None
    if args.mode == 'agentic' or config.agentic is not None:
        if config.agentic is None:
            print(f"\nError: --mode {args.mode} requires an 'agentic' block in config.json")
            sys.exit(1)
        # Warn about ignored flags in agentic mode
        if args.mode == 'agentic':
            if args.module or args.doc_type:
                print("Warning: --module and --type are not supported in agentic mode and will be ignored.",
                      file=sys.stderr)
            if args.deep_dive:
                print("Warning: --deep-dive is not supported in agentic mode and will be ignored.",
                      file=sys.stderr)
            if (args.k is not None or args.top_n is not None or args.no_rerank
                    or args.no_hyde or args.no_bm25 or args.no_title_boost
                    or args.k_retrieve is not None or args.deep_dive_batch_size is not None
                    or args.expansion_char_budget is not None or args.reranker_model is not None
                    or args.reranker_type is not None
                    or args.k_standard is not None or args.k_deep_dive is not None):
                print("Warning: --k, --k-standard, --k-deep-dive, --top-n, --no-rerank, --no-hyde, "
                      "--no-bm25, --no-title-boost, --k-retrieve, --deep-dive-batch-size, "
                      "--expansion-char-budget, --reranker-model and --reranker-type are not "
                      "supported in agentic mode and will be ignored.", file=sys.stderr)
        try:
            from core import AgenticChatbot
            print("Loading agentic chatbot (deepagents)...")
            agentic_chatbot = AgenticChatbot(config, vectorstore=chatbot.vectorstore, bm25_index=chatbot.bm25_index,
                                              reranker_score_fn=reranker_score_fn,
                                              section_select_fn=chatbot.expand_sections)
            print("Agentic chatbot ready.")
        except ImportError as e:
            if args.mode == 'agentic':
                print(f"\nError: --mode agentic requires the deepagents package. {e}")
                sys.exit(1)
            print(f"Warning: agentic mode unavailable ({e}). Continuing without it.", file=sys.stderr)

    # Route to appropriate interface
    if args.web:
        from ui import WebUI
        # Web interface
        web_ui = WebUI(chatbot, agentic_chatbot=agentic_chatbot)
        web_ui.launch(share=args.share, port=args.port)
    else:
        from ui import TerminalUI
        # Terminal interface
        terminal_ui = TerminalUI(chatbot, agentic_chatbot=agentic_chatbot)

        if args.question:
            if args.mode == 'agentic':
                result = agentic_chatbot.ask(
                    args.question,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    max_steps=args.max_steps,
                    max_chars_per_page=args.max_chars_per_page,
                )
            elif args.mode == 'no-rag':
                result = chatbot.ask_no_rag(
                    args.question,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                if not args.json_output:
                    print(f"Question: {args.question}\n")
                    print(f"Answer: {result['answer']}\n" if result['error'] is None
                          else f"Error: {result['error']}\n")
            elif args.json_output:
                # run_single_question() only prints, so ask directly to get the dict.
                result = chatbot.ask(
                    args.question,
                    module=args.module,
                    doc_type=args.doc_type,
                    deep_dive=args.deep_dive,
                    k=args.k,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    reranker_enabled=not args.no_rerank,
                    top_n=args.top_n,
                    hyde_enabled=not args.no_hyde,
                    bm25_enabled=not args.no_bm25,
                    title_boost_enabled=not args.no_title_boost,
                    k_retrieve=args.k_retrieve,
                    deep_dive_batch_size=args.deep_dive_batch_size,
                    expansion_char_budget=args.expansion_char_budget,
                )
            else:
                # Single question mode
                terminal_ui.run_single_question(
                    args.question,
                    module=args.module,
                    doc_type=args.doc_type,
                    deep_dive=args.deep_dive,
                    k=args.k,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    reranker_enabled=not args.no_rerank,
                    top_n=args.top_n,
                    hyde_enabled=not args.no_hyde,
                    bm25_enabled=not args.no_bm25,
                    title_boost_enabled=not args.no_title_boost,
                    k_retrieve=args.k_retrieve,
                    deep_dive_batch_size=args.deep_dive_batch_size,
                    expansion_char_budget=args.expansion_char_budget,
                )
                result = None

            if args.json_output:
                # A result carrying an "error" is still a valid document, so exit 0
                # and let the caller read the payload. Non-zero is reserved for the
                # setup failures handled above (missing ChromaDB, bad config, ...).
                _emit_json(result, real_stdout)
            elif result is not None:
                print(result['answer'] or result['error'])
        else:
            # Interactive mode
            terminal_ui.run_interactive(initial_mode=args.mode)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Unified entry point for Documentation RAG Chatbot
Routes to terminal or web interface based on CLI arguments
"""

import sys
import json
import argparse
from pathlib import Path

from core import ChatbotConfig, DocumentationChatbot


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
                # Prefer the untruncated chunk; fall back to the preview for the
                # agentic modes, whose sources carry no chunk text at all.
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
        '--json-output',
        action='store_true',
        help='With --question, print the result as a single JSON document on stdout '
             '(answer, error, filters, sources with full content). All progress and '
             'debug output goes to stderr, so stdout is machine-parseable.'
    )

    parser.add_argument(
        '--mode',
        choices=['rag', 'agentic', 'agentic-smol'],
        default='rag',
        help='Chatbot mode: rag (default), agentic (reads HTML pages directly), '
             'or agentic-smol (smolagents CodeAgent, real multi-hop browsing)'
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

        # Initialize core chatbot
        print("Loading documentation database...")
        chatbot = DocumentationChatbot(config)

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

    # Initialize agentic chatbot if requested
    agentic_chatbot = None
    if args.mode in ('agentic', 'agentic-smol') or config.agentic is not None:
        if config.agentic is None:
            print(f"\nError: --mode {args.mode} requires an 'agentic' block in config.json")
            sys.exit(1)
        # Warn about ignored flags in agentic modes
        if args.mode in ('agentic', 'agentic-smol'):
            if args.module or args.doc_type:
                print(f"Warning: --module and --type are not supported in {args.mode} mode and will be ignored.",
                      file=sys.stderr)
            if args.deep_dive:
                print(f"Warning: --deep-dive is not supported in {args.mode} mode and will be ignored.",
                      file=sys.stderr)
        from core import AgenticChatbot
        print("Loading agentic chatbot (page index)...")
        agentic_chatbot = AgenticChatbot(config)
        print("Agentic chatbot ready.")

    # Initialize smolagents agentic chatbot if enabled (opt-in, requires config.agentic too)
    agentic_smol_chatbot = None
    if config.agentic is not None and config.smol_enabled:
        try:
            from core import AgenticSmolChatbot
            print("Loading agentic-smol chatbot (smolagents)...")
            agentic_smol_chatbot = AgenticSmolChatbot(config)
            print("Agentic-smol chatbot ready.")
        except ImportError as e:
            if args.mode == 'agentic-smol':
                print(f"\nError: --mode agentic-smol requires the smolagents package. {e}")
                sys.exit(1)
            print(f"Warning: agentic-smol mode unavailable ({e}). Continuing without it.", file=sys.stderr)
    elif args.mode == 'agentic-smol':
        if config.agentic is None:
            print("\nError: --mode agentic-smol requires an 'agentic' block in config.json")
        else:
            print("\nError: --mode agentic-smol requires smol_enabled: true in config.json")
        sys.exit(1)

    # Route to appropriate interface
    if args.web:
        from ui import WebUI
        # Web interface
        web_ui = WebUI(chatbot, agentic_chatbot=agentic_chatbot, agentic_smol_chatbot=agentic_smol_chatbot)
        web_ui.launch(share=args.share, port=args.port)
    else:
        from ui import TerminalUI
        # Terminal interface
        terminal_ui = TerminalUI(chatbot, agentic_chatbot=agentic_chatbot, agentic_smol_chatbot=agentic_smol_chatbot)

        if args.question:
            if args.mode == 'agentic':
                result = agentic_chatbot.ask(args.question)
            elif args.mode == 'agentic-smol':
                result = agentic_smol_chatbot.ask(args.question)
            elif args.json_output:
                # run_single_question() only prints, so ask directly to get the dict.
                result = chatbot.ask(
                    args.question,
                    module=args.module,
                    doc_type=args.doc_type,
                    deep_dive=args.deep_dive,
                )
            else:
                # Single question mode
                terminal_ui.run_single_question(
                    args.question,
                    module=args.module,
                    doc_type=args.doc_type,
                    deep_dive=args.deep_dive
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
            terminal_ui.run_interactive()


if __name__ == "__main__":
    main()

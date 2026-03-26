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
        choices=['dev', 'user'],
        help='Filter by doc type (terminal single-question mode)'
    )
    parser.add_argument(
        '--deep-dive',
        action='store_true',
        help='Use deep dive mode (terminal single-question mode)'
    )

    parser.add_argument(
        '--mode',
        choices=['rag', 'agentic'],
        default='rag',
        help='Chatbot mode: rag (default) or agentic (reads HTML pages directly)'
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
    if args.mode == 'agentic' or config.agentic is not None:
        if config.agentic is None:
            print("\nError: --mode agentic requires an 'agentic' block in config.json")
            sys.exit(1)
        # Warn about ignored flags in agentic mode
        if args.mode == 'agentic':
            if args.module or args.doc_type:
                print("Warning: --module and --type are not supported in agentic mode and will be ignored.",
                      file=sys.stderr)
            if args.deep_dive:
                print("Warning: --deep-dive is not supported in agentic mode and will be ignored.",
                      file=sys.stderr)
        from core import AgenticChatbot
        print("Loading agentic chatbot (page index)...")
        agentic_chatbot = AgenticChatbot(config)
        print("Agentic chatbot ready.")

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
                result = agentic_chatbot.ask(args.question)
                if args.json_output:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    print(result['answer'] or result['error'])
            else:
                # Single question mode
                if args.json_output:
                    # JSON output for evaluation
                    result = chatbot.ask(
                        args.question,
                        module=args.module,
                        doc_type=args.doc_type,
                        deep_dive=args.deep_dive
                    )
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    # Normal terminal output
                    terminal_ui.run_single_question(
                        args.question,
                        module=args.module,
                        doc_type=args.doc_type,
                        deep_dive=args.deep_dive
                    )

        else:
            # Interactive mode
            terminal_ui.run_interactive()


if __name__ == "__main__":
    main()

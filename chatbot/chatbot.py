#!/usr/bin/env python3
"""
Unified entry point for SALOME Documentation Chatbot
Routes to terminal or web interface based on CLI arguments
"""

import sys
import argparse
from pathlib import Path

from core import ChatbotConfig, SALOMEChatbot
from ui import TerminalUI, WebUI


def main():
    """Main entry point with unified CLI"""

    parser = argparse.ArgumentParser(
        description='SALOME Multi-Module Documentation Chatbot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive terminal (default)
  python chatbot.py

  # Single question in terminal
  python chatbot.py --question "What is ModelAPI::Feature?"

  # Filter by module and doc type
  python chatbot.py --module SHAPER --type dev

  # Use deep dive mode
  python chatbot.py --question "Explain mesh generation" --deep-dive

  # Launch web interface
  python chatbot.py --web

  # Web interface on custom port
  python chatbot.py --web --port 8080

  # Use custom config file
  python chatbot.py --config my_config.json

  # Override config with CLI args
  python chatbot.py --config prod.json --model mistral-mini
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
        '--litellm-url',
        help='LiteLLM URL (overrides config)'
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
        '--module',
        choices=['SHAPER', 'SMESH', 'GUI'],
        help='Filter by module (terminal single-question mode)'
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
        print("Initializing SALOME Multi-Module Chatbot...")

        # Start with config file or defaults
        config = ChatbotConfig.load(args.config)

        # Override with CLI arguments
        if args.chromadb:
            config.chromadb_path = args.chromadb
        if args.litellm_url:
            config.litellm_url = args.litellm_url
        if args.model:
            config.model_name = args.model

        # Initialize core chatbot
        print("Loading documentation database...")
        chatbot = SALOMEChatbot(config)

        print(f"  Loaded {chatbot.get_chunk_count()} documentation chunks")
        print(f"  Modules: {', '.join(chatbot.available_modules)}")
        print(f"Initializing {config.model_name} via LiteLLM...")
        print(f"  Connected to {config.model_name}")
        print("Chatbot ready!\n")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("  1. ChromaDB exists (run: cd ../extraction && python process_multi_module_salome_docs.py)")
        print("  2. Path is correct in config or --chromadb argument")
        sys.exit(1)
    except Exception as e:
        print(f"\nFailed to initialize: {e}")
        print("\nMake sure:")
        print("  1. ChromaDB exists and is accessible")
        print("  2. LiteLLM is running: curl http://localhost:8080/v1/models")
        print("  3. Config file is valid JSON (if using --config)")
        sys.exit(1)

    # Route to appropriate interface
    if args.web:
        # Web interface
        web_ui = WebUI(chatbot)
        web_ui.launch(share=args.share, port=args.port)
    else:
        # Terminal interface
        terminal_ui = TerminalUI(chatbot)

        if args.question:
            # Single question mode
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

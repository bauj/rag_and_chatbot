"""
Terminal interface for documentation chatbot
Handles user interaction, formatting, and command parsing
"""

from typing import Optional
import sys

# Handle both package and direct script execution
try:
    from ..core import DocumentationChatbot
except ImportError:
    from core import DocumentationChatbot


class TerminalUI:
    """Interactive terminal interface for documentation chatbot"""

    def __init__(self, chatbot: DocumentationChatbot, agentic_chatbot=None):
        self.chatbot = chatbot
        self.agentic_chatbot = agentic_chatbot

    def _print_header(self):
        """Print welcome header"""
        project = self.chatbot.config.project_name
        has_agentic = self.agentic_chatbot is not None
        has_reranker = self.chatbot.reranker is not None
        print("=" * 70)
        print(f"{project} Documentation Chatbot")
        print("=" * 70)
        print(f"Modules: {', '.join(self.chatbot.available_modules)}")
        print("\nCommands:")
        if has_agentic:
            print("  mode:rag          - Switch to RAG mode (vector retrieval)")
            print("  mode:agentic      - Switch to Agentic mode (reads HTML pages directly)")
        for mod in self.chatbot.available_modules:
            print(f"  module:{mod:<12} - Filter by {mod} (RAG only)")
        print("  type:dev          - Filter developer docs only (RAG only)")
        print("  type:user         - Filter user docs only (RAG only)")
        print("  type:methodology  - Filter methodology docs only (RAG only)")
        print("  deep              - Toggle Deep Dive mode (RAG only)")
        if has_reranker:
            print("  reranker          - Toggle cross-encoder reranker on/off (RAG only)")
        print("  clear             - Clear all filters")
        print("  stats             - Show statistics (RAG only)")
        print("  exit/quit         - Exit\n")

    def _print_answer(self, result: dict):
        """
        Format and print answer with sources

        Args:
            result: Result dict from chatbot.ask()
        """
        if result.get('error'):
            print(f"Error: {result['error']}\n")
            return

        answer = result.get('answer')
        sources = result.get('sources', [])
        filters = result.get('filters', {})

        # Show active filters
        if filters:
            filter_strs = []
            if 'module' in filters:
                filter_strs.append(f"module={filters['module']}")
            if 'doc_type' in filters:
                filter_strs.append(f"type={filters['doc_type']}")
            if filters.get('deep_dive'):
                filter_strs.append("DEEP DIVE")
            if filter_strs:
                print(f"Filters: {', '.join(filter_strs)}")

        # Print answer
        print(f"\nAnswer:\n{answer}\n")

        # Show sources
        if sources:
            print("Sources:")
            modules_used = set()
            doc_types_used = set()

            for i, source in enumerate(sources[:5], 1):
                title = source.get('title', 'Unknown')
                url = source.get("url", "")
                mod = source.get('module', 'Unknown')
                dtype = source.get('doc_category', 'Unknown')
                modules_used.add(mod)
                doc_types_used.add(dtype)
                print(f"   {i}. [{mod}/{dtype}] [{title}]({url})")

            if len(modules_used) > 1 or len(doc_types_used) > 1:
                info = []
                if len(modules_used) > 1:
                    info.append(f"{len(modules_used)} modules: {', '.join(sorted(modules_used))}")
                if len(doc_types_used) > 1:
                    info.append(f"dev + user + methodology docs")
                print(f"\n   ℹ️  Answer uses {', '.join(info)}")
            print()

    def _print_stats(self):
        """Print database statistics"""
        stats = self.chatbot.get_stats()

        print("\n" + "="*70)
        print(f"{self.chatbot.config.project_name} Documentation Statistics")
        print("="*70)

        print(f"\nTotal chunks: {stats['total_chunks']}")
        print("\nBy module:")

        for module in sorted(stats['modules'].keys()):
            mod_stats = stats['modules'][module]
            print(f"  {module}:")
            print(f"    Dev:  {mod_stats['dev']:5} chunks")
            print(f"    User: {mod_stats['user']:5} chunks")
            print(f"    Methodology: {mod_stats['methodology']:5} chunks")
            print(f"    Total: {mod_stats['total']:5} chunks")

        print("="*70 + "\n")

    def run_interactive(self):
        """Run interactive chat session"""
        self._print_header()

        current_mode = "rag"        # "rag" or "agentic"
        current_module = None
        current_type = None
        deep_dive_mode = False
        reranker_enabled = self.chatbot.reranker is not None

        while True:
            try:
                # Build prompt
                prompt_parts = ["You"]
                if current_mode == "agentic":
                    prompt_parts.append("[AGENTIC]")
                else:
                    if current_module:
                        prompt_parts.append(f"[{current_module}]")
                    if current_type:
                        prompt_parts.append(f"[{current_type}]")
                    if deep_dive_mode:
                        prompt_parts.append("[DEEP DIVE]")
                    if not reranker_enabled:
                        prompt_parts.append("[no reranker]")
                    if not current_module and not current_type and not deep_dive_mode:
                        prompt_parts.append("[all]")

                prompt = " ".join(prompt_parts) + ": "
                user_input = input(prompt).strip()

                # Handle exit
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    break

                # Handle empty
                if not user_input:
                    continue

                # Handle mode switch
                if user_input.lower().startswith('mode:'):
                    requested = user_input.split(':', 1)[1].strip().lower()
                    if requested == 'agentic':
                        if self.agentic_chatbot is None:
                            print("Warning: Agentic mode is not configured (add 'agentic' block to config.json)\n")
                        else:
                            current_mode = "agentic"
                            print("OK: Switched to Agentic mode\n")
                    elif requested == 'rag':
                        current_mode = "rag"
                        print("OK: Switched to RAG mode\n")
                    else:
                        print("Warning: Unknown mode (use: rag or agentic)\n")
                    continue

                # Handle reranker toggle (RAG only)
                if user_input.lower() == 'reranker':
                    if self.chatbot.reranker is None:
                        print("Warning: No reranker model is configured\n")
                    else:
                        reranker_enabled = not reranker_enabled
                        status = "enabled" if reranker_enabled else "disabled"
                        print(f"OK: Reranker {status}\n")
                    continue

                # RAG-only commands
                if current_mode == "rag":
                    # Handle stats
                    if user_input.lower() == 'stats':
                        self._print_stats()
                        continue

                    # Handle deep dive toggle
                    if user_input.lower() == 'deep':
                        deep_dive_mode = not deep_dive_mode
                        status = "enabled" if deep_dive_mode else "disabled"
                        print(f"OK: Deep Dive mode {status}\n")
                        continue

                    # Handle clear
                    if user_input.lower() == 'clear':
                        current_module = None
                        current_type = None
                        deep_dive_mode = False
                        print("OK: Cleared all filters\n")
                        continue

                    # Handle module filter
                    if user_input.lower().startswith('module:'):
                        module = user_input.split(':', 1)[1].strip().upper()
                        if module in self.chatbot.available_modules:
                            current_module = module
                            print(f"OK: Module filter: {module}\n")
                        else:
                            print(f"Warning: Unknown module: {module}")
                            print(f"Available: {', '.join(self.chatbot.available_modules)}\n")
                        continue

                    # Handle type filter
                    if user_input.lower().startswith('type:'):
                        dtype = user_input.split(':', 1)[1].strip().lower()
                        if dtype in ['dev', 'user', 'methodology']:
                            current_type = dtype
                            print(f"OK: Doc type filter: {dtype}\n")
                        else:
                            print("Warning: Unknown type (use: dev or user or methodology)\n")
                        continue

                # Ask question
                if current_mode == "agentic" and self.agentic_chatbot is not None:
                    result = self.agentic_chatbot.ask(user_input)
                else:
                    result = self.chatbot.ask(
                        user_input,
                        module=current_module,
                        doc_type=current_type,
                        deep_dive=deep_dive_mode,
                        reranker_enabled=reranker_enabled,
                    )
                self._print_answer(result)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")

    def run_single_question(self,
                           question: str,
                           module: Optional[str] = None,
                           doc_type: Optional[str] = None,
                           deep_dive: bool = False):
        """
        Run single question and exit

        Args:
            question: Question to ask
            module: Optional module filter
            doc_type: Optional doc type filter
            deep_dive: Use deep dive mode
        """
        print(f"Question: {question}\n")
        result = self.chatbot.ask(question, module, doc_type, deep_dive)
        self._print_answer(result)

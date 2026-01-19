"""
Terminal interface for SALOME Documentation Chatbot
Handles user interaction, formatting, and command parsing
"""

from typing import Optional
import sys

# Handle both package and direct script execution
try:
    from ..core import SALOMEChatbot
except ImportError:
    from core import SALOMEChatbot


class TerminalUI:
    """Interactive terminal interface for SALOME chatbot"""

    def __init__(self, chatbot: SALOMEChatbot):
        """
        Initialize terminal UI

        Args:
            chatbot: SALOMEChatbot instance
        """
        self.chatbot = chatbot

    def _print_header(self):
        """Print welcome header"""
        print("=" * 70)
        print("SALOME Multi-Module Documentation Chatbot")
        print("=" * 70)
        print(f"Modules: {', '.join(self.chatbot.available_modules)}")
        print("\nCommands:")
        print("  module:SHAPER     - Filter by SHAPER")
        print("  module:SMESH      - Filter by SMESH")
        print("  module:GUI        - Filter by GUI")
        print("  type:dev          - Filter developer docs only")
        print("  type:user         - Filter user docs only")
        print("  deep              - Toggle DEEP DIVE mode (comprehensive analysis)")
        print("  clear             - Clear all filters")
        print("  stats             - Show statistics")
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
            print(f"Filters: Filters: {', '.join(filter_strs)}")

        # Print answer
        print(f"\nAnswer:\n{answer}\n")

        # Show sources
        if sources:
            print("Sources: Sources:")
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
                    info.append(f"dev + user docs")
                print(f"\n   ℹ️  Answer uses {', '.join(info)}")
            print()

    def _print_stats(self):
        """Print database statistics"""
        stats = self.chatbot.get_stats()

        print("\n" + "="*70)
        print("SALOME Documentation Statistics")
        print("="*70)

        print(f"\nTotal chunks: {stats['total_chunks']}")
        print("\nBy module:")

        for module in sorted(stats['modules'].keys()):
            mod_stats = stats['modules'][module]
            print(f"  {module}:")
            print(f"    Dev:  {mod_stats['dev']:5} chunks")
            print(f"    User: {mod_stats['user']:5} chunks")
            print(f"    Total: {mod_stats['total']:5} chunks")

        print("="*70 + "\n")

    def run_interactive(self):
        """Run interactive chat session"""
        self._print_header()

        current_module = None
        current_type = None
        deep_dive_mode = False

        while True:
            try:
                # Build prompt with filters
                prompt_parts = ["You"]
                if current_module:
                    prompt_parts.append(f"[{current_module}]")
                if current_type:
                    prompt_parts.append(f"[{current_type}]")
                if deep_dive_mode:
                    prompt_parts.append("[DEEP DIVE]")
                if not current_module and not current_type and not deep_dive_mode:
                    prompt_parts.append("[all]")

                prompt = " ".join(prompt_parts) + ": "
                user_input = input(prompt).strip()

                # Handle exit
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye !")
                    break

                # Handle empty
                if not user_input:
                    continue

                # Handle stats
                if user_input.lower() == 'stats':
                    self._print_stats()
                    continue

                # Handle deep dive toggle
                if user_input.lower() == 'deep':
                    deep_dive_mode = not deep_dive_mode
                    status = "enabled" if deep_dive_mode else "disabled"
                    print(f" DEEP DIVE mode {status}\n")
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
                    if dtype in ['dev', 'user']:
                        current_type = dtype
                        print(f"OK: Doc type filter: {dtype}\n")
                    else:
                        print("Warning: Unknown type (use: dev or user)\n")
                    continue

                # Ask question
                result = self.chatbot.ask(
                    user_input,
                    module=current_module,
                    doc_type=current_type,
                    deep_dive=deep_dive_mode
                )
                self._print_answer(result)

            except KeyboardInterrupt:
                print("\n\nGoodbye! Goodbye!")
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

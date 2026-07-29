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

    def __init__(self, chatbot: DocumentationChatbot, agentic_chatbot=None, agentic_smol_chatbot=None):
        self.chatbot = chatbot
        self.agentic_chatbot = agentic_chatbot
        self.agentic_smol_chatbot = agentic_smol_chatbot

    def _print_header(self):
        """Print welcome header"""
        project = self.chatbot.config.project_name
        has_agentic = self.agentic_chatbot is not None
        has_agentic_smol = self.agentic_smol_chatbot is not None
        has_reranker = self.chatbot.reranker is not None
        has_hyde = self.chatbot.hyde_llm is not None
        has_bm25 = self.chatbot.bm25_index is not None
        # No dedicated object gates this one (the title index lives inside bm25_index
        # regardless of the flag) — availability is config.title_boost_enabled plus an
        # index actually being loaded to search titles against.
        has_title_boost = self.chatbot.config.title_boost_enabled and has_bm25
        print("=" * 70)
        print(f"{project} Documentation Chatbot")
        print("=" * 70)
        print(f"Modules: {', '.join(self.chatbot.available_modules)}")

        # Show which retrieval stages are actually loaded — otherwise BM25/HyDE being
        # off (or silently unavailable) is invisible until you read the config.
        stages = []
        stages.append(f"BM25 hybrid: {'on' if has_bm25 else 'off'}")
        stages.append(f"title boost: {'on' if has_title_boost else 'off'}")
        stages.append(f"HyDE: {'on' if has_hyde else 'off'}")
        stages.append(f"reranker: {'on' if has_reranker else 'off'}")
        print(f"Retrieval:  {' · '.join(stages)}")
        print("\nCommands:")
        if has_agentic:
            print("  mode:rag          - Switch to RAG mode (vector retrieval)")
            print("  mode:agentic      - Switch to Agentic mode (reads HTML pages directly)")
        if has_agentic_smol:
            print("  mode:agentic-smol - Switch to Agentic-smol mode (smolagents, multi-hop browsing)")
        for mod in self.chatbot.available_modules:
            print(f"  module:{mod:<12} - Filter by {mod} (RAG only)")
        print("  type:dev          - Filter developer docs only (RAG only)")
        print("  type:user         - Filter user docs only (RAG only)")
        print("  deep              - Toggle Deep Dive mode (RAG only)")
        if has_reranker:
            print("  reranker          - Toggle cross-encoder reranker on/off (RAG only)")
            print("  topn:<n>          - Set top-N docs kept after rerank (RAG only)")
        if has_hyde:
            print("  hyde              - Toggle HyDE on/off (RAG only)")
        if has_bm25:
            print("  bm25              - Toggle BM25 hybrid retrieval on/off (RAG only)")
        if has_agentic:
            print("  agentic:chars:<n>  - Set max chars read per page (classic Agentic only)")
            print("  agentic:pages1:<n> - Set pages read in round 1 (classic Agentic only)")
            print("  agentic:pages2:<n> - Set pages read in round 2 (classic Agentic only)")
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

            # Report which retrieval stages actually ran, not which ones are toggled on.
            bypassed = filters.get('bypassed_by_deep_dive')
            if bypassed:
                print(f"Note: {', '.join(bypassed)} skipped — Deep Dive uses its own retrieval path")
            else:
                stages = [name for key, name in
                          (('bm25', 'BM25 hybrid'), ('title_boost', 'title boost'),
                           ('hyde', 'HyDE'), ('reranker', 'reranker'))
                          if filters.get(key)]
                if stages:
                    print(f"Retrieval: {' + '.join(stages)}")

            if filters.get('rounds_used') is not None:
                print(f"Agentic rounds used: {filters['rounds_used']}")
            if filters.get('steps_used') is not None:
                print(f"Agent steps used: {filters['steps_used']}")

        # agentic-smol reports grounded=False when it answered without reading any page,
        # i.e. from the model's own knowledge rather than the documentation.
        if filters.get('grounded') is False:
            print("\n⚠️  Warning: answer produced without reading any documentation page — "
                  "it may come from the model's own knowledge rather than the docs.")

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
                    info.append(f"dev + user docs")
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
        hyde_enabled = self.chatbot.hyde_llm is not None
        bm25_enabled = self.chatbot.bm25_index is not None
        # No runtime toggle for this one (see chatbot.ask()'s title_boost_enabled
        # param) — it is config-only for now, so "active" just mirrors availability.
        title_boost_active = self.chatbot.config.title_boost_enabled and self.chatbot.bm25_index is not None
        top_n_override = None
        agentic_chars_override = None
        agentic_pages1_override = None
        agentic_pages2_override = None

        while True:
            try:
                # Build prompt
                prompt_parts = ["You"]
                if current_mode == "agentic":
                    prompt_parts.append("[AGENTIC]")
                elif current_mode == "agentic-smol":
                    prompt_parts.append("[AGENTIC-SMOL]")
                else:
                    if current_module:
                        prompt_parts.append(f"[{current_module}]")
                    if current_type:
                        prompt_parts.append(f"[{current_type}]")
                    if deep_dive_mode:
                        prompt_parts.append("[DEEP DIVE]")
                    if not reranker_enabled:
                        prompt_parts.append("[no reranker]")
                    elif top_n_override is not None and top_n_override != self.chatbot.config.top_n_after_rerank:
                        prompt_parts.append(f"[top_n={top_n_override}]")
                    if hyde_enabled:
                        prompt_parts.append("[hyde]")
                    if bm25_enabled:
                        prompt_parts.append("[bm25]")
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
                    elif requested == 'agentic-smol':
                        if self.agentic_smol_chatbot is None:
                            print("Warning: Agentic-smol mode is not configured (set smol_enabled: true in config.json)\n")
                        else:
                            current_mode = "agentic-smol"
                            print("OK: Switched to Agentic-smol mode\n")
                    elif requested == 'rag':
                        current_mode = "rag"
                        print("OK: Switched to RAG mode\n")
                    else:
                        print("Warning: Unknown mode (use: rag, agentic, or agentic-smol)\n")
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

                # Handle HyDE toggle (RAG only)
                if user_input.lower() == 'hyde':
                    if self.chatbot.hyde_llm is None:
                        print("Warning: HyDE is not configured (set hyde_enabled: true in config.json)\n")
                    else:
                        hyde_enabled = not hyde_enabled
                        status = "enabled" if hyde_enabled else "disabled"
                        print(f"OK: HyDE {status}\n")
                    continue

                # Handle BM25 toggle (RAG only)
                if user_input.lower() == 'bm25':
                    if self.chatbot.bm25_index is None:
                        print("Warning: BM25 is not available (needs bm25_enabled: true in config.json "
                              "and the extraction JSONL next to chromadb_path)\n")
                    else:
                        bm25_enabled = not bm25_enabled
                        status = "enabled" if bm25_enabled else "disabled"
                        print(f"OK: BM25 hybrid retrieval {status}\n")
                    continue

                # Handle top_n override (RAG only)
                if user_input.lower().startswith('topn:'):
                    if self.chatbot.reranker is None:
                        print("Warning: No reranker model is configured\n")
                    else:
                        raw = user_input.split(':', 1)[1].strip()
                        if raw.isdigit() and int(raw) > 0:
                            top_n_override = int(raw)
                            print(f"OK: top_n set to {top_n_override}\n")
                        else:
                            print("Warning: top_n must be a positive integer\n")
                    continue

                # Handle agentic param overrides (Agentic only)
                if user_input.lower().startswith('agentic:'):
                    if self.agentic_chatbot is None:
                        print("Warning: Agentic mode is not configured (add 'agentic' block to config.json)\n")
                        continue
                    if current_mode == "agentic-smol":
                        print("Warning: agentic:* params apply to classic Agentic mode only, not agentic-smol\n")
                        continue
                    parts = user_input.split(':')
                    if len(parts) == 3 and parts[1].lower() in ('chars', 'pages1', 'pages2'):
                        param, raw = parts[1].lower(), parts[2].strip()
                        if raw.isdigit() and int(raw) > 0:
                            value = int(raw)
                            if param == 'chars':
                                agentic_chars_override = value
                            elif param == 'pages1':
                                agentic_pages1_override = value
                            else:
                                agentic_pages2_override = value
                            print(f"OK: agentic:{param} set to {value}\n")
                        else:
                            print(f"Warning: agentic:{param} must be a positive integer\n")
                    else:
                        print("Warning: Unknown agentic setting (use: agentic:chars:<n>, agentic:pages1:<n>, agentic:pages2:<n>)\n")
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
                        print(f"OK: Deep Dive mode {status}")
                        if deep_dive_mode:
                            # Deep dive runs its own retrieve-and-summarize path — say which
                            # currently-active stages it will bypass instead of silently dropping them.
                            skipped = [name for active, name in
                                       ((bm25_enabled, "BM25 hybrid retrieval"),
                                        (title_boost_active, "title/identifier boost"),
                                        (hyde_enabled, "HyDE"),
                                        (reranker_enabled, "reranking"))
                                       if active]
                            if skipped:
                                print(f"Note: {', '.join(skipped)} will be skipped while Deep Dive is on")
                        print()
                        continue

                    # Handle clear
                    if user_input.lower() == 'clear':
                        current_module = None
                        current_type = None
                        deep_dive_mode = False
                        top_n_override = None
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
                if current_mode == "agentic" and self.agentic_chatbot is not None:
                    result = self.agentic_chatbot.ask(
                        user_input,
                        max_chars_per_page=agentic_chars_override,
                        max_pages_per_round=agentic_pages1_override,
                        max_pages_round2=agentic_pages2_override,
                    )
                elif current_mode == "agentic-smol" and self.agentic_smol_chatbot is not None:
                    result = self.agentic_smol_chatbot.ask(user_input)
                else:
                    result = self.chatbot.ask(
                        user_input,
                        module=current_module,
                        doc_type=current_type,
                        deep_dive=deep_dive_mode,
                        reranker_enabled=reranker_enabled,
                        top_n=top_n_override,
                        hyde_enabled=hyde_enabled,
                        bm25_enabled=bm25_enabled,
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

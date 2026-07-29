"""
Web interface for Documentation Chatbot using Gradio
Handles web UI, formatting, and Gradio-specific logic
"""

from pathlib import Path
from typing import List
import gradio as gr

# Handle both package and direct script execution
try:
    from ..core import DocumentationChatbot, ChatbotConfig
except ImportError:
    from core import DocumentationChatbot
    from core.config import ChatbotConfig


class WebUI:
    """Gradio web interface for documentation chatbot"""

    def __init__(self, chatbot: DocumentationChatbot, agentic_chatbot=None, agentic_smol_chatbot=None):
        """
        Initialize web UI

        Args:
            chatbot: DocumentationChatbot instance
            agentic_chatbot: Optional AgenticChatbot instance
            agentic_smol_chatbot: Optional AgenticSmolChatbot instance
        """
        self.chatbot = chatbot
        self.agentic_chatbot = agentic_chatbot
        self.agentic_smol_chatbot = agentic_smol_chatbot

    def _format_answer_markdown(self, result: dict) -> str:
        """
        Format answer as markdown for Gradio

        Args:
            result: Result dict from chatbot.ask()

        Returns:
            Markdown-formatted answer string
        """
        if result.get("error"):
            return f"**Error:** {result['error']}"

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        filters = result.get("filters", {})

        # Surface a groundedness warning before anything else — agentic-smol reports
        # grounded=False when the agent answered without ever reading a page, i.e.
        # straight from the model's parametric knowledge rather than the docs.
        if filters.get("grounded") is False:
            answer = (
                "> **Warning:** this answer was produced without reading any documentation "
                "page, so it may come from the model's own knowledge rather than the docs.\n\n"
                + answer
            )

        # Add sources section
        if sources:
            answer += "\n\n---\n**Sources:**\n"
            modules_used = set()

            for i, source in enumerate(sources[:5], 1):
                title = source.get("title", "Unknown")
                url = source.get("url", "")
                mod = source.get("module", "?")
                dtype = source.get("doc_category", "?")
                modules_used.add(mod)
                answer += f"{i}. [{mod}/{dtype}] [{title}]({url})\n"

            if len(modules_used) > 1:
                answer += f"\n*Uses {len(modules_used)} modules: {', '.join(sorted(modules_used))}*"

        # Retrieval / agent pipeline summary — makes it visible which stages actually ran
        # rather than which checkboxes happened to be ticked.
        notes = []
        if filters.get("deep_dive"):
            notes.append("Deep Dive mode (batch summarization then synthesis)")
            bypassed = filters.get("bypassed_by_deep_dive")
            if bypassed:
                pretty = {"reranker": "reranking", "hyde": "HyDE", "bm25": "BM25 hybrid retrieval",
                          "title_boost": "title/identifier boost"}
                names = ", ".join(pretty.get(b, b) for b in bypassed)
                notes.append(f"**{names} skipped** — Deep Dive uses its own retrieval path")
        else:
            stages = [name for key, name in
                      (("bm25", "BM25 hybrid"), ("title_boost", "title boost"),
                       ("hyde", "HyDE"), ("reranker", "reranker"))
                      if filters.get(key)]
            if stages:
                notes.append("Retrieval: " + " + ".join(stages))
        if filters.get("rounds_used") is not None:
            notes.append(f"Agentic rounds used: {filters['rounds_used']}")
        if filters.get("steps_used") is not None:
            notes.append(f"Agent steps used: {filters['steps_used']}")

        if notes:
            answer += "\n\n" + "\n".join(f"*{n}*" for n in notes)

        return answer

    def _handle_message(
        self,
        message: str,
        history: List,
        mode: str,
        # RAG params
        module_filter: str,
        doc_type_filter: str,
        response_style: str,
        deep_dive: bool,
        search_depth: int,
        reranker_enabled: bool,
        top_n: int,
        hyde_enabled: bool,
        bm25_enabled: bool,
        answer_length: int,
        # Agentic params
        max_chars_per_page: int,
        max_pages_per_round: int,
        max_pages_round2: int,
    ) -> str:
        """Handle incoming chat message."""
        if mode == "Agentic" and self.agentic_chatbot is not None:
            try:
                result = self.agentic_chatbot.ask(
                    message,
                    max_chars_per_page=max_chars_per_page,
                    max_pages_per_round=max_pages_per_round,
                    max_pages_round2=max_pages_round2,
                    max_tokens=answer_length,
                )
                return self._format_answer_markdown(result)
            except Exception as e:
                return f"**Error:** {str(e)}"

        if mode == "Agentic (smolagents)" and self.agentic_smol_chatbot is not None:
            try:
                result = self.agentic_smol_chatbot.ask(message, max_tokens=answer_length)
                return self._format_answer_markdown(result)
            except Exception as e:
                return f"**Error:** {str(e)}"

        try:
            module = module_filter if module_filter != "All" else None
            doc_type = doc_type_filter if doc_type_filter != "All" else None

            # Temperature comes from the style preset; k and deep_dive from explicit controls.
            # The "Config default" style passes None so config.temperature applies.
            style_config = ChatbotConfig.RESPONSE_STYLES.get(response_style, {})
            temperature = style_config.get("temperature")

            result = self.chatbot.ask(
                message,
                module=module,
                doc_type=doc_type,
                deep_dive=deep_dive,
                k=search_depth,
                temperature=temperature,
                max_tokens=answer_length,
                reranker_enabled=reranker_enabled,
                top_n=top_n,
                hyde_enabled=hyde_enabled,
                bm25_enabled=bm25_enabled,
            )

            return self._format_answer_markdown(result)

        except Exception as e:
            return f"**Error:** {str(e)}"

    def update_button_text(theme):
        return "Switch to Light Mode" if theme == "dark" else "Switch to Dark Mode"

    def _build_gradio_interface(self) -> gr.Blocks:
        """
        Build Gradio interface

        Returns:
            Gradio Blocks interface
        """
        # Example questions with new parameter format
        # [question, module, doc_type, response_style, search_depth, answer_length]
        examples = []

        cfg = self.chatbot.config
        project = cfg.project_name
        k_default = cfg.k_standard

        # Build interface
        has_agentic = self.agentic_chatbot is not None
        has_agentic_smol = self.agentic_smol_chatbot is not None
        has_reranker = self.chatbot.reranker is not None
        has_hyde = self.chatbot.hyde_llm is not None
        has_bm25 = self.chatbot.bm25_index is not None
        agentic_cfg = self.agentic_chatbot._agentic_cfg if has_agentic else (
            self.agentic_smol_chatbot._agentic_cfg if has_agentic_smol else None
        )

        with gr.Blocks(title=f"{project} Documentation Chatbot") as demo:
            with gr.Row():
                with gr.Column(scale=10):
                    gr.Markdown(f"# {project} Documentation Chatbot")
                with gr.Column(scale=1, min_width=120):
                    theme_toggle = gr.Button("Switch theme", size="sm", elem_id="theme-toggle-btn")

            with gr.Row():
                with gr.Column(scale=1, min_width=240):

                    # ── Mode toggle (only shown when at least one agentic mode is available) ──
                    mode_choices = ["RAG"]
                    if has_agentic:
                        mode_choices.append("Agentic")
                    if has_agentic_smol:
                        mode_choices.append("Agentic (smolagents)")
                    mode_radio = gr.Radio(
                        choices=mode_choices,
                        value="RAG",
                        label="Mode",
                        info="RAG: vector retrieval · Agentic: reads HTML pages directly · "
                             "Agentic (smolagents): multi-hop page browsing, experimental",
                        visible=has_agentic or has_agentic_smol,
                    )

                    # ── RAG params ────────────────────────────────────────────────
                    with gr.Column(visible=True) as rag_col:
                        gr.Markdown("### Filters")
                        module_filter = gr.Dropdown(
                            choices=["All"] + self.chatbot.available_modules,
                            value="All",
                            label="Module",
                            info="Restrict search to one module, or keep All to search across all.",
                        )
                        doc_type_filter = gr.Dropdown(
                            choices=["All", "Dev", "User"],
                            value="All",
                            label="Doc Type",
                            info="Dev = API reference & classes · User = tutorials & guides",
                        )

                        gr.Markdown("---")
                        gr.Markdown("### Retrieval")
                        search_depth = gr.Slider(
                            minimum=10,
                            maximum=max(80, cfg.k_standard, cfg.k_deep_dive),
                            value=k_default,
                            step=5,
                            label="Search depth (chunks)",
                            info=f"How many documentation chunks to retrieve before ranking. More = broader "
                                 f"context, slower. Defaults to k_standard ({cfg.k_standard}); switches to "
                                 f"k_deep_dive ({cfg.k_deep_dive}) when Deep Dive is enabled.",
                        )

                        deep_dive_notice = gr.Markdown(
                            "**Deep Dive is on** — it runs its own retrieve-and-summarize pipeline, so the "
                            "reranker, HyDE and BM25 options below do not apply and are hidden.",
                            visible=False,
                        )

                        # Grouped so Deep Dive can hide the whole lot at once — these stages
                        # are genuinely bypassed by the deep-dive chain.
                        with gr.Column(visible=True) as pipeline_col:
                            gr.Markdown("---")
                            gr.Markdown("### Hybrid search")
                            bm25_enabled = gr.Checkbox(
                                label="Enable BM25 hybrid retrieval",
                                value=has_bm25,
                                interactive=has_bm25,
                                info=(
                                    "Fuses a BM25 keyword search over the same corpus with the dense vector "
                                    "search using Reciprocal Rank Fusion. BM25 matches exact tokens, so it "
                                    "rescues precise symbol names (addFeature vs addNode) that embeddings "
                                    "blur together; the vector side still handles paraphrase. Costs no extra "
                                    "LLM call. Requires bm25_enabled: true in config.json and the extraction "
                                    "JSONL next to chromadb_path."
                                    if has_bm25 else
                                    "Requires bm25_enabled: true in config.json and "
                                    f"{Path(cfg.bm25_jsonl_path).name} next to chromadb_path "
                                    "(not available for this project)."
                                ),
                            )

                            gr.Markdown("---")
                            gr.Markdown("### Reranker")
                            reranker_enabled = gr.Checkbox(
                                label="Enable reranker",
                                value=has_reranker,
                                interactive=has_reranker,
                                info="Cross-encoder reranking. Requires the reranker model to be configured." if not has_reranker else "Cross-encoder reranking (BAAI/bge-reranker-v2-m3).",
                            )
                            with gr.Column(visible=has_reranker) as top_n_col:
                                top_n = gr.Slider(
                                    minimum=1,
                                    maximum=max(40, cfg.top_n_after_rerank),
                                    value=cfg.top_n_after_rerank,
                                    step=1,
                                    label="Top-N after rerank",
                                    info="Docs kept after cross-encoder reranking.",
                                )

                            gr.Markdown("---")
                            gr.Markdown("### HyDE")
                            hyde_enabled = gr.Checkbox(
                                label="Enable HyDE",
                                value=has_hyde,
                                interactive=has_hyde,
                                info=(
                                "Hypothetical Document Embeddings: before searching, an LLM writes a short "
                                "fake documentation passage answering your question, and that passage — not "
                                "your literal question — is embedded and used for the vector search. This "
                                "helps when your question is phrased very differently from how the docs word "
                                "things (paraphrase gap), since the fake passage is written in \"documentation "
                                "voice\" and embeds closer to real doc chunks. It costs one extra LLM call per "
                                "question and can hurt exact symbol/keyword lookups (BM25 and reranking still "
                                "use your real question, so those aren't affected). Requires hyde_enabled: true "
                                "in config.json."
                                    if has_hyde else
                                    "Requires hyde_enabled: true in config.json (not configured for this project)."
                                ),
                            )

                        gr.Markdown("---")
                        gr.Markdown("### Response")
                        style_choices = list(ChatbotConfig.RESPONSE_STYLES) + [ChatbotConfig.CONFIG_DEFAULT_STYLE]
                        style_info = " · ".join(
                            f"{name.split(' (')[0]}: temp={preset['temperature']}"
                            for name, preset in ChatbotConfig.RESPONSE_STYLES.items()
                        )
                        response_style = gr.Dropdown(
                            choices=style_choices,
                            value="Precise (Recommended)",
                            label="Style",
                            info=f"{style_info} · {ChatbotConfig.CONFIG_DEFAULT_STYLE}: "
                                 f"temp={cfg.temperature} (from config.json)",
                        )
                        deep_dive = gr.Checkbox(
                            label="Deep Dive mode",
                            value=False,
                            info="Multi-step analysis: summaries per batch then a final synthesis. Slower but "
                                 "more thorough. Uses its own retrieval path — reranker, HyDE and BM25 do not apply.",
                        )

                    # ── Agentic params ────────────────────────────────────────────
                    with gr.Column(visible=False) as agentic_col:
                        gr.Markdown("### Page index")
                        gr.Textbox(
                            value=agentic_cfg.page_index_path if agentic_cfg else "",
                            label="page_index.json",
                            interactive=False,
                            info="Configured in config.json → agentic.page_index_path",
                        )

                        gr.Markdown("---")
                        gr.Markdown("### Agentic params")
                        max_chars_per_page = gr.Slider(
                            minimum=1000,
                            maximum=20000,
                            value=agentic_cfg.max_chars_per_page if agentic_cfg else 8000,
                            step=1000,
                            label="Max chars per page",
                            info="Characters read per HTML page. More = richer context, higher token cost.",
                        )
                        max_pages_per_round = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=agentic_cfg.max_pages_per_round if agentic_cfg else 3,
                            step=1,
                            label="Pages per round 1",
                            info="Pages selected and read in the first retrieval round.",
                        )
                        max_pages_round2 = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=agentic_cfg.max_pages_round2 if agentic_cfg else 2,
                            step=1,
                            label="Pages per round 2",
                            info="Additional pages read if NEED_MORE_INFO is triggered.",
                        )

                    # ── Agentic-smol params ───────────────────────────────────────
                    with gr.Column(visible=False) as agentic_smol_col:
                        gr.Markdown("### Page index")
                        gr.Textbox(
                            value=agentic_cfg.page_index_path if agentic_cfg else "",
                            label="page_index.json",
                            interactive=False,
                            info="Configured in config.json → agentic.page_index_path",
                        )

                        gr.Markdown("---")
                        gr.Markdown("### Agentic-smol")
                        gr.Markdown(
                            "Experimental: a smolagents CodeAgent decides for itself how many "
                            "search/read cycles to run (up to the configured step budget), instead "
                            "of a fixed 1-2 round script. Not yet measured against classic Agentic "
                            "mode — treat answers as unverified."
                        )
                        gr.Textbox(
                            value=str(agentic_cfg.max_steps) if agentic_cfg else "",
                            label="Max steps",
                            interactive=False,
                            info="Configured in config.json → agentic.max_steps (not runtime-tunable yet).",
                        )

                    # ── Shared ────────────────────────────────────────────────────
                    gr.Markdown("---")
                    answer_length = gr.Slider(
                        minimum=500,
                        maximum=max(4000, cfg.max_tokens),
                        value=cfg.max_tokens,
                        step=500,
                        label="Max answer length (tokens)",
                        info=f"Defaults to max_tokens ({cfg.max_tokens}) from config.json.",
                    )

                with gr.Column(scale=3):
                    chatbot_interface = gr.ChatInterface(
                        fn=self._handle_message,
                        additional_inputs=[
                            mode_radio,
                            module_filter,
                            doc_type_filter,
                            response_style,
                            deep_dive,
                            search_depth,
                            reranker_enabled,
                            top_n,
                            hyde_enabled,
                            bm25_enabled,
                            answer_length,
                            max_chars_per_page,
                            max_pages_per_round,
                            max_pages_round2,
                        ],
                        examples=examples,
                        title=None,
                        description=None,
                        chatbot=gr.Chatbot(height=800),
                    )

            # ── Event handlers ────────────────────────────────────────────────────
            if has_agentic or has_agentic_smol:
                def _on_mode_change(mode):
                    return (
                        gr.update(visible=mode == "RAG"),
                        gr.update(visible=mode == "Agentic"),
                        gr.update(visible=mode == "Agentic (smolagents)"),
                    )

                mode_radio.change(
                    fn=_on_mode_change,
                    inputs=[mode_radio],
                    outputs=[rag_col, agentic_col, agentic_smol_col],
                )

            if has_reranker:
                reranker_enabled.change(
                    fn=lambda enabled: gr.update(visible=enabled),
                    inputs=[reranker_enabled],
                    outputs=[top_n_col],
                )

            # Deep Dive bypasses the rerank/HyDE/BM25 pipeline entirely, so hide those
            # controls rather than leaving them ticked and inert. Search depth also
            # follows k_deep_dive/k_standard, which is otherwise unreachable from the web UI.
            def _on_deep_dive_change(enabled):
                return (
                    gr.update(visible=not enabled),
                    gr.update(visible=enabled),
                    gr.update(value=cfg.k_deep_dive if enabled else cfg.k_standard),
                )

            deep_dive.change(
                fn=_on_deep_dive_change,
                inputs=[deep_dive],
                outputs=[pipeline_col, deep_dive_notice, search_depth],
            )

            js_toggle_light_dark = """
                () => {
                    const url = new URL(window.location);
                    const currentTheme = url.searchParams.get('__theme');
                    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
                    url.searchParams.set('__theme', newTheme);
                    window.location.href = url.toString();
                }
                """

            # Theme toggle functionality
            theme_toggle.click(
                fn=None,
                inputs=None,
                outputs=theme_toggle,
                js=js_toggle_light_dark,
            )

        return demo

    def launch(self, share: bool = False, port: int = 7860):
        """
        Launch web interface

        Args:
            share: Create public URL via Gradio
            port: Port number for web server
        """
        
        demo = self._build_gradio_interface()

        class BlackColor(gr.themes.Color):
            def __init__(self):
                # Define the color palette (50-950 brightness values)
                super().__init__(
                    c50="#000000",  # Lightest black
                    c100="#000000",
                    c200="#000000",
                    c300="#1a1a1a",
                    c400="#333333",
                    c500="#4d4d4d",  # Mid black
                    c600="#666666",
                    c700="#808080",
                    c800="#999999",
                    c900="#b3b3b3",
                    c950="#cccccc",  # Lightest variant
                )


        # Create theme with larger text - using Default which is explicitly light
        theme = gr.themes.Default(
            primary_hue="red",
            # neutral_hue="slate",
            text_size=gr.themes.Size(
                lg="22px",
                md="20px",
                sm="18px",
                xl="26px",
                xs="16px",
                xxl="30px",
                xxs="14px",
            ),
        )

        print("\n" + "=" * 70)
        print(f"Starting web interface on http://localhost:{port}")
        print("=" * 70)

        demo.launch(share=share, server_name="127.0.0.1", server_port=port, theme=theme)

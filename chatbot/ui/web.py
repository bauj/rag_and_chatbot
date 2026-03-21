"""
Web interface for Documentation Chatbot using Gradio
Handles web UI, formatting, and Gradio-specific logic
"""

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

    def __init__(self, chatbot: DocumentationChatbot):
        """
        Initialize web UI

        Args:
            chatbot: DocumentationChatbot instance
        """
        self.chatbot = chatbot

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

            # Add deep dive notice if enabled
            if filters.get("deep_dive"):
                answer += "\n\n*Note: This answer was generated using Deep Dive mode for comprehensive analysis.*"

        return answer

    def _handle_message(
        self,
        message: str,
        history: List,
        module_filter: str,
        doc_type_filter: str,
        response_style: str,
        deep_dive: bool,
        search_depth: int,
        answer_length: int,
    ) -> str:
        """Handle incoming chat message."""
        try:
            module = module_filter if module_filter != "All" else None
            doc_type = doc_type_filter if doc_type_filter != "All" else None

            # Temperature comes from the style preset; k and deep_dive from explicit controls
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

        project = self.chatbot.config.project_name
        k_default = self.chatbot.config.k_standard

        self._custom_css = """
        /* === CHAT BUBBLE CONTRAST === */
        /* User messages — deep navy with bright text */
        .message.user,
        .bubble-wrap.user .message,
        div[data-testid="user"] .message {
            background: #0f2d52 !important;
            color: #c8e0fa !important;
            border: 1px solid #1e5296 !important;
        }
        /* Bot messages — dark teal with soft green text */
        .message.bot,
        .bubble-wrap.bot .message,
        div[data-testid="bot"] .message {
            background: #0d2118 !important;
            color: #b8e4c9 !important;
            border: 1px solid #1a5c38 !important;
        }
        /* Text inside bubbles */
        .message.user p, .message.user li, .message.user code,
        div[data-testid="user"] p {
            color: #c8e0fa !important;
        }
        .message.bot p, .message.bot li, .message.bot code,
        div[data-testid="bot"] p {
            color: #b8e4c9 !important;
        }

        /* === TOOLTIPS === */
        /* Make the block container a positioning context */
        .block {
            position: relative !important;
        }
        /* Hide info text by default */
        .block .info,
        span.info {
            visibility: hidden !important;
            opacity: 0 !important;
            position: absolute !important;
            top: auto !important;
            bottom: calc(100% + 4px) !important;
            left: 0 !important;
            width: 240px !important;
            background: #1a1a2e !important;
            color: #d0d8e8 !important;
            border: 1px solid #334 !important;
            padding: 7px 11px !important;
            border-radius: 6px !important;
            font-size: 12px !important;
            font-style: normal !important;
            line-height: 1.5 !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.5) !important;
            z-index: 9999 !important;
            pointer-events: none !important;
            transition: opacity 0.15s ease !important;
            white-space: normal !important;
        }
        /* Show on hover */
        .block:hover .info,
        .block:hover span.info {
            visibility: visible !important;
            opacity: 1 !important;
        }
        /* Small triangle pointer */
        .block .info::after {
            content: "" !important;
            position: absolute !important;
            top: 100% !important;
            left: 16px !important;
            border: 5px solid transparent !important;
            border-top-color: #334 !important;
        }
        """

        # Build interface
        with gr.Blocks(title=f"{project} Documentation Chatbot") as demo:
            with gr.Row():
                with gr.Column(scale=10):
                    gr.Markdown(f"# {project} Documentation Chatbot")
                with gr.Column(scale=1, min_width=120):
                    theme_toggle = gr.Button("Switch theme", size="sm", elem_id="theme-toggle-btn")

            with gr.Row():
                with gr.Column(scale=1, min_width=240):
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
                    gr.Markdown("### Response")

                    response_style = gr.Dropdown(
                        choices=["Precise (Recommended)", "Balanced", "Comprehensive"],
                        value="Precise (Recommended)",
                        label="Style",
                        info="Precise: temp=0.0 · Balanced: temp=0.2 · Comprehensive: temp=0.1",
                    )
                    deep_dive = gr.Checkbox(
                        label="Deep Dive mode",
                        value=False,
                        info="Multi-step analysis: summaries per batch then a final synthesis. Slower but more thorough.",
                    )
                    search_depth = gr.Slider(
                        minimum=10,
                        maximum=80,
                        value=k_default,
                        step=5,
                        label="Search depth (chunks)",
                        info="How many documentation chunks to retrieve before ranking. More = broader context, slower.",
                    )
                    answer_length = gr.Slider(
                        minimum=500,
                        maximum=4000,
                        value=2000,
                        step=500,
                        label="Max answer length (tokens)",
                    )

                with gr.Column(scale=3):
                    chatbot_interface = gr.ChatInterface(
                        fn=self._handle_message,
                        additional_inputs=[
                            module_filter,
                            doc_type_filter,
                            response_style,
                            deep_dive,
                            search_depth,
                            answer_length,
                        ],
                        examples=examples,
                        title=None,
                        description=None,
                        chatbot=gr.Chatbot(height=600),
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

        demo.launch(share=share, server_name="127.0.0.1", server_port=port, theme=theme, css=self._custom_css)

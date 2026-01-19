"""
Web interface for SALOME Documentation Chatbot using Gradio
Handles web UI, formatting, and Gradio-specific logic
"""

from typing import List
import gradio as gr

# Handle both package and direct script execution
try:
    from ..core import SALOMEChatbot, ChatbotConfig
except ImportError:
    from core import SALOMEChatbot
    from core.config import ChatbotConfig


class WebUI:
    """Gradio web interface for SALOME chatbot"""

    def __init__(self, chatbot: SALOMEChatbot):
        """
        Initialize web UI

        Args:
            chatbot: SALOMEChatbot instance
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
        search_depth: int,
        answer_length: int,
    ) -> str:
        """
        Handle incoming chat message

        Args:
            message: User's message
            history: Chat history (unused, maintained by Gradio)
            module_filter: Selected module filter
            doc_type_filter: Selected doc type filter
            response_style: Response style preset
            search_depth: Number of chunks to retrieve
            answer_length: Maximum answer length in tokens

        Returns:
            Formatted answer string
        """
        try:
            # Convert filter values
            module = module_filter if module_filter != "All" else None
            doc_type = doc_type_filter if doc_type_filter != "All" else None

            # Get response style preset from config
            style_config = ChatbotConfig.RESPONSE_STYLES.get(response_style, {})

            # Priority: UI controls > style preset > config defaults
            # Deep dive mode from style preset
            deep_dive = style_config.get("deep_dive", False)

            # Temperature: style preset (UI doesn't expose direct control)
            temperature = style_config.get("temperature")

            # K and max_tokens: UI controls override everything
            k = search_depth
            max_tokens = answer_length

            # Get answer with runtime overrides
            result = self.chatbot.ask(
                message,
                module=module,
                doc_type=doc_type,
                deep_dive=deep_dive,
                k=k,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Format and return
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
        examples = [
            [
                "What is ModelAPI::Feature in SHAPER?",
                "SHAPER",
                "Dev",
                "Precise (Recommended)",
                5,
                2000,
            ],
            [
                "How do I create a mesh from geometry?",
                "All",
                "User",
                "Precise (Recommended)",
                4,
                2000,
            ],
            [
                "What meshing algorithms are in SMESH?",
                "SMESH",
                "All",
                "Balanced",
                5,
                2000,
            ],
            [
                "Show me GUI components for dialogs",
                "GUI",
                "Dev",
                "Precise (Recommended)",
                4,
                1500,
            ],
            [
                "Tutorial for creating a box in SHAPER",
                "SHAPER",
                "User",
                "Precise (Recommended)",
                4,
                2000,
            ],
            [
                "Explain the complete mesh generation workflow in SMESH",
                "SMESH",
                "All",
                "Comprehensive",
                6,
                3000,
            ],
        ]

        # Build interface
        with gr.Blocks(title="SALOME Documentation Chatbot") as demo:
            with gr.Row():
                with gr.Column(scale=10):
                    gr.Markdown("""<img src="http://example.com/wp-content/uploads/2019/08/salome_text_alpha.png" style="width: 250px; height: auto; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"> Documentation Chatbot""")

                with gr.Column(scale=1, min_width=100):
                    theme_toggle = gr.Button(
                        "Switch theme", size="sm", elem_id="theme-toggle-btn"
                    )

            gr.Markdown(
                f"*Multi-module support: {', '.join(self.chatbot.available_modules)}*"
            )
        
            with gr.Row():
                with gr.Column(scale=1):
                    # Content filters
                    gr.Markdown("### Filters")
                    module_filter = gr.Dropdown(
                        choices=["All"] + self.chatbot.available_modules,
                        value="All",
                        label="Module",
                        info="Filter by SALOME module",
                    )
                    doc_type_filter = gr.Dropdown(
                        choices=["All", "Dev", "User"],
                        value="All",
                        label="Doc Type",
                        info="Filter by documentation type",
                    )

                    gr.Markdown("---")

                    # Response controls
                    gr.Markdown("### Response Controls")
                    response_style = gr.Dropdown(
                        choices=["Precise (Recommended)", "Balanced", "Comprehensive"],
                        value="Precise (Recommended)",
                        label="Response Style",
                        info="Controls temperature and retrieval strategy",
                    )
                    search_depth = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=1,
                        step=1,
                        label="Search Depth",
                        info="Number of documentation chunks to retrieve",
                    )
                    answer_length = gr.Slider(
                        minimum=500,
                        maximum=4000,
                        value=2000,
                        step=500,
                        label="Answer Length (tokens)",
                        info="Maximum length of generated answer",
                    )

                    gr.Markdown("---")
                    gr.Markdown("""
                    **Module Filters:**
                    - **SHAPER**: CAD modeling, geometry
                    - **SMESH**: Mesh generation
                    - **GUI**: Interface components

                    **Doc Types:**
                    - **Dev**: API reference, classes
                    - **User**: Tutorials, guides

                    **Response Styles:**
                    - **Precise**: Temp=0.0, focused (40 chunks)
                    - **Balanced**: Temp=0.2, broader (50 chunks)
                    - **Comprehensive**: Temp=0.1, deep dive (60 chunks)
                    """)

                with gr.Column(scale=3):
                    # Use ChatInterface with resizable chatbot
                    chatbot_interface = gr.ChatInterface(
                        fn=self._handle_message,
                        additional_inputs=[
                            module_filter,
                            doc_type_filter,
                            response_style,
                            search_depth,
                            answer_length,
                        ],
                        examples=examples,
                        title=None,
                        description="Ask anything about SALOME! Use controls on the left to customize responses.",
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

        demo.launch(share=share, server_name="127.0.0.1", server_port=port, theme=theme)

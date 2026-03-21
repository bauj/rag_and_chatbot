# Documentation RAG Chatbot

RAG chatbot with clean separation between business logic and UI. Queries a ChromaDB vector database built by the extraction pipeline.

## Architecture

```
chatbot/
├── core/
│   ├── config.py           # ChatbotConfig, LLMConfig, EmbeddingConfig, RerankerConfig
│   └── rag_chatbot.py      # DocumentationChatbot — RAG logic
├── ui/
│   ├── terminal.py         # Interactive terminal interface
│   └── web.py              # Gradio web interface
├── chatbot.py              # Unified entry point
└── config.example.json     # Example configuration
```

## Configuration

Copy and edit the example:

```bash
cp config.example.json config.json
```

| Field | Default | Description |
|---|---|---|
| `project_name` | `"docs"` | Must match `project_name` used during extraction |
| `chromadb_path` | derived | Path to ChromaDB produced by the extractor |
| `llm.base_url` | `http://localhost:8080/v1` | OpenAI-compatible LLM endpoint |
| `llm.model` | `"mistral"` | Model name |
| `llm.api_key` | `"dummy"` | Use `"dummy"` for local models |
| `embedding.model` | `"all-MiniLM-L6-v2"` | Must match extraction config |
| `embedding.type` | `"local"` | `"local"` or `"api"` |
| `reranker.model` | `"BAAI/bge-reranker-v2-m3"` | Cross-encoder reranker |
| `k_standard` | `40` | Chunks retrieved in standard mode |
| `k_deep_dive` | `60` | Chunks retrieved in deep dive mode |
| `temperature` | `0.0` | LLM temperature |
| `max_tokens` | `2000` | Max tokens per response |

Config is loaded in this priority order: CLI arguments > `config.json` > defaults.

**`embedding.model` must be identical to `extraction/config.json`** — mismatch causes wrong retrieval with no error.

Remove the `reranker` block entirely to disable reranking.

## Usage

### Terminal Interface

```bash
# Interactive mode
python chatbot.py

# Single question
python chatbot.py --question "How do I create a mesh?"

# With filters
python chatbot.py --question "How do I create a mesh?" --module MODULE_A --type user

# Deep dive mode
python chatbot.py --question "Explain the full workflow" --deep-dive
```

Commands in interactive mode:
```
module:MODULE_A   - Filter by module
type:dev          - Filter developer docs only
type:user         - Filter user docs only
deep              - Toggle deep dive mode
clear             - Clear all filters
stats             - Show database statistics
exit              - Exit
```

### Web Interface

```bash
# Launch on default port 7860
python chatbot.py --web

# Custom port
python chatbot.py --web --port 8080

# Public URL via Gradio
python chatbot.py --web --share
```

### CLI Overrides

```bash
# Override LLM endpoint
python chatbot.py --base-url http://localhost:11434/v1 --model llama3

# Load a specific config file
python chatbot.py --config /path/to/my_config.json
```

### Python API

Run from inside the `chatbot/` directory:

```python
from core import ChatbotConfig, DocumentationChatbot

config = ChatbotConfig.load("config.json")
chatbot = DocumentationChatbot(config)

# Basic query
result = chatbot.ask("How do I create a mesh?")
print(result["answer"])
for source in result["sources"]:
    print(f"  - {source['title']}")

# With filters
result = chatbot.ask(
    "Show mesh generation examples",
    module="MODULE_A",
    doc_type="user"
)

# Custom parameters
result = chatbot.ask(
    "Explain the full workflow",
    deep_dive=True,
    k=70,
    temperature=0.1,
    max_tokens=3000
)

stats = chatbot.get_stats()
print(f"Modules: {stats['available_modules']}")
```

## Response Styles (Web UI)

| Style | Temperature | Chunks | Deep Dive |
|---|---|---|---|
| Precise (default) | 0.0 | 40 | No |
| Balanced | 0.2 | 50 | No |
| Comprehensive | 0.1 | 60 | Yes |

**Deep Dive Mode:** retrieves 60 chunks, splits them into batches of 10, generates one summary per batch, then synthesizes a final answer. Use for complex multi-part questions.

## Troubleshooting

**"ChromaDB not found"**
```bash
cd ../extraction && python process_docs.py --config config.json
```

**"Connection refused" (LLM endpoint)**
```bash
curl http://localhost:8080/v1/models  # check endpoint is up
ollama serve                          # if using Ollama
```

**Retrieval returns wrong results**
- Check that `embedding.model` matches exactly in both `extraction/config.json` and `chatbot/config.json`

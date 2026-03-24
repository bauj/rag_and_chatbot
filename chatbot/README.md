# Documentation Chatbot

Chatbot with two retrieval modes and clean separation between business logic and UI.

- **RAG mode** — queries a ChromaDB vector database, with optional cross-encoder reranking
- **Agentic mode** — searches a page index, asks the LLM to select pages, reads HTML files directly; no vector retrieval at query time

## Architecture

```
chatbot/
├── core/
│   ├── config.py               # ChatbotConfig, LLMConfig, EmbeddingConfig, RerankerConfig, AgenticConfig
│   ├── rag_chatbot.py          # DocumentationChatbot — RAG mode
│   └── agentic_chatbot.py      # AgenticChatbot — agentic mode
├── ui/
│   ├── terminal.py             # Interactive terminal interface
│   └── web.py                  # Gradio web interface
├── chatbot.py                  # Unified entry point
└── config.example.json         # Example configuration
```

## Configuration

Copy and edit the example:

```bash
cp config.example.json config.json
```

| Field | Default | Description |
|---|---|---|
| `project_name` | `"docs"` | Must match `project_name` used during extraction |
| `chromadb_path` | derived | Path to ChromaDB produced by the extractor (RAG mode) |
| `llm.base_url` | `http://localhost:8080/v1` | OpenAI-compatible LLM endpoint |
| `llm.model` | `"mistral"` | Model name |
| `llm.api_key` | `"dummy"` | Use `"dummy"` for local models |
| `embedding.model` | `"all-MiniLM-L6-v2"` | Must match extraction config |
| `embedding.type` | `"local"` | `"local"` or `"api"` |
| `reranker` | `null` | Set to `null` to disable. Enable with `{"model": "BAAI/bge-reranker-v2-m3", "type": "local"}` |
| `agentic` | `null` | Set to `null` to disable. Enable with `{"page_index_path": "...", ...}` (see below) |
| `k_standard` | `40` | Chunks retrieved in standard mode (RAG) |
| `k_deep_dive` | `60` | Chunks retrieved in deep dive mode (RAG) |
| `top_n_after_rerank` | `15` | Docs kept after cross-encoder reranking |
| `temperature` | `0.0` | LLM temperature |
| `max_tokens` | `2000` | Max tokens per response |

Config is loaded in this priority order: CLI arguments > `config.json` > defaults.

**`embedding.model` must be identical to `extraction/config.json`** — mismatch causes wrong retrieval with no error.

### Agentic config block

```json
"agentic": {
  "page_index_path": "../extraction/my_project_docs_extracted/page_index.json",
  "max_chars_per_page": 8000,
  "max_pages_per_round": 3,
  "max_pages_round2": 2
}
```

| Field | Default | Description |
|---|---|---|
| `page_index_path` | required | Path to `page_index.json` (relative to `chatbot/` or absolute) |
| `max_chars_per_page` | `8000` | Max characters read per HTML page |
| `max_pages_per_round` | `3` | Pages selected and read in Round 1 |
| `max_pages_round2` | `2` | Additional pages read in Round 2 (if NEED_MORE_INFO triggered) |

`page_index.json` is generated automatically when running `extraction/process_docs.py`.

## Usage

### Terminal Interface

```bash
# Interactive mode (RAG, default)
python chatbot.py

# Agentic mode
python chatbot.py --mode agentic

# Single question
python chatbot.py --question "How do I create a mesh?"

# With filters (RAG mode only)
python chatbot.py --question "How do I create a mesh?" --module MODULE_A --type user

# Deep dive mode (RAG mode only)
python chatbot.py --question "Explain the full workflow" --deep-dive
```

Commands in interactive mode:
```
mode:rag          - Switch to RAG mode (vector retrieval)
mode:agentic      - Switch to Agentic mode (reads HTML pages directly)
module:MODULE_A   - Filter by module (RAG only)
type:dev          - Filter developer docs only (RAG only)
type:user         - Filter user docs only (RAG only)
deep              - Toggle Deep Dive mode (RAG only)
reranker          - Toggle cross-encoder reranker on/off (RAG only)
clear             - Clear all filters
stats             - Show database statistics (RAG only)
exit              - Exit
```

`mode:agentic` is only available when `config.agentic` is set. `reranker` is only shown when a reranker model is configured.

### Web Interface

```bash
# Launch on default port 7860
python chatbot.py --web

# Custom port
python chatbot.py --web --port 8080

# Public URL via Gradio
python chatbot.py --web --share
```

The web UI includes a **Mode** toggle (RAG / Agentic). Agentic mode is only available if `config.agentic` is set.

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
from core import ChatbotConfig, DocumentationChatbot, AgenticChatbot

config = ChatbotConfig.load("config.json")

# --- RAG mode ---
chatbot = DocumentationChatbot(config)

result = chatbot.ask("How do I create a mesh?")
print(result["answer"])
for source in result["sources"]:
    print(f"  - {source['title']}")

result = chatbot.ask(
    "Show mesh generation examples",
    module="MODULE_A",
    doc_type="user"
)

result = chatbot.ask(
    "Explain the full workflow",
    deep_dive=True,
    k=70,
    temperature=0.1,
    max_tokens=3000,
    reranker_enabled=True,  # disable at runtime to skip reranking
    top_n=10,               # override config.top_n_after_rerank
)

stats = chatbot.get_stats()
print(f"Modules: {stats['available_modules']}")

# --- Agentic mode (requires config.agentic to be set) ---
agentic = AgenticChatbot(config)

result = agentic.ask(
    "How do I create a mesh?",
    max_chars_per_page=10000,   # override config at runtime
    max_pages_per_round=5,
    max_pages_round2=3,
)
print(result["answer"])
print(result["filters"]["rounds_used"])  # 1 or 2
for source in result["sources"]:
    print(f"  - {source['title']} ({source['module']}/{source['doc_category']})")
```

Both `.ask()` methods return the same dict shape: `{answer, sources, filters, error}`.

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

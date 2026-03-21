# Documentation RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for any project with HTML documentation (Sphinx, Doxygen). Point it at your docs, configure a few JSON fields, and get an intelligent chatbot with multi-module support, cross-encoder reranking, and dual interfaces (terminal + web).

## Overview

**Workflow:**
1. **Extract** — parse your HTML docs into a ChromaDB vector database
2. **Chat** — query the database through a terminal or web interface

**Key Features:**
- Works with any HTML documentation (Sphinx, Doxygen, or custom)
- Multi-module support — query across multiple doc sets simultaneously
- Token-aware chunking (prevents embedding truncation)
- Cross-encoder reranking (`BAAI/bge-reranker-v2-m3`) for better result ranking
- Quality scoring (filters low-value content)
- Code block extraction (enables code-aware retrieval)
- Dual interfaces (terminal + Gradio web)
- Configurable LLM and embedding endpoints (local or remote)

## Architecture

```
rag_and_chatbot/
├── extraction/              # Documentation processing pipeline
│   ├── process_docs.py      # Main extractor
│   ├── config.example.json
│   └── README.md
│
├── chatbot/                 # RAG chatbot
│   ├── core/
│   │   ├── config.py        # Configuration (ChatbotConfig, LLMConfig, ...)
│   │   └── rag_chatbot.py   # RAG logic (DocumentationChatbot)
│   ├── ui/
│   │   ├── terminal.py      # CLI interface
│   │   └── web.py           # Gradio web UI
│   ├── chatbot.py           # Unified entry point
│   └── config.example.json
│
├── tests/                   # Unit tests
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd rag_and_chatbot

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Extract Documentation

```bash
cd extraction
cp config.example.json config.json
# Edit config.json: set project_name, output_dir, and module paths
python process_docs.py --config config.json
```

**Expected output** (in `output_dir`):
- `chromadb/` — vector database
- `{project_name}_docs.json` / `{project_name}_docs.jsonl` — processed chunks
- `statistics.json` — extraction stats

### 3. Configure the Chatbot

```bash
cd ../chatbot
cp config.example.json config.json
# Edit config.json: set project_name, chromadb_path, llm endpoint
```

### 4. Run the Chatbot

```bash
# Interactive terminal
python chatbot.py

# Web interface (Gradio)
python chatbot.py --web

# Single question
python chatbot.py --question "How do I create a mesh?"

# With filters
python chatbot.py --question "How do I create a mesh?" --module MODULE_A --type user
```

## Configuration

### Extraction Config (`extraction/config.json`)

```json
{
  "project_name": "my_project",
  "output_dir": "./my_project_docs_extracted",
  "use_token_chunking": false,

  "modules": {
    "MODULE_A": {
      "description": "What this module does",
      "dev_path": "./module_a/html",
      "user_path": "./module_a/html_gui"
    }
  },

  "embedding": {
    "model": "all-MiniLM-L6-v2",
    "type": "local",
    "base_url": null,
    "api_key": null
  },

  "chunking": {
    "max_tokens": 384,
    "overlap_tokens": 50
  },

  "quality": {
    "min_score": 0.3,
    "min_word_count": 50
  }
}
```

- `dev_path` — API reference / developer docs (Doxygen `html/`)
- `user_path` — tutorials / user guides (Sphinx `_build/html/`)
- Both are optional per module; omit either if not applicable
- `embedding.type: "local"` — runs `sentence-transformers` locally (default)
- `embedding.type: "api"` — calls an OpenAI-compatible embeddings endpoint; fill in `base_url` and `api_key`
- `embedding.model` **must be identical** in both `extraction/config.json` and `chatbot/config.json` — mismatch causes wrong retrieval with no error

### Chatbot Config (`chatbot/config.json`)

```json
{
  "project_name": "my_project",
  "chromadb_path": "../extraction/my_project_docs_extracted/chromadb",
  "_comment_chromadb_path": "Update my_project to match project_name and output_dir from extraction/config.json",

  "llm": {
    "base_url": "http://localhost:8080/v1",
    "model": "mistral",
    "api_key": "dummy"
  },

  "embedding": {
    "model": "all-MiniLM-L6-v2",
    "type": "local",
    "base_url": null,
    "api_key": null
  },

  "reranker": {
    "model": "BAAI/bge-reranker-v2-m3",
    "type": "local"
  }
}
```

**LLM endpoints:**

| Provider | `base_url` | `api_key` |
|---|---|---|
| Ollama (local) | `http://localhost:11434/v1` | `"dummy"` |
| OpenAI | `https://api.openai.com/v1` | your key |
| Mistral | `https://api.mistral.ai/v1` | your key |
| Any OpenAI-compatible | your endpoint | your key |

**Reranker:** remove the `reranker` block entirely to disable reranking (faster, lower quality).

## Features

### Extraction Pipeline

**Token-Aware Chunking** (`use_token_chunking: true`):
- Uses the actual tokenizer for the embedding model (384 tokens max)
- Prevents silent truncation during embedding
- More accurate but slower — use `false` for large datasets

**Quality Scoring:**
```
score = min(word_count / 200, 1.0)
      + 0.1 if has_title
      + 0.1 if word_count > 50
      + 0.1 if word_count > 100

Filter: score >= 0.3
```

**Parent Document Tracking:**
- Every chunk records its source document (`parent_doc_id`)
- Enables parent-expansion retrieval strategies

### Chatbot Pipeline

```
Vector retrieval (large K)
        ↓
Cross-encoder reranking
        ↓
LLM answer generation
```

**Response Styles:**

| Style | Temperature | Chunks | Deep Dive |
|---|---|---|---|
| Precise (default) | 0.0 | 40 | No |
| Balanced | 0.2 | 50 | No |
| Comprehensive | 0.1 | 60 | Yes |

**Deep Dive Mode:**
- Retrieves 60 chunks, batches them (10 per batch)
- Multiple LLM calls: per-batch summaries + final synthesis
- Use for complex, multi-part questions

### Terminal Interface

```
Commands in interactive mode:
  module:MODULE_A   - Filter by module
  type:dev          - Filter developer docs only
  type:user         - Filter user docs only
  deep              - Toggle deep dive mode
  clear             - Clear all filters
  stats             - Show database statistics
  exit              - Exit
```

### Python API

Run from inside the `chatbot/` directory:

```python
from core import ChatbotConfig, DocumentationChatbot

config = ChatbotConfig.load("config.json")
chatbot = DocumentationChatbot(config)

# Basic query
result = chatbot.ask("How do I create a mesh?")
print(result['answer'])
print(result['sources'])

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

## Troubleshooting

**"No HTML files found"**
- Verify that `dev_path` / `user_path` point to actual HTML directories
- Check the path is relative to the working directory when running `process_docs.py`

**"ChromaDB not found"**
- Run the extractor first: `cd extraction && python process_docs.py --config config.json`
- Check `chromadb_path` in `chatbot/config.json` matches the extraction `output_dir`

**Retrieval returns wrong results**
- Most likely cause: embedding model mismatch between extraction and chatbot configs
- The `embedding.model` must be identical in both `extraction/config.json` and `chatbot/config.json`

**"Connection refused" (LLM endpoint)**
```bash
curl http://localhost:8080/v1/models  # check endpoint is running
ollama serve                          # if using Ollama
```

**Web interface not loading**
```bash
python chatbot.py --web --port 7861  # try a different port
```

## Development

### Running Tests

```bash
python -m pytest tests/ -v
```

### Adding a Module

Add an entry to `extraction/config.json` under `modules` and re-run the extractor:

```json
"modules": {
  "NEW_MODULE": {
    "description": "What this module does",
    "dev_path": "./new_module/html"
  }
}
```

The chatbot detects available modules automatically from the database at startup.

## Dependencies

- Python 3.8+
- `beautifulsoup4` — HTML parsing
- `chromadb` — vector database
- `sentence-transformers` — embeddings and reranking
- `langchain` — RAG framework
- `gradio` — web interface
- `transformers` — token-aware chunking (recommended)

Any OpenAI-compatible LLM endpoint (OpenAI, Mistral, Ollama, local models).

See [requirements.txt](requirements.txt) for exact versions.

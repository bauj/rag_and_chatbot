# To do (en vrac)
- Plus généraliste que SALOME : réfléchir à quel type d'arborescence mettre en input
- Améliorer la modularité : faire en sorte que le projet soit standalone, que les dev branchent en input leur doc et modifient les paramètres dans des fichiers config seulement
- Extraction à partir de PDFs
- Extraction à partir d'images (à venir dans MAIA à priori)
- Quantifier la qualité des réponses : améliorer le listing des sources, voir création d'une base de tests ?

# SALOME Documentation RAG System

Advanced Retrieval-Augmented Generation (RAG) chatbot for SALOME platform documentation with multi-module support, intelligent chunking, and customizable response generation.

## Overview

This system extracts SALOME documentation (Doxygen API docs + Sphinx user guides), processes it into a vector database, and provides an intelligent chatbot interface for querying across multiple modules.

**Key Features:**
- Multi-module support (SHAPER, SMESH, GUI)
- Token-aware chunking (prevents embedding truncation)
- Quality scoring (filters low-value content)
- Code block extraction (enables code-aware retrieval)
- Dual interfaces (terminal + web)
- Customizable response styles
- Deep dive mode for complex questions

## Architecture

```
salome_docs_RAG/
├── extraction/              # Documentation processing pipeline
│   ├── process_multi_module_salome_docs.py
│   ├── config.example.json
│   └── README.md
│
├── chatbot/                 # RAG chatbot
│   ├── core/               # Business logic
│   │   ├── config.py
│   │   └── salome_chatbot.py
│   ├── ui/                 # User interfaces
│   │   ├── terminal.py
│   │   └── web.py
│   ├── chatbot.py          # Unified entry point
│   ├── config.example.json
│   └── README.md
│
└── requirements.txt        # All dependencies
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd salome_docs_RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (CPU-only PyTorch)
pip install -r requirements.txt
```

### 2. Extract Documentation

```bash
cd extraction

# Option A: Use config file (recommended)
cp config.example.json config.json
# Edit config.json to add your documentation paths
python process_multi_module_salome_docs.py --config config.json

# Option B: Command line arguments
python process_multi_module_salome_docs.py \
  --shaper-dir ./shaper_docs_extracted \
  --smesh-dir ./smesh_docs_extracted \
  --gui-dir ./gui_docs_extracted
```

**Expected output:**
- `salome_docs_extracted/chromadb/` - Vector database
- `salome_docs_extracted/salome_docs.json` - Processed chunks
- `salome_docs_extracted/statistics.json` - Extraction stats

### 3. Configure LLM Endpoint

The chatbot requires an OpenAI-compatible API endpoint. Configure it in `chatbot/config.json`:

```json
{
  "base_url": "http://localhost:8080/v1",
  "model_name": "mistral",
  "api_key": "your-api-key"
}
```

Supported endpoints:
- **OpenAI**: `https://api.openai.com/v1` with your API key
- **Mistral**: `https://api.mistral.ai/v1` with your API key
- **Local (Ollama)**: `http://localhost:11434/v1` with `api_key: "dummy"`
- **Any OpenAI-compatible endpoint**

### 4. Run Chatbot

```bash
cd chatbot

# Terminal interface (interactive)
python chatbot.py

# Web interface
python chatbot.py --web

# Single question
python chatbot.py --question "What is ModelAPI::Feature in SHAPER?"

# With filters
python chatbot.py --question "How to create a mesh?" --module SMESH --type user
```

## Features in Detail

### Extraction Pipeline

**Token-Aware Chunking:**
- Uses actual tokenizer (384 tokens max for all-MiniLM-L6-v2)
- Prevents embedding truncation
- Respects sentence boundaries

**Quality Scoring:**
```python
score = min(word_count/200, 1.0)
      + 0.1 if has_title
      + 0.1 if word_count > 50
      + 0.1 if word_count > 100

Filter: score >= 0.3
```

**Code Block Extraction:**
- Automatically extracts code snippets from `<pre>` and `<code>` tags
- Stored in metadata for code-aware retrieval
- Enables filtering by `has_code: true`

**Parent Document Tracking:**
- Every chunk knows its source document
- Enables hierarchical retrieval strategies
- Format: `parent_doc_id: "SHAPER:dev:FeatureAPI"`

See [extraction/README.md](extraction/README.md) for full details.

### Chatbot Features

**Multi-Module Support:**
- SHAPER (CAD modeling and geometry creation)
- SMESH (Mesh generation and manipulation)
- GUI (Graphical user interface components)
- Automatic module detection from database

**Dual Interfaces:**

1. **Terminal** - Interactive CLI with filters
   ```bash
   You [all]: What is ModelAPI::Feature?
   You [SHAPER] [dev]: Show me code examples
   ```

2. **Web** - Gradio interface with controls
   - Module and doc type filters
   - Response style presets
   - Search depth slider (20-80 chunks)
   - Answer length slider (500-4000 tokens)

**Response Styles:**

| Style | Temperature | Chunks | Deep Dive | Best For |
|-------|-------------|--------|-----------|----------|
| Precise (Recommended) | 0.0 | 40 | No | Technical queries, API lookups |
| Balanced | 0.2 | 50 | No | General questions, broader context |
| Comprehensive | 0.1 | 60 | Yes | Complex workflows, multi-part questions |

**Deep Dive Mode:**
- Retrieves 60 chunks (vs 40 standard)
- Batch summarization (10 chunks per batch)
- 7 LLM calls (6 summaries + 1 final answer)
- Best for complex, multi-faceted questions

See [chatbot/README.md](chatbot/README.md) for full details.

## Configuration

### Extraction Config (extraction/config.json)

```json
{
  "output_dir": "./salome_docs_extracted",
  "use_token_chunking": true,
  "modules": {
    "SHAPER": {
      "description": "CAD modeling and geometry creation",
      "dev_path": "./shaper_docs_extracted/html",
      "user_path": "./shaper_docs_extracted/html_gui"
    }
  },
  "chunking": {
    "max_tokens": 384,
    "overlap_tokens": 50
  },
  "quality": {
    "min_score": 0.3,
    "min_word_count": 50,
    "substantial_word_count": 100
  }
}
```

### Chatbot Config (chatbot/config.json)

```json
{
  "chromadb_path": "../extraction/salome_docs_extracted/chromadb",
  "base_url": "http://localhost:8080/v1",
  "model_name": "mistral",
  "api_key": "your-api-key",
  "embedding_model": "all-MiniLM-L6-v2",
  "k_standard": 40,
  "k_deep_dive": 60,
  "temperature": 0.0,
  "max_tokens": 2000
}
```

**Priority Order (highest to lowest):**
1. Web UI controls (search depth, answer length)
2. Response style presets (temperature, deep_dive)
3. config.json
4. Code defaults

## Usage Examples

### Terminal Mode

```bash
# Interactive mode
python chatbot.py

# Commands in interactive mode:
module:SHAPER      # Filter by module
type:dev           # Filter by doc type
deep               # Toggle deep dive mode
clear              # Clear all filters
stats              # Show database statistics
exit               # Exit

# Single question mode
python chatbot.py --question "How do I create a Feature in SHAPER?" \
  --module SHAPER --type dev

# Deep dive mode
python chatbot.py --question "Explain the complete mesh generation workflow" \
  --module SMESH --deep-dive
```

### Web Mode

```bash
# Launch web interface
python chatbot.py --web

# Custom port
python chatbot.py --web --port 8080

# Public URL via Gradio
python chatbot.py --web --share
```

**Web Interface Controls:**
- **Module Filter**: SHAPER, SMESH, GUI, All
- **Doc Type**: Dev (API), User (guides), All
- **Response Style**: Precise, Balanced, Comprehensive
- **Search Depth**: 20-80 chunks (slider)
- **Answer Length**: 500-4000 tokens (slider)

### Python API

```python
from chatbot.core import ChatbotConfig, SALOMEChatbot

# Initialize
config = ChatbotConfig.load("config.json")
chatbot = SALOMEChatbot(config)

# Basic query
result = chatbot.ask("What is ModelAPI::Feature?")
print(result['answer'])
print(result['sources'])

# With filters
result = chatbot.ask(
    "Show me mesh generation examples",
    module="SMESH",
    doc_type="user"
)

# Custom parameters (override config)
result = chatbot.ask(
    "Explain the workflow",
    deep_dive=True,
    k=70,              # Override chunk count
    temperature=0.1,   # Override temperature
    max_tokens=3000    # Override max length
)

# Get statistics
stats = chatbot.get_stats()
print(f"Total chunks: {stats['total_chunks']}")
print(f"Modules: {stats['available_modules']}")
```

## Technical Details

### Embedding Model

**all-MiniLM-L6-v2:**
- 384-dimensional embeddings
- 512 token max sequence length
- Fast CPU inference (~2000 chunks/min)
- Good quality for technical documentation

### Chunking Strategy

**Token-aware mode** (recommended):
```
Max: 384 tokens (safe for 512 limit)
Overlap: 50 tokens (~13%)
Respects sentence boundaries
```

**Fallback mode** (if transformers not installed):
```
Max: 1000 characters
Overlap: 200 characters
```

### Vector Database

**ChromaDB with:**
- Cosine similarity metric
- Explicit embedding function (no silent mismatches)
- Persistent storage
- Metadata filtering support

### LLM Integration

**LangChain with OpenAI-compatible APIs:**
- Works with any OpenAI-compatible endpoint
- Supports: OpenAI, Mistral, Anthropic, Ollama, local models
- Default: Mistral
- Easy to swap providers (change base_url + api_key)

## Troubleshooting

### Extraction Issues

**"No HTML files found"**
```bash
# Check directory structure
ls -R extraction/
# Should have: module_docs_extracted/html and module_docs_extracted/html_gui
```

**"transformers not installed"**
```bash
pip install transformers
# Falls back to character-based chunking (still works, less optimal)
```

**Low chunk count**
- Check `salome_docs_extracted/statistics.json`
- Adjust `min_score` in config.json (default 0.3)
- Verify HTML files are valid

### Chatbot Issues

**"ChromaDB not found"**
```bash
# Run extraction first
cd extraction && python process_multi_module_salome_docs.py
```

**"Connection refused" (LLM endpoint)**
```bash
# Check if your LLM endpoint is running
curl http://localhost:8080/v1/models

# For Ollama:
ollama serve

# Verify base_url and api_key in config.json
```

**Web interface not loading**
```bash
# Check port availability
python chatbot.py --web --port 7861

# Check firewall settings
# Open http://localhost:7860 in browser
```

## Adding New Modules

### Via Config File (Recommended)

Edit `extraction/config.json`:
```json
{
  "modules": {
    "GEOM": {
      "description": "Geometry module",
      "url": "https://docs.salome-platform.org/latest/tui/GEOM",
      "dev_path": "./geom_docs/html",
      "user_path": "./geom_docs/html_gui"
    }
  }
}
```

### Via Command Line

```bash
python process_multi_module_salome_docs.py \
  --geom-dev ./geom_docs/html \
  --geom-user ./geom_docs/html_gui
```

Chatbot will automatically detect new modules on next startup.

## Dependencies

**Core:**
- Python 3.8+
- beautifulsoup4 (HTML parsing)
- chromadb (vector database)
- sentence-transformers (embeddings)
- langchain (RAG framework)
- gradio (web interface)

**Optional:**
- transformers (token-aware chunking - recommended)

**LLM Access:**
- Any OpenAI-compatible endpoint (OpenAI, Mistral, Ollama, etc.)

See [requirements.txt](requirements.txt) for exact versions.

## Project Structure

```
salome_docs_RAG/
│
├── extraction/                      # Documentation extraction
│   ├── process_multi_module_salome_docs.py
│   ├── config.example.json
│   ├── README.md
│   └── salome_docs_extracted/      # Generated output (gitignored)
│       ├── chromadb/
│       ├── salome_docs.json
│       └── statistics.json
│
├── chatbot/                         # RAG chatbot
│   ├── core/                       # Business logic
│   │   ├── __init__.py
│   │   ├── config.py               # Configuration management
│   │   └── salome_chatbot.py       # RAG logic
│   ├── ui/                         # User interfaces
│   │   ├── __init__.py
│   │   ├── terminal.py             # CLI interface
│   │   └── web.py                  # Gradio web UI
│   ├── _old/                       # Legacy code (archived)
│   ├── chatbot.py                  # Unified entry point
│   ├── config.example.json
│   └── README.md
│
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md                        # This file
```

## Development

### Architecture Principles

**Separation of Concerns:**
- `core/` - Pure business logic (no UI)
- `ui/` - Interface layers (terminal, web)
- `chatbot.py` - Entry point and routing

**Configuration Priority:**
```
Runtime parameters > Response styles > config.json > defaults
```

**Extensibility:**
- Easy to add new modules (via config)
- Easy to add new interfaces (implement UI class)
- Easy to swap LLMs (change base_url + api_key)

### Testing

```bash
# Compile check
python -m py_compile chatbot/core/*.py chatbot/ui/*.py

# Test extraction
cd extraction && python process_multi_module_salome_docs.py --help

# Test chatbot
cd chatbot && python chatbot.py --help
```

---

Built for the SALOME platform documentation system.

**Technologies:**
- LangChain for RAG orchestration
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Gradio for web interface

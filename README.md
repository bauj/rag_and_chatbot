# Documentation RAG Chatbot

A chatbot for any project with HTML documentation (Sphinx, Doxygen). Point it at your docs, configure a few JSON fields, and get an intelligent chatbot with multi-module support, dual retrieval modes, and dual interfaces (terminal + web).

## Overview

**Workflow:**
1. **Extract** — parse your HTML docs into a ChromaDB vector database (and a page index for agentic mode)
2. **Chat** — query the docs through a terminal or web interface, in RAG or Agentic mode

**Key Features:**
- Works with any HTML documentation (Sphinx, Doxygen, or custom)
- Multi-module support — query across multiple doc sets simultaneously
- Three chatbot modes: **RAG** (vector retrieval + reranking), **Agentic** (fixed-round page-browsing pipeline), and **Agentic-smol** (smolagents `CodeAgent`, real multi-hop browsing — opt-in, experimental)
- Token-aware chunking (prevents embedding truncation)
- Cross-encoder reranking (`BAAI/bge-reranker-v2-m3`) for better result ranking
- Optional BM25 keyword hybrid retrieval (opt-in, fused with vector search via Reciprocal Rank Fusion)
- Optional HyDE retrieval (opt-in — an LLM-written hypothetical passage steers the vector search instead of the raw question)
- Quality scoring (filters low-value content)
- Code block extraction (enables code-aware retrieval, preserved as Markdown)
- Per-symbol Doxygen chunking (one chunk per documented class/method, not just per section)
- Dual interfaces (terminal + Gradio web)
- Configurable LLM and embedding endpoints (local or remote)

## Architecture

```
rag_and_chatbot/
├── extraction/              # Documentation processing pipeline
│   ├── process_docs.py      # Main extractor (ChromaDB + page_index.json)
│   ├── html_parser.py       # Shared HTML parsing utilities
│   ├── config.example.json
│   └── README.md
│
├── chatbot/                 # Chatbot (RAG, Agentic, and Agentic-smol modes)
│   ├── core/
│   │   ├── config.py        # Configuration (ChatbotConfig, AgenticConfig, ...)
│   │   ├── rag_chatbot.py   # RAG mode (DocumentationChatbot) — BM25 + HyDE live here too
│   │   ├── bm25_index.py    # BM25 keyword index + Reciprocal Rank Fusion
│   │   ├── agentic_chatbot.py       # Agentic mode (AgenticChatbot) — fixed 1-2 round script
│   │   └── agentic_smol_chatbot.py  # Agentic-smol mode (AgenticSmolChatbot) — smolagents CodeAgent
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

### 2. Download Local Models

If you are using local embeddings or reranking (the default), download the models once:

```bash
python download_models.py
```

This downloads to `~/.cache/huggingface/hub`:
- `all-MiniLM-L6-v2` — embedding model (required for local embeddings)
- `BAAI/bge-reranker-v2-m3` — reranker model (required if `reranker` is enabled in chatbot config)

> **Offline use:** the chatbot runs with `HF_HUB_OFFLINE=1` by default, so models must be downloaded before first use. Skip this step only if you are using API-based embeddings and no reranker.

### 3. Extract Documentation

```bash
cd extraction
cp config.example.json config.json
# Edit config.json: set project_name, output_dir, and module paths
python process_docs.py --config config.json
```

**Expected output** (in `output_dir`):
- `chromadb/` — vector database (used by RAG mode)
- `page_index.json` — page catalogue (used by Agentic mode)
- `{project_name}_docs.json` / `{project_name}_docs.jsonl` — processed chunks
- `statistics.json` — extraction stats

### 4. Configure the Chatbot

```bash
cd ../chatbot
cp config.example.json config.json
# Edit config.json: set project_name, chromadb_path, llm endpoint
```

### 5. Run the Chatbot

```bash
# Interactive terminal (RAG mode, default)
python chatbot.py

# Agentic mode (reads HTML pages directly, no vector retrieval)
python chatbot.py --mode agentic

# Agentic-smol mode (smolagents CodeAgent, real multi-hop browsing — opt-in, see Configuration)
python chatbot.py --mode agentic-smol

# Web interface (Gradio) — all configured modes available via UI toggle
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
      "user_path": "./module_a/html_gui",
      "url_for_sources_citation": "https://docs.myproject.org/module_a/html"
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
- `url_for_sources_citation` — optional base URL of the directory containing the HTML files, no trailing slash. Source links become `base_url/filename.html`. To find the right value, open any page of the online docs and remove the filename: e.g. `https://docs.salome-platform.org/latest/tui/SHAPER/classModelAPI__Feature.html` → `https://docs.salome-platform.org/latest/tui/SHAPER`. When omitted, sources show the local filename only.
- Both paths are optional per module; omit either if not applicable
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
  },

  "agentic": {
    "page_index_path": "../extraction/my_project_docs_extracted/page_index.json",
    "max_chars_per_page": 8000,
    "max_pages_per_round": 3,
    "max_pages_round2": 2,
    "max_steps": 6
  },

  "bm25_enabled": false,
  "hyde_enabled": false,
  "smol_enabled": false
}
```

**LLM endpoints:**

| Provider | `base_url` | `api_key` |
|---|---|---|
| Ollama (local) | `http://localhost:11434/v1` | `"dummy"` |
| OpenAI | `https://api.openai.com/v1` | your key |
| Mistral | `https://api.mistral.ai/v1` | your key |
| Any OpenAI-compatible | your endpoint | your key |

**Reranker:** set to `null` or remove the block to disable reranking (faster, lower quality).

**Agentic mode:** set `agentic` to `null` or remove the block to disable. When enabled, the web UI and `--mode agentic` CLI flag become available. Requires `page_index.json` produced by the extractor.

**Agentic-smol mode:** requires both the `agentic` block (same page index, reused) **and** `smol_enabled: true`. Uses [smolagents](https://github.com/huggingface/smolagents)' `CodeAgent` instead of the classic fixed 1-2 round script, so the LLM decides how many search/read cycles to run, bounded by `agentic.max_steps` (default `6`). Opt-in and experimental — not yet measured against classic Agentic mode. Requires the `smolagents` package (`pip install smolagents`, included in `requirements.txt`).

**BM25 hybrid retrieval:** set `bm25_enabled: true` to fuse keyword search (BM25) with vector search via Reciprocal Rank Fusion, in RAG mode's standard (non-deep-dive) path. Requires `{project_name}_docs.jsonl` next to `chromadb_path` (produced by extraction). Opt-in — not yet measured, off by default.

**HyDE retrieval:** set `hyde_enabled: true` to have an LLM write a short hypothetical documentation passage per question and embed *that* (instead of the raw question) for the vector search — helps when questions are phrased very differently from how the docs word things. Costs one extra LLM call per question; BM25 and reranking still use the real question. Opt-in, off by default.

**Runtime toggles:** BM25, HyDE and the reranker each have a web-UI checkbox and a terminal command (`bm25`, `hyde`, `reranker`). All three are gated the same way — the config flag must be `true` at startup for the feature to load, and the toggle can then only turn it **off**. Every answer reports which stages actually ran.

**Deep Dive caveat:** Deep Dive mode runs its own retrieve-and-summarize pipeline that **bypasses BM25, HyDE and reranking entirely.** The web UI hides those controls while Deep Dive is on and the terminal warns when you enable it, so the bypass is never silent.

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

Three modes are available and can be selected per-question from the web UI or via `--mode` in the CLI (Agentic-smol requires `smol_enabled: true`).

**RAG mode (default):**
```
[HyDE, if enabled: LLM writes a hypothetical passage from the question]
        ↓
Vector retrieval (large K), on the HyDE passage if enabled, else the raw question
        ↓
[BM25 keyword search, if enabled — fused with vector results via Reciprocal Rank Fusion]
        ↓
Cross-encoder reranking (scores against the real question, never the HyDE passage)
        ↓
LLM answer generation
```

**Agentic mode:**
```
Keyword search on page index → candidate pages
        ↓
LLM selects most relevant pages (up to max_pages_per_round)
by reading each candidate's title, module, and doc_category
        ↓
Selected HTML pages read + parsed (Round 1)
        ↓
LLM answers — appends NEED_MORE_INFO:<gap> if insufficient
        ↓ (only if NEED_MORE_INFO)
Keyword search using the identified gap → new candidates
        ↓
LLM selects more pages (up to max_pages_round2)
        ↓
All pages read + parsed (Round 2)
        ↓
LLM final answer
```

Agentic mode requires no ChromaDB at query time — it reads the original HTML files directly via the `page_index.json` catalogue produced during extraction.

**Agentic-smol mode** (opt-in, `smol_enabled: true`):
```
CodeAgent (smolagents) given two tools: search_pages, read_page
        ↓
Agent decides for itself how many search → read cycles to run,
bounded by agentic.max_steps (default 6) — real multi-hop browsing,
e.g. can follow a cross-reference found mid-read
        ↓
LLM final answer
```
Same page index as classic Agentic mode, no ChromaDB required. `read_page` only accepts filepaths present in the page index (rejects anything else) — the LLM cannot read arbitrary files off disk. Ships as a separate mode, not a replacement for classic Agentic — not yet measured against it.

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
  mode:rag           - Switch to RAG mode (vector retrieval)
  mode:agentic       - Switch to Agentic mode (reads HTML pages directly)
  mode:agentic-smol  - Switch to Agentic-smol mode (smolagents, multi-hop browsing)
  module:MODULE_A    - Filter by module (RAG only)
  type:dev           - Filter developer docs only (RAG only)
  type:user          - Filter user docs only (RAG only)
  deep               - Toggle Deep Dive mode (RAG only)
  reranker           - Toggle cross-encoder reranker on/off (RAG only)
  topn:<n>           - Set top-N docs kept after rerank (RAG only)
  hyde               - Toggle HyDE on/off (RAG only)
  agentic:chars:<n>  - Set max chars read per page (classic Agentic only)
  agentic:pages1:<n> - Set pages read in round 1 (classic Agentic only)
  agentic:pages2:<n> - Set pages read in round 2 (classic Agentic only)
  clear              - Clear all filters
  stats              - Show database statistics (RAG only)
  exit               - Exit
```

### Python API

Run from inside the `chatbot/` directory:

```python
from core import ChatbotConfig, DocumentationChatbot, AgenticChatbot, AgenticSmolChatbot

config = ChatbotConfig.load("config.json")

# --- RAG mode ---
chatbot = DocumentationChatbot(config)

result = chatbot.ask("How do I create a mesh?")
print(result['answer'])
print(result['sources'])

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
    max_tokens=3000
)

result = chatbot.ask(
    "How do I create a mesh?",
    reranker_enabled=True,  # disable at runtime to skip reranking
    top_n=10,               # override config.top_n_after_rerank
    hyde_enabled=True,      # override config.hyde_enabled (requires hyde_enabled: true at startup)
)

stats = chatbot.get_stats()
print(f"Modules: {stats['available_modules']}")

# --- Agentic mode (requires config.agentic to be set) ---
agentic = AgenticChatbot(config)

result = agentic.ask("How do I create a mesh?")
print(result['answer'])
print(result['filters']['rounds_used'])  # 1 or 2
for source in result['sources']:
    print(f"  - {source['title']} ({source['module']}/{source['doc_category']})")

# --- Agentic-smol mode (requires config.agentic AND config.smol_enabled to be set) ---
agentic_smol = AgenticSmolChatbot(config)

result = agentic_smol.ask("How do I create a mesh?", max_steps=8)  # override config.agentic.max_steps
print(result['answer'])
print(result['filters']['steps_used'])
print(result['filters']['grounded'])  # False if the agent answered without reading any page
```

All three `.ask()` methods return the same dict shape: `{answer, sources, filters, error}`.

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
    "dev_path": "./new_module/html",
    "url_for_sources_citation": "https://docs.myproject.org/new_module/html"
  }
}
```

The chatbot detects available modules automatically from the database at startup.

## Dependencies

- Python 3.8+
- `beautifulsoup4` — HTML parsing
- `markdownify` — HTML → Markdown (code block preservation)
- `chromadb` — vector database
- `sentence-transformers` — embeddings and reranking
- `langchain` — RAG framework
- `rank_bm25` — BM25 keyword retrieval (optional, only used if `bm25_enabled: true`)
- `smolagents` — Agentic-smol mode (optional, only used if `smol_enabled: true`)
- `gradio` — web interface
- `transformers` — token-aware chunking (recommended)

Any OpenAI-compatible LLM endpoint (OpenAI, Mistral, Ollama, local models).

See [requirements.txt](requirements.txt) for exact versions.

## RAG vs Agentic vs Agentic-smol — When to Use Which

| | RAG (no reranker) | RAG (+ reranker) | Agentic | Agentic-smol |
|---|---|---|---|---|
| **Retrieval** | Vector similarity search | Vector search + cross-encoder (+ optional BM25/HyDE) | Keyword search + LLM page selection | Keyword search, agent-directed |
| **Context** | Chunks (sub-page fragments) | Chunks, re-scored and expanded | Full pages (up to `max_chars_per_page`) | Full pages, read across as many hops as the agent decides |
| **LLM calls** | 1 (+1 if HyDE enabled) | 1 (+1 if HyDE enabled) | 2–4 (selection + answer, ×2 rounds, fixed) | Variable, bounded by `max_steps` (default 6) |
| **Local inference** | Embeddings only | Embeddings + reranker (slow) | None beyond the LLM | None beyond the LLM |
| **Latency** | Fast | Can be slower than Agentic | Moderate | Variable, less predictable than classic Agentic |
| **Maturity** | Established | Established | Established | Experimental, opt-in, not yet measured against classic Agentic |

**Prefer RAG when:**
- Your questions target specific facts buried inside long pages (chunk-level retrieval wins)
- You want module/type filtering
- You don't need the reranker and want the lowest latency

**Prefer Agentic when:**
- Your questions need the full context of a page (not just a chunk)
- You want more transparent sourcing — the LLM explicitly chooses which pages to read
- You don't want to maintain a ChromaDB (lighter setup for quick experiments)
- RAG retrieval returns irrelevant chunks (e.g. poor embedding alignment with your docs)
- You want a predictable, bounded cost shape (fixed ≤4 LLM calls)

**Consider Agentic-smol when:**
- Your questions need real multi-hop browsing (e.g. read a class page, follow a cross-referenced method) — structurally impossible in classic Agentic mode's fixed round count
- You're comfortable with less predictable latency/cost and an experimental, unmeasured mode

# Documentation RAG Chatbot

A chatbot for any project with HTML documentation (Sphinx, Doxygen). Point it at your docs, configure a few JSON fields, and get an intelligent chatbot with multi-module support, dual retrieval modes, and dual interfaces (terminal + web).

## Overview

**Workflow:**
1. **Extract** — parse your HTML docs into a ChromaDB vector database (and a page index for agentic mode)
2. **Chat** — query the docs through a terminal or web interface, in RAG or Agentic mode

**Key Features:**
- Works with any HTML documentation (Sphinx, Doxygen, or custom)
- Multi-module support — query across multiple doc sets simultaneously
- Two chatbot modes: **RAG** (vector retrieval + reranking) and **Agentic** (deepagents harness, multi-hop page browsing)
- Token-aware chunking (prevents embedding truncation)
- Reranking for better result ranking — cross-encoder (`BAAI/bge-reranker-v2-m3`, default) or opt-in late-interaction/ColBERT-style (`answerdotai/answerai-colbert-small-v1`, lower latency)
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
├── chatbot/                 # Chatbot (RAG and Agentic modes)
│   ├── core/
│   │   ├── config.py        # Configuration (ChatbotConfig, AgenticConfig, ...)
│   │   ├── rag_chatbot.py   # RAG mode (DocumentationChatbot) — BM25 + HyDE live here too
│   │   ├── bm25_index.py    # BM25 keyword index + Reciprocal Rank Fusion
│   │   └── agentic_chatbot.py       # Agentic mode (AgenticChatbot) — deepagents harness
│   ├── ui/
│   │   ├── terminal.py      # CLI interface
│   │   └── web.py           # Gradio web UI
│   ├── chatbot.py           # Unified entry point
│   └── config.example.json
│
├── evaluator/               # Benchmark & evaluation suite
│   ├── base/                # Base evaluator module
│   │   ├── evaluator.py     # Core evaluation metrics
│   │   ├── config.json      # Evaluator configuration
│   │   ├── dataset.json     # Test questions with tags
│   │   └── README.md
│   │
│   └── benchmark/           # Benchmark orchestration
│       ├── run_benchmark.py # CLI entry point for benchmarks
│       ├── benchmark.py     # Benchmark execution engine
│       ├── analyze_results.py # Results analysis & graphing
│       ├── benchmark_config.json # Hyperparameter configurations
│       └── README.md
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
- `BAAI/bge-reranker-v2-m3` — cross-encoder reranker model (required if `reranker.type: "cross_encoder"` is enabled in chatbot config)
- `answerdotai/answerai-colbert-small-v1` — late-interaction reranker model (required if `reranker.type: "late_interaction"` is enabled instead)

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

# Agentic mode (deepagents harness, multi-hop page browsing, see Configuration)
python chatbot.py --mode agentic

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
    "type": "cross_encoder"
  },

  "agentic": {
    "page_index_path": "../extraction/my_project_docs_extracted/page_index.json",
    "max_chars_per_page": 8000,
    "max_steps": 6
  },

  "bm25_enabled": false,
  "hyde_enabled": false
}
```

**LLM endpoints:**

| Provider | `base_url` | `api_key` |
|---|---|---|
| Ollama (local) | `http://localhost:11434/v1` | `"dummy"` |
| OpenAI | `https://api.openai.com/v1` | your key |
| Mistral | `https://api.mistral.ai/v1` | your key |
| Any OpenAI-compatible | your endpoint | your key |

**Reranker:** set to `null` or remove the block to disable reranking (faster, lower quality). Two `type`s: `cross_encoder` (default, `BAAI/bge-reranker-v2-m3`) or `late_interaction` (ColBERT-style MaxSim scoring via `sentence-transformers` `MultiVectorEncoder`, e.g. `answerdotai/answerai-colbert-small-v1` — requires `sentence-transformers >= 6.0`). Measured on the SHAPER eval set: quality is a wash between the two, `late_interaction` cuts average answer latency roughly in half (see `notes/late_interaction_reranker_eval_230826.md`).

**Agentic mode:** set `agentic` to `null` or remove the block to disable. When enabled, the web UI and `--mode agentic` CLI flag become available. Requires `page_index.json` produced by the extractor, and the `deepagents` package (`pip install deepagents`, included in `requirements.txt`). Uses LangChain's [deepagents](https://github.com/langchain-ai/deepagents) tool-calling loop: the LLM decides for itself how many search/read cycles to run, bounded by `agentic.max_steps` (default `6`, mapped to the LangGraph recursion limit).

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

Two modes are available and can be selected per-question from the web UI or via `--mode` in the CLI.

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
deepagents agent given two tools: search_pages, read_page
        ↓
Agent decides for itself how many search → read cycles to run,
bounded by agentic.max_steps (default 6) — real multi-hop browsing,
e.g. can follow a cross-reference found mid-read
        ↓
LLM final answer
```

Agentic mode requires no ChromaDB at query time — it reads the original HTML files directly via the `page_index.json` catalogue produced during extraction. `read_page` only accepts filepaths present in the page index (rejects anything else) — the LLM cannot read arbitrary files off disk.

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
  mode:rag             - Switch to RAG mode (vector retrieval)
  mode:agentic         - Switch to Agentic mode (deepagents, multi-hop browsing)
  module:MODULE_A      - Filter by module (RAG only)
  type:dev             - Filter developer docs only (RAG only)
  type:user            - Filter user docs only (RAG only)
  deep                 - Toggle Deep Dive mode (RAG only)
  reranker             - Toggle configured reranker on/off (RAG only)
  topn:<n>             - Set top-N docs kept after rerank (RAG only)
  hyde                 - Toggle HyDE on/off (RAG only)
  agentic:chars:<n>    - Set max chars read per page (Agentic only)
  agentic:maxsteps:<n> - Set the agent step budget (Agentic only)
  clear                - Clear all filters
  stats                - Show database statistics (RAG only)
  exit                 - Exit
```

### Python API

Run from inside the `chatbot/` directory:

```python
from core import ChatbotConfig, DocumentationChatbot, AgenticChatbot

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

result = agentic.ask("How do I create a mesh?", max_steps=8)  # override config.agentic.max_steps
print(result['answer'])
print(result['filters']['steps_used'])
print(result['filters']['grounded'])  # False if the agent answered without reading any page
for source in result['sources']:
    print(f"  - {source['title']} ({source['module']}/{source['doc_category']})")
```

Both `.ask()` methods return the same dict shape: `{answer, sources, filters, error}`.

## Evaluation & Benchmarking

### Overview

The evaluator suite provides comprehensive testing and analysis of RAG and Agentic chatbot modes. It includes:

- **Base Evaluator** — Core evaluation metrics (Correctness, Relevance, Groundedness, Retrieval Relevance)
- **Benchmark Suite** — Automated testing across multiple hyperparameter configurations
- **Analysis Tools** — Detailed results analysis with graphs and per-question/tag breakdowns

### Step 1: Configure the Base Evaluator

```bash
cd evaluator/base

# Copy configuration template
cp config.example.json config.json

# Edit config.json to set:
# - chatbot_path: relative path to your chatbot directory (e.g., "../../chatbot")
```

The base evaluator configuration defines:
- `chatbot_path` — Path to the chatbot directory for running evaluations
- `evaluator_options` — Additional LLM settings for scoring

### Step 2: Configure the Test Dataset

The test questions are defined in `evaluator/base/dataset.json`. Each question can include tags for categorization:

```json
[
  {
    "inputs": {"question": "How do I create a mesh?"},
    "outputs": {"answer": "Expected answer or reference"},
    "tags": ["code", "tutorial", "mesh"]
  }
]
```

Edit this file to add or modify test questions. Tags help group results by theme in the analysis.

### Step 3: Run Base Evaluator

Evaluate all questions in the dataset directly:

```bash
cd evaluator/base
python evaluator.py
```

This:
- Loads all questions from `dataset.json`
- Runs the chatbot in both RAG or Agentic mode accordingly to the config file
- Scores each answer using the evaluation metrics
- Saves results to `evaluation_results.json`

### Evaluation Metrics

All evaluations use LLM-based scoring (0-10 scale):

- **Correctness** — Factual accuracy of the answer
- **Relevance** — Overall relevance to the question
- **Groundedness** — Answers grounded in retrieved documents (RAG only)
- **Retrieval Relevance** — Quality of retrieved documents (RAG only)

Each metric includes:
- `score` — Numeric score (0-10)
- `explanation` — Reasoning behind the score

### Base Evaluator Results

Results are saved to `evaluation_results.json` with structure:

```json
[
  {
    "question": "How do I create a mesh?",
    "mode": "rag",
    "generated_answer": "You can create a mesh by...",
    "evaluations": {
      "correctness": {"score": 8.5, "explanation": "..."},
      "relevance": {"score": 9.0, "explanation": "..."}
    }
  }
]
```

---

### Step 4: Configure Benchmark Hyperparameters

For systematic testing across multiple configurations:

```bash
cd evaluator/benchmark

# Copy benchmark configuration template
cp benchmark_config.example.json benchmark_config.json
```

Edit `benchmark_config.json` to define which modes and hyperparameters to test — every combination of the values below is run (a full cross-product), so each extra value multiplies the total run count:

```json
{
  "modes": ["rag", "agentic"],
  "rag_hyperparams": {
    "k": [3, 5, 10],
    "temperature": [0.3, 0.7],
    "reranker_enabled": [true, false],
    "top_n": [5, 10],
    "deep_dive": [false],
    "hyde_enabled": [false],
    "bm25_enabled": [true, false],
    "title_boost_enabled": [true]
  },
  "agentic_hyperparams": {
    "temperature": [0.7],
    "max_steps": [4, 6, 8],
    "max_chars_per_page": [8000]
  }
}
```

`hyde_enabled`, `bm25_enabled` and `title_boost_enabled` are RAG retrieval-pipeline toggles (see `chatbot/config.json`'s comments for what each does); `max_steps` is the agentic mode's agent step budget, and `max_chars_per_page` is how much of each page it reads. With the values above, this generates **48 RAG configurations** and **3 Agentic configurations** (all combinations), each tested against every question — prefer varying one or two dimensions at a time to keep run counts manageable.

### Step 5: Run Benchmarks

```bash
cd evaluator/benchmark

# Validate setup
python3 benchmark.py --help

# Quick test (2 questions)
python3 benchmark.py --limit 2

# Full test (all questions)
python3 benchmark.py

# Compare with previous results
python3 benchmark.py --compare
```

Which modes run is controlled by `benchmark_config.json`'s `"modes"` key (above), not a CLI flag.

### Benchmark Results Format

Benchmark results are saved to `benchmark_results/benchmark_YYYYMMDD_HHMMSS.json` with structure:

```json
{
  "metadata": {
    "timestamp": "20260407_120000",
    "benchmark_config": { "modes": ["rag", "agentic"], "rag_hyperparams": {...}, "agentic_hyperparams": {...} },
    "total_questions": 13,
    "workers": 1,
    "timeout_seconds": null
  },
  "results": [
    {
      "mode": "rag",
      "configuration": {"k": 5, "temperature": 0.3, "reranker_enabled": true, "top_n": 10},
      "average_scores": {"correctness": 7.2, "relevance": 7.5, ...},
      "average_request_time": 24.658,
      "questions": [
        {
          "question_id": 1,
          "question": "How do I create a mesh?",
          "tags": ["Py", "Sampler", "easy", "En"],
          "evaluations": {"correctness": {"score": 8.0}, ...},
          "request_time": 24.658
        }
      ]
    }
  ]
}
```

### Step 6: Analyze Results and Generate Graphs

```bash
cd evaluator/benchmark

# Analyze latest results
python3 analyze_results.py --latest

# Generate visualization graphs
python3 analyze_results.py --latest --graphs

# List all benchmark results
python3 analyze_results.py --list

# Analyze specific file
python3 analyze_results.py benchmark_results/benchmark_20260407_120000.json
```

The analysis tool generates:

- **Summary table** — Average scores for all configurations
- **Per-question analysis** — Best/worst configurations per question
- **Mode comparison** — RAG vs Agentic across all metrics
- **Metric statistics** — Min/max/average per metric
- **Graphs** (with `--graphs` flag):
  - Scores by question (bar chart comparing all metrics)
  - Metric distributions (histograms showing score spread)
  - Scores by tag (grouped bar chart showing performance by question theme)

Graphs are saved in the `graphs/` folder.



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
- `deepagents` — Agentic mode (required if `agentic` block is configured)
- `gradio` — web interface
- `transformers` — token-aware chunking (recommended)

Any OpenAI-compatible LLM endpoint (OpenAI, Mistral, Ollama, local models).

See [requirements.txt](requirements.txt) for exact versions.

## RAG vs Agentic — When to Use Which

| | RAG (no reranker) | RAG (+ reranker) | Agentic |
|---|---|---|---|
| **Retrieval** | Vector similarity search | Vector search + reranker (+ optional BM25/HyDE) | Keyword search, agent-directed |
| **Context** | Chunks (sub-page fragments) | Chunks, re-scored and expanded | Full pages, read across as many hops as the agent decides |
| **LLM calls** | 1 (+1 if HyDE enabled) | 1 (+1 if HyDE enabled) | Variable, bounded by `max_steps` (default 6) |
| **Local inference** | Embeddings only | Embeddings + reranker (slow with `cross_encoder`, ~2x faster with `late_interaction`) | None beyond the LLM |
| **Latency** | Fast | Can be slower than Agentic | Variable, less predictable than RAG |
| **Maturity** | Established | Established | Established |

**Prefer RAG when:**
- Your questions target specific facts buried inside long pages (chunk-level retrieval wins)
- You want module/type filtering
- You don't need the reranker and want the lowest latency

**Prefer Agentic when:**
- Your questions need real multi-hop browsing (e.g. read a class page, follow a cross-referenced method)
- You want more transparent sourcing — the LLM explicitly chooses which pages to read
- You don't want to maintain a ChromaDB (lighter setup for quick experiments)
- RAG retrieval returns irrelevant chunks (e.g. poor embedding alignment with your docs)
- You're comfortable with less predictable latency/cost than RAG's fixed shape

# Documentation Chatbot

Chatbot with three retrieval modes and clean separation between business logic and UI.

- **RAG mode** — queries a ChromaDB vector database, with optional cross-encoder reranking, optional BM25 keyword hybrid retrieval, and optional HyDE
- **Agentic mode** — searches a page index, asks the LLM to select pages, reads HTML files directly, fixed 1-2 round script; no vector retrieval at query time
- **Agentic-smol mode** (opt-in, experimental) — same page index as Agentic mode, but driven by a smolagents `CodeAgent` that decides for itself how many search/read cycles to run instead of a fixed round count

## Architecture

```
chatbot/
├── core/
│   ├── config.py                    # ChatbotConfig, LLMConfig, EmbeddingConfig, RerankerConfig, AgenticConfig
│   ├── rag_chatbot.py               # DocumentationChatbot — RAG mode (+ BM25 hybrid, HyDE)
│   ├── bm25_index.py                # BM25Index + reciprocal_rank_fusion
│   ├── agentic_chatbot.py           # AgenticChatbot — agentic mode (fixed 1-2 round script)
│   └── agentic_smol_chatbot.py      # AgenticSmolChatbot — agentic-smol mode (smolagents CodeAgent)
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
| `k_standard` | `40` | Chunks retrieved in standard mode (RAG). Web UI search-depth slider defaults to this. |
| `k_deep_dive` | `60` | Chunks retrieved in deep dive mode (RAG). Web UI search-depth slider switches to this when Deep Dive is enabled. |
| `top_n_after_rerank` | `15` | Docs kept after cross-encoder reranking. Web UI top-N slider defaults to this. |
| `temperature` | `0.0` | LLM temperature. Used directly in the terminal; in the web UI, select the "Config default" response style to apply it (the other styles override it). |
| `max_tokens` | `2000` | Max tokens per response. Web UI answer-length slider defaults to this. |
| `bm25_enabled` | `false` | Opt-in BM25 keyword search fused with vector search via Reciprocal Rank Fusion (RAG standard mode only). Requires `{project_name}_docs.jsonl` next to `chromadb_path`. |
| `hyde_enabled` | `false` | Opt-in HyDE — an LLM writes a hypothetical passage per question, embedded instead of the raw question for vector search. BM25/reranking still use the real question. One extra LLM call per question. |
| `smol_enabled` | `false` | Opt-in Agentic-smol mode. Requires `agentic` block to also be set (same page index). Requires the `smolagents` package. |

Config is loaded in this priority order: CLI arguments > `config.json` > defaults.

**`embedding.model` must be identical to `extraction/config.json`** — mismatch causes wrong retrieval with no error.

### Agentic config block

```json
"agentic": {
  "page_index_path": "../extraction/my_project_docs_extracted/page_index.json",
  "max_chars_per_page": 8000,
  "max_pages_per_round": 3,
  "max_pages_round2": 2,
  "max_steps": 6
}
```

| Field | Default | Description |
|---|---|---|
| `page_index_path` | required | Path to `page_index.json` (relative to `chatbot/` or absolute) |
| `max_chars_per_page` | `8000` | Max characters read per HTML page |
| `max_pages_per_round` | `3` | Pages selected and read in Round 1 (classic Agentic mode) |
| `max_pages_round2` | `2` | Additional pages read in Round 2 (classic Agentic mode, if NEED_MORE_INFO triggered) |
| `max_steps` | `6` | CodeAgent step budget (Agentic-smol mode only, requires `smol_enabled: true`) |

`page_index.json` is generated automatically when running `extraction/process_docs.py`. This same block is shared by both classic Agentic and Agentic-smol modes — `max_chars_per_page` in particular is reused as-is by both, even though Agentic-smol's step loop resends read pages on every iteration, so its effective context cost grows faster with this value than classic Agentic mode's single read-and-answer shape. Consider a lower value if running Agentic-smol.

## Usage

### Terminal Interface

```bash
# Interactive mode (RAG, default)
python chatbot.py

# Agentic mode
python chatbot.py --mode agentic

# Agentic-smol mode (requires config.agentic AND smol_enabled: true)
python chatbot.py --mode agentic-smol

# Single question
python chatbot.py --question "How do I create a mesh?"

# With filters (RAG mode only)
python chatbot.py --question "How do I create a mesh?" --module MODULE_A --type user

# Deep dive mode (RAG mode only)
python chatbot.py --question "Explain the full workflow" --deep-dive
```

Commands in interactive mode:
```
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
bm25               - Toggle BM25 hybrid retrieval on/off (RAG only)
agentic:chars:<n>  - Set max chars read per page (classic Agentic only)
agentic:pages1:<n> - Set pages read in round 1 (classic Agentic only)
agentic:pages2:<n> - Set pages read in round 2 (classic Agentic only)
clear              - Clear all filters
stats              - Show database statistics (RAG only)
exit               - Exit
```

`mode:agentic` is only available when `config.agentic` is set. `mode:agentic-smol` additionally requires `smol_enabled: true`. `reranker`/`topn:<n>` are only shown when a reranker model is configured. `hyde` is only shown when `hyde_enabled: true`, and `bm25` only when `bm25_enabled: true` **and** the extraction JSONL was found. `agentic:*` commands apply to classic Agentic mode only — they're ignored (with a warning) while in Agentic-smol mode.

The header prints a `Retrieval:` line showing which stages are actually loaded, and the prompt carries `[bm25]` / `[hyde]` / `[no reranker]` tags for live state. Each answer is preceded by the stages that actually ran — so a stage being toggled on but bypassed (see Deep Dive below) is visible rather than silent.

### Web Interface

```bash
# Launch on default port 7860
python chatbot.py --web

# Custom port
python chatbot.py --web --port 8080

# Public URL via Gradio
python chatbot.py --web --share
```

The web UI includes a **Mode** toggle (RAG / Agentic / Agentic (smolagents)). Agentic mode is only available if `config.agentic` is set; Agentic (smolagents) additionally requires `smol_enabled: true`.

In RAG mode the sidebar exposes BM25 hybrid retrieval, the reranker (plus top-N), and HyDE as checkboxes. Each is gated the same way: the config flag must be `true` at startup for the feature to load at all — the checkbox can only turn a loaded feature **off**, never conjure one that wasn't configured. Slider defaults (search depth, top-N, max answer length) are read from `config.json`, not hardcoded.

Enabling **Deep Dive** hides the BM25/reranker/HyDE block and shows a notice, because the deep-dive chain runs its own retrieve-and-summarize path that bypasses all three. Search depth also switches between `k_standard` and `k_deep_dive` automatically.

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
from core import ChatbotConfig, DocumentationChatbot, AgenticChatbot, AgenticSmolChatbot

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
    hyde_enabled=True,      # override config.hyde_enabled (requires hyde_enabled: true at startup)
    bm25_enabled=True,      # override config.bm25_enabled (requires bm25_enabled: true at startup)
)

# result["filters"] reports which retrieval stages actually ran:
#   {"mode": "rag", "reranker": True, "hyde": False, "bm25": True, ...}
# In deep dive all three are False and "bypassed_by_deep_dive" lists what was skipped.

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

# --- Agentic-smol mode (requires config.agentic AND config.smol_enabled to be set) ---
agentic_smol = AgenticSmolChatbot(config)

result = agentic_smol.ask("How do I create a mesh?", max_steps=8)  # override config.agentic.max_steps
print(result["answer"])
print(result["filters"]["steps_used"])
print(result["filters"]["grounded"])  # False if the agent answered without reading any page
for source in result["sources"]:
    print(f"  - {source['title']} ({source['module']}/{source['doc_category']})")
```

All three `.ask()` methods return the same dict shape: `{answer, sources, filters, error}`.

## Response Styles (Web UI)

The style dropdown sets **temperature only**. Search depth and Deep Dive have their own explicit controls and are not touched by the style.

| Style | Temperature |
|---|---|
| Precise (default) | 0.0 |
| Balanced | 0.2 |
| Comprehensive | 0.1 |
| Config default | whatever `temperature` is in `config.json` |

**Deep Dive Mode:** retrieves `k_deep_dive` chunks, splits them into batches of `deep_dive_batch_size`, generates one summary per batch, then synthesizes a final answer. Use for complex multi-part questions.

Deep Dive uses a separate retrieval path, so **BM25 hybrid retrieval, HyDE and cross-encoder reranking do not run in Deep Dive mode.** The web UI hides those controls while it is on; the terminal warns when you enable it, and both report the bypass on each answer.

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

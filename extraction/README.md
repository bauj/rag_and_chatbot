# Documentation Extraction Pipeline

Processes HTML documentation (Sphinx, Doxygen) into:
- a **ChromaDB vector database** for RAG mode
- a **`page_index.json`** catalogue for Agentic mode

## Usage

```bash
cp config.example.json config.json
# Edit config.json: set project_name, output_dir, module paths
python process_docs.py

# Explicit config path
python process_docs.py --config /path/to/config.json

# Override output directory
python process_docs.py --output ./my_output
```

## Configuration

All configuration is in `config.json`. See `config.example.json` for a full template.

### Modules

```json
"modules": {
  "MODULE_A": {
    "description": "What this module does",
    "dev_path": "./module_a/html",
    "user_path": "./module_a/html_gui",
    "url_for_sources_citation": "https://docs.myproject.org/module_a/html"
  }
}
```

- `dev_path` — API reference / developer docs (Doxygen `html/`)
- `user_path` — tutorials / user guides (Sphinx `_build/html/`)
- `url_for_sources_citation` — optional base URL of the directory containing the HTML files, no trailing slash. Source links become `base_url/filename.html`. To find the right value, open any page of the online docs and remove the filename from the URL: e.g. `https://docs.salome-platform.org/latest/tui/SHAPER/classModelAPI__Feature.html` → `https://docs.salome-platform.org/latest/tui/SHAPER`. When omitted, sources show the local filename only.
- Both paths are optional per module; omit either if not applicable
- Add as many modules as needed; chatbot detects them automatically

### Embedding

```json
"embedding": {
  "model": "all-MiniLM-L6-v2",
  "type": "local",
  "base_url": null,
  "api_key": null
}
```

- `type: "local"` — runs `sentence-transformers` locally (default)
- `type: "api"` — calls an OpenAI-compatible embeddings endpoint; fill in `base_url` and `api_key`
- **`model` must match exactly** the `embedding.model` in `chatbot/config.json` — mismatch causes wrong retrieval with no error

### Chunking

```json
"chunking": {
  "max_tokens": 384,
  "overlap_tokens": 50,
  "char_chunk_size": 1000,
  "char_overlap": 200
}
```

Two chunking modes:

**Character-based** (`use_token_chunking: false`, default): fast, works well for large datasets.

**Token-aware** (`use_token_chunking: true`): uses the actual tokenizer for the embedding model, prevents silent truncation, more precise — but significantly slower. Recommended only for small datasets or when precision matters.

### Quality filtering

```json
"quality": {
  "min_score": 0.3,
  "min_word_count": 50,
  "substantial_word_count": 100
}
```

Quality score formula:
```
score = min(word_count / 200, 1.0)
      + 0.1 if has_title
      + 0.1 if word_count > min_word_count
      + 0.1 if word_count > substantial_word_count

Filter: score < min_score → chunk discarded
```

## Output

Generated in `output_dir`:

```
my_project_docs_extracted/
├── chromadb/                    # Vector database (used by RAG mode)
├── page_index.json              # Page catalogue (used by Agentic mode)
├── my_project_docs.json         # All chunks (JSON)
├── my_project_docs.jsonl        # All chunks (JSONL, one per line)
└── statistics.json              # Per-module chunk counts and quality stats
```

### page_index.json

A flat list of all processed HTML pages with metadata used by `AgenticChatbot` to search and select pages at query time:

```json
[
  {
    "filename": "classModelAPI__Feature.html",
    "filepath": "/abs/path/to/module_a/html/classModelAPI__Feature.html",
    "title": "ModelAPI_Feature Class Reference",
    "module": "MODULE_A",
    "doc_category": "dev"
  },
  ...
]
```

Point `chatbot/config.json → agentic.page_index_path` at this file to enable Agentic mode.

## Pipeline

```
HTML files (Doxygen + Sphinx)
        ↓
Content extraction (format-specific parsers, navigation removed)
        ↓
Dev-category pages with Doxygen memitems → one chunk per documented symbol
(class/method/function), see "Per-symbol Doxygen chunking" below.
All other pages → section splitting (h2/h3 boundaries → section_id; the
chatbot rebuilds a whole section from the chunks sharing its section_id)
        ↓
Code blocks (<pre>, Doxygen div.fragment) → converted to fenced Markdown via
markdownify, preserved inline in chunk text (not just stripped/stored separately)
        ↓
Quality scoring (filter score < min_score)
        ↓
Chunking (token-aware or character-based)
        ↓
Metadata enrichment (module, doc_category, parent_doc_id, section_id, position, quality_score, has_code)
        ↓
ChromaDB storage (cosine similarity, persistent)   ← RAG mode
page_index.json (flat page catalogue)              ← Agentic mode
```

### Per-symbol Doxygen chunking

For `doc_category: "dev"` pages that contain Doxygen `div.memitem` blocks (class
reference pages — one block per documented method/function), extraction
produces **one chunk per documented symbol** instead of one chunk per h2/h3
section. Each symbol's signature, description, and anchor are parsed directly
from the HTML (`has_memitems`/`extract_memitems_with_soup` in
`html_parser.py`). This gives retrieval much finer granularity on API
reference pages — a query about one specific method no longer has to compete
with every other method on the same class page inside a single large chunk.

Each such chunk's `hierarchy` is the symbol name (e.g. `createFeature`), its
`url` gets the symbol's in-page anchor appended (`...html#a1b2c3`), and its
metadata includes `symbol_name` and `anchor_id`. Pages without memitems (user
docs, pages without a Doxygen class-reference structure) fall back to the
original per-section chunking unchanged.

### Code block preservation

Code blocks (`<pre>`/`<code>` tags, and Doxygen's `div.fragment`/`div.line`
constructs) are converted to fenced Markdown (` ``` `) via `markdownify`
rather than being flattened to plain text or stripped. This happens in the
actual per-section chunking path (`_split_into_sections` in
`html_parser.py`), not just as a whole-page fallback, so code examples inside
regular sections are preserved too, not only on memitem pages.

## Chunk Metadata

Each chunk stored in ChromaDB includes:

| Field | Description | Example |
|---|---|---|
| `module` | Module name from config | `"MODULE_A"` |
| `doc_category` | `"dev"` or `"user"` | `"dev"` |
| `doc_type` | Detected type | `"class"`, `"tutorial"`, `"guide"` |
| `title` | Document title | `"FeatureAPI Class Reference"` |
| `parent_doc_id` | Source document ID | `"MODULE_A:dev:FeatureAPI"` |
| `section_id` | Section anchor within the page | `"classFeatureAPI#createFeature"` |
| `chunk_position` | Position within parent | `"3/7"` |
| `quality_score` | Content quality (0–1) | `0.85` |
| `has_code` | Contains code blocks | `true` |
| `source` | Human-readable source | `"my_project MODULE_A Dev Documentation"` |
| `symbol_name` | Documented symbol name (memitem chunks only) | `"createFeature"` |
| `anchor_id` | In-page anchor for the symbol (memitem chunks only) | `"a1b2c3d4"` |

## Troubleshooting

**"No HTML files found"**
- Check that `dev_path` / `user_path` point to actual HTML directories
- Paths are relative to the working directory when running `process_docs.py`

**Low chunk count**
- Lower `min_score` in config (default 0.3)
- Check `statistics.json` for per-module breakdown
- Verify the HTML files are valid

**ChromaDB creation failed**
- Check disk space and write permissions
- Delete the existing `chromadb/` directory and re-run

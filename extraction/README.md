# Documentation Extraction Pipeline

Processes HTML documentation (Sphinx, Doxygen) into a ChromaDB vector database for use by the RAG chatbot.

## Usage

```bash
cp config.example.json config.json
# Edit config.json: set project_name, output_dir, module paths
python process_docs.py --config config.json

# Override output directory
python process_docs.py --config config.json --output ./my_output
```

## Configuration

All configuration is in `config.json`. See `config.example.json` for a full template.

### Modules

```json
"modules": {
  "MODULE_A": {
    "description": "What this module does",
    "dev_path": "./module_a/html",
    "user_path": "./module_a/html_gui"
  }
}
```

- `dev_path` — API reference / developer docs (Doxygen `html/`)
- `user_path` — tutorials / user guides (Sphinx `_build/html/`)
- Both are optional per module; omit either if not applicable
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
├── chromadb/                    # Vector database (used by chatbot)
├── my_project_docs.json         # All chunks (JSON)
├── my_project_docs.jsonl        # All chunks (JSONL, one per line)
└── statistics.json              # Per-module chunk counts and quality stats
```

## Pipeline

```
HTML files (Doxygen + Sphinx)
        ↓
Content extraction (format-specific parsers, navigation removed)
        ↓
Code block extraction (<pre>/<code> tags → stored in metadata)
        ↓
Quality scoring (filter score < min_score)
        ↓
Chunking (token-aware or character-based)
        ↓
Metadata enrichment (module, doc_category, parent_doc_id, position, quality_score, has_code)
        ↓
ChromaDB storage (cosine similarity, persistent)
```

## Chunk Metadata

Each chunk stored in ChromaDB includes:

| Field | Description | Example |
|---|---|---|
| `module` | Module name from config | `"MODULE_A"` |
| `doc_category` | `"dev"` or `"user"` | `"dev"` |
| `doc_type` | Detected type | `"class"`, `"tutorial"`, `"guide"` |
| `title` | Document title | `"FeatureAPI Class Reference"` |
| `parent_doc_id` | Source document ID | `"MODULE_A:dev:FeatureAPI"` |
| `chunk_position` | Position within parent | `"3/7"` |
| `quality_score` | Content quality (0–1) | `0.85` |
| `has_code` | Contains code blocks | `true` |
| `source` | Human-readable source | `"my_project MODULE_A Dev Documentation"` |

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

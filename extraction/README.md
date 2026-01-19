# SALOME Documentation Extraction Pipeline

Advanced RAG extraction system that processes HTML documentation into optimized vector embeddings with rich metadata.

## Overview

Converts SALOME documentation (Doxygen API docs + Sphinx user guides) into a searchable ChromaDB vector database with:
- Token-aware chunking (prevents embedding truncation)
- Content quality scoring (filters noise)
- Code block extraction (enables code-aware retrieval)
- Parent document tracking (hierarchical retrieval)
- Multi-module support (SHAPER, SMESH, GUI)

## Features

### 1. Token-Aware Chunking (Optional)

**IMPORTANT:** Token-aware chunking is **DISABLED by default** due to performance impact on large datasets.

**Character-based (default, recommended for large datasets):**
```python
use_token_chunking: false  # Fast, works well for most cases
chunk_size = 1000 chars, overlap = 200 chars
```

**Token-aware (optional, for precision):**
```python
use_token_chunking: true  # Slow but more precise
max_tokens = 384  # Safe for 512 token limit
overlap_tokens = 50  # ~13% overlap for context
```

**Why token-aware is optional:**
- More precise (prevents embedding truncation)
- MUCH slower (tokenizer processes every document)
- Example: 2350 files can take 10-20 minutes vs 2-3 minutes

**When to enable:**
- Small datasets (<500 files)
- Need exact token control
- Have time for longer processing

**Fallback:** If `transformers` not installed, always uses character-based chunking.

### 2. Content Quality Scoring

Filters low-quality chunks using multi-factor heuristics:

```python
Score = min(word_count/200, 1.0)
      + 0.1 if has_title
      + 0.1 if word_count > 50
      + 0.1 if word_count > 100
```

**Filters out:** Navigation pages, auto-generated indices, content with score < 0.3

**Keeps:** Meaningful documentation with substantial content

### 3. Code Block Extraction

Automatically detects and extracts code snippets:
- Finds `<pre>` and `<code>` tags
- Deduplicates nested tags
- Stores up to 5 code blocks per document
- Adds `has_code: true` metadata flag

**Enables queries like:**
```python
# Retrieve only chunks with code examples
filter={"has_code": True}
```

### 4. Parent Document Tracking

Every chunk knows its origin:

```python
parent_doc_id: "SHAPER:dev:FeatureAPI"  # module:category:filename
chunk_position: "3/7"                    # position/total
```

**Enables:**
- Retrieve all chunks from same document
- Context-aware re-ranking
- Hierarchical retrieval strategies

### 5. Explicit Embedding Configuration

```python
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
collection.create_collection(
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"}  # Cosine similarity
)
```

**Benefits:**
- Matches chatbot's embedding model exactly
- Uses cosine similarity (better than L2 for text)
- No silent model mismatches

### 6. Rich Metadata

Each chunk includes:

| Field | Description | Example |
|-------|-------------|---------|
| `title` | Document title | "FeatureAPI Class Reference" |
| `module` | SALOME module | "SHAPER" |
| `doc_category` | Documentation type | "dev" or "user" |
| `doc_type` | Specific type | "class", "tutorial", "guide" |
| `parent_doc_id` | Parent document | "SHAPER:dev:FeatureAPI" |
| `chunk_position` | Position in parent | "3/7" |
| `quality_score` | Content quality (0-1) | 0.85 |
| `has_code` | Contains code blocks | true |
| `code_blocks` | Extracted code snippets | [...] |

## Installation

```bash
# Required dependencies
pip install beautifulsoup4 chromadb sentence-transformers

# Recommended (for token-aware chunking)
pip install transformers

# Or install all
pip install -r ../requirements.txt
```

## Usage

### Recommended: Use Config File

```bash
# Create config from example
cp config.example.json config.json

# Edit config.json to add your modules and paths
# Then run:
python process_multi_module_salome_docs.py --config config.json
```

**Why use config:**
- Easy to add new modules
- Centralized configuration
- Fine-tune chunking/quality parameters
- Share configurations across team
- Version control settings

### Basic Usage (CLI Arguments)

```bash
# Process documentation from default locations
python process_multi_module_salome_docs.py
```

**Expected directory structure:**
```
extraction/
├── shaper_docs_extracted/
│   ├── html/          # Dev docs (Doxygen)
│   └── html_gui/      # User docs (Sphinx)
├── smesh_docs_extracted/
│   ├── html/
│   └── html_gui/
└── gui_docs_extracted/
    ├── html/
    └── html_gui/
```

### Custom Paths

```bash
# Specify custom directories
python process_multi_module_salome_docs.py \
  --shaper-dir /path/to/shaper_docs \
  --smesh-dir /path/to/smesh_docs \
  --gui-dir /path/to/gui_docs \
  --output ./my_output

# Manual control over individual paths
python process_multi_module_salome_docs.py \
  --shaper-dev /path/to/shaper/html \
  --shaper-user /path/to/shaper/html_gui \
  --smesh-dev /path/to/smesh/html
```

### Skip Missing Modules

```bash
# Continue if some modules are missing
python process_multi_module_salome_docs.py --skip-missing
```

## Output

### Generated Files

```
salome_docs_extracted/
├── chromadb/              # Vector database
│   └── ...                # (Used by chatbot)
├── salome_docs.json       # All chunks (JSON)
├── salome_docs.jsonl      # All chunks (JSONL, one per line)
└── statistics.json        # Extraction statistics
```

### Statistics Example

```json
{
  "total_chunks": 4523,
  "modules": {
    "SHAPER": {
      "total_chunks": 2145,
      "dev_chunks": 1823,
      "user_chunks": 322,
      "description": "CAD modeling and geometry creation"
    },
    "SMESH": {
      "total_chunks": 1689,
      "dev_chunks": 1234,
      "user_chunks": 455,
      "description": "Mesh generation and manipulation"
    },
    "GUI": {
      "total_chunks": 689,
      "dev_chunks": 512,
      "user_chunks": 177,
      "description": "Graphical user interface components"
    }
  }
}
```

## Pipeline Flow

```
1. HTML FILES
   Doxygen (dev) + Sphinx (user)
        ↓
2. CONTENT EXTRACTION
   Format-specific parsers
   Remove navigation/sidebars
        ↓
3. CODE BLOCK EXTRACTION
   Find <pre>/<code> tags
   Store in metadata
        ↓
4. QUALITY SCORING
   Multi-factor heuristics
   Filter score < 0.3
        ↓
5. TOKEN-AWARE CHUNKING
   384 tokens max, 50 overlap
   Respect sentence boundaries
        ↓
6. METADATA ENRICHMENT
   Parent tracking, position, quality, code flags
        ↓
7. VECTOR EMBEDDING
   all-MiniLM-L6-v2
   Cosine similarity
        ↓
8. CHROMADB STORAGE
   Persistent vector database
```

## Configuration

### Using config.json (Recommended)

Create `config.json` based on `config.example.json`:

```json
{
  "output_dir": "./salome_docs_extracted",
  "use_token_chunking": true,
  "modules": {
    "SHAPER": {
      "description": "CAD modeling and geometry creation",
      "url": "https://docs.salome-platform.org/latest/tui/SHAPER",
      "dev_path": "./shaper_docs_extracted/html",
      "user_path": "./shaper_docs_extracted/html_gui"
    },
    "MY_NEW_MODULE": {
      "description": "My custom SALOME module",
      "url": "https://example.com/docs",
      "dev_path": "./my_module/html",
      "user_path": "./my_module/html_gui"
    }
  },
  "chunking": {
    "max_tokens": 384,
    "overlap_tokens": 50,
    "char_chunk_size": 1000,
    "char_overlap": 200
  },
  "quality": {
    "min_score": 0.3,
    "min_word_count": 50,
    "substantial_word_count": 100
  }
}
```

**Then run:**
```bash
python process_multi_module_salome_docs.py --config config.json
```

### Configuration Parameters

**Chunking:**
- `max_tokens`: Max tokens per chunk (token-aware mode) [default: 384]
- `overlap_tokens`: Token overlap between chunks [default: 50]
- `char_chunk_size`: Max chars per chunk (fallback mode) [default: 1000]
- `char_overlap`: Character overlap [default: 200]

**Quality:**
- `min_score`: Minimum quality score threshold [default: 0.3]
- `min_word_count`: Minimum words for "has content" bonus [default: 50]
- `substantial_word_count`: Minimum for "substantial" bonus [default: 100]

**Modules:**
Each module can define:
- `description`: Human-readable description
- `url`: Documentation URL
- `dev_path`: Path to dev docs (Doxygen HTML)
- `user_path`: Path to user docs (Sphinx HTML)

### Adding a New Module

**Option 1: Via config.json** (easiest)
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

**Option 2: Via CLI**
```bash
python process_multi_module_salome_docs.py \
  --geom-dev ./geom_docs/html \
  --geom-user ./geom_docs/html_gui
```

## Advanced Features

### Document Type Detection

Automatically classifies documents:

**Dev Docs (Doxygen):**
- `class` - Class reference
- `namespace` - Namespace reference
- `module` - Module/group reference
- `page` - General page

**User Docs (Sphinx):**
- `tutorial` - Tutorial content
- `guide` - How-to guide
- `reference` - API reference
- `page` - General page

### Dual Parser System

**Doxygen Parser:**
```python
def extract_content_doxygen(self, soup):
    contents = soup.find('div', class_='contents')
    # Clean extraction from Doxygen structure
```

**Sphinx Parser:**
```python
def extract_content_sphinx(self, soup):
    # Try multiple content containers
    for area in [div[role='main'], div.document, article]:
        if area:
            # Remove sidebars/navigation
            return clean_content
```

### Batch Processing

ChromaDB insertion uses batching for efficiency:

```python
batch_size = 100  # Process 100 chunks at a time
for i in range(0, len(chunks), batch_size):
    collection.add(batch_docs, batch_metas, batch_ids)
```

## Performance

### Typical Processing Times

| Module | Files | Chunks | Time |
|--------|-------|--------|------|
| SHAPER | ~500 | ~2000 | 2-3 min |
| SMESH | ~400 | ~1500 | 2 min |
| GUI | ~200 | ~700 | 1 min |
| **Total** | ~1100 | ~4200 | **5-6 min** |

*With transformers installed and token-aware chunking enabled*

### Memory Usage

- Peak RAM: ~2-3 GB
- ChromaDB size: ~100-150 MB
- JSON output: ~20-30 MB

## Troubleshooting

### No HTML files found

```
Warning: No HTML files found
```

**Solution:** Check directory structure. Expected:
```
module_docs_extracted/
├── html/       # Dev docs
└── html_gui/   # User docs
```

### Transformers not installed

```
WARNING: transformers not installed - using character-based chunking
```

**Solution:** Install for better chunking:
```bash
pip install transformers
```

**Impact:** Falls back to character-based chunking (still works, just less optimal)

### ChromaDB creation failed

```
Warning: ChromaDB error: ...
```

**Solution:**
1. Check disk space
2. Ensure write permissions
3. Delete old `chromadb/` directory
4. Reinstall: `pip install --upgrade chromadb`

### Low chunk count

```
Total: 150 chunks (expected ~4000)
```

**Causes:**
1. Quality threshold too high (adjust from 0.3)
2. Missing documentation directories
3. Malformed HTML files

**Solution:** Check statistics.json for per-module breakdown

## Technical Details

### Chunk Size Rationale

**384 tokens chosen because:**
- all-MiniLM-L6-v2 max: 512 tokens
- Leave buffer for [CLS]/[SEP] tokens
- 384 + 50 overlap = safe margin
- ~300-350 words per chunk (readable)

### Quality Score Formula

```
Base = min(word_count / 200, 1.0)  # Expected ~200 words
     + 0.1 if has_title
     + 0.1 if word_count > 50
     + 0.1 if word_count > 100

Score = min(Base, 1.0)  # Cap at 1.0
```

**Filter:** score < 0.3

**Examples:**
- 25 words, no title: 0.125 (filtered)
- 75 words, has title: 0.575 (kept)
- 300 words, has title: 1.0 (kept)

### Embedding Model

**all-MiniLM-L6-v2:**
- Size: 384 dimensions
- Max sequence: 512 tokens
- Speed: Fast (~2000 chunks/min on CPU)
- Quality: Good for technical docs
- License: Apache 2.0 (free)

## Integration with Chatbot

The chatbot automatically uses the generated ChromaDB:

```bash
cd ../chatbot
python chatbot.py
```

**Chatbot finds ChromaDB at:**
```
../extraction/salome_docs_extracted/chromadb
```

**Chatbot can filter by metadata:**
```python
# Only SHAPER dev docs with code examples
retriever = vectorstore.as_retriever(
    search_kwargs={
        "filter": {
            "$and": [
                {"module": "SHAPER"},
                {"doc_category": "dev"},
                {"has_code": True}
            ]
        }
    }
)
```

## Contributing

### Adding a New Module

1. Add to `MODULE_INFO` dict:
```python
MODULE_INFO = {
    'MYNEWMODULE': {
        'description': 'Module description',
        'url': 'https://docs.salome-platform.org/...'
    }
}
```

2. Run with new module paths:
```bash
python process_multi_module_salome_docs.py \
  --mynewmodule-dev /path/to/html \
  --mynewmodule-user /path/to/html_gui
```

### Customizing Quality Scoring

Edit `calculate_quality_score()`:
```python
def calculate_quality_score(self, content, title):
    # Add your own heuristics
    has_diagrams = 'image' in content.lower()
    if has_diagrams:
        base_score += 0.15

    return min(base_score, 1.0)
```

### Adding New Metadata Fields

Edit `process_file()`:
```python
metadata={
    # ... existing fields ...
    'has_diagrams': detect_diagrams(soup),
    'word_count': len(words),
    'last_updated': extract_date(soup)
}
```

## License

Part of the SALOME Documentation RAG System.

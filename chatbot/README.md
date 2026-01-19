# SALOME Documentation Chatbot

Refactored RAG chatbot with clean separation between business logic and UI.

## Architecture

```
chatbot/
├── core/                      # Business logic (no UI)
│   ├── config.py             # Hybrid config (JSON + dataclass)
│   └── salome_chatbot.py     # RAG logic
├── ui/                        # User interfaces
│   ├── terminal.py           # Terminal interface
│   └── web.py                # Gradio web interface
├── chatbot.py                # Unified entry point
├── config.example.json       # Example configuration
└── _old/                     # Legacy implementations
```

## Features

- **Multi-module support**: SHAPER, SMESH, GUI
- **Flexible filtering**: By module and doc type (dev/user)
- **Deep Dive mode**: Comprehensive analysis with batch summarization
- **Hybrid configuration**: JSON files + Python defaults + CLI overrides
- **Two interfaces**: Terminal (interactive) and Web (Gradio)
- **Clean architecture**: Core logic separated from UI

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r ../requirements.txt
```

### Configuration

Three ways to configure (priority: CLI > JSON > defaults):

**1. Use defaults** (no config needed)
```bash
python chatbot.py
```

**2. Use JSON config file**
```bash
# Create config from example
cp config.example.json config.json

# Edit config.json with your settings
# Chatbot automatically loads config.json if present

python chatbot.py
```

**3. Override with CLI arguments**
```bash
python chatbot.py --model mistral-mini --chromadb /path/to/chromadb
```

## Usage

### Terminal Interface

**Interactive mode** (default):
```bash
python chatbot.py
```

Commands:
- `module:SHAPER` - Filter by module
- `type:dev` - Filter by doc type
- `deep` - Toggle deep dive mode
- `stats` - Show database statistics
- `clear` - Clear filters
- `exit` - Quit

**Single question mode**:
```bash
python chatbot.py --question "What is ModelAPI::Feature?"
python chatbot.py --question "How to create mesh?" --module SMESH --deep-dive
```

### Web Interface

```bash
# Launch on default port 7860
python chatbot.py --web

# Custom port
python chatbot.py --web --port 8080

# Create public URL (via Gradio)
python chatbot.py --web --share
```

## Configuration Options

All settings from `config.example.json`:

| Setting | Default | Description |
|---------|---------|-------------|
| `chromadb_path` | `../extraction/salome_docs_extracted/chromadb` | Path to ChromaDB |
| `litellm_url` | `http://localhost:8080` | LiteLLM server URL |
| `model_name` | `mistral` | LLM model name |
| `embedding_model` | `all-MiniLM-L6-v2` | Embedding model |
| `k_standard` | `40` | Chunks for standard retrieval |
| `k_deep_dive` | `60` | Chunks for deep dive mode |
| `deep_dive_batch_size` | `10` | Batch size for summarization |
| `temperature` | `0.0` | LLM temperature |
| `max_tokens` | `2000` | Max tokens per response |

## Examples

```bash
# Interactive terminal with default settings
python chatbot.py

# Single question with custom config
python chatbot.py --config prod.json --question "Explain mesh generation"

# Web interface with overrides
python chatbot.py --web --model mistral-mini --port 9000

# Deep dive mode for complex questions
python chatbot.py --question "Complete SMESH workflow" --module SMESH --deep-dive
```

## Migration from Legacy

**Old scripts** (moved to `_old/`):
- `chatbot_salome_doc.py` → `python chatbot.py`
- `chatbot_web_salome_doc.py` → `python chatbot.py --web`

**Equivalent commands**:
```bash
# Old
python chatbot_salome_doc.py --question "test" --module SHAPER

# New
python chatbot.py --question "test" --module SHAPER
```

## Using as a Library

```python
from core import ChatbotConfig, SALOMEChatbot

# Create config
config = ChatbotConfig(
    model_name="mistral-mini",
    k_standard=50
)

# Or load from JSON
config = ChatbotConfig.from_json("my_config.json")

# Initialize chatbot
bot = SALOMEChatbot(config)

# Ask questions
result = bot.ask(
    "What is ModelAPI::Feature?",
    module="SHAPER",
    doc_type="dev"
)

print(result["answer"])
for source in result["sources"]:
    print(f"  - {source['title']}")
```

## Development

**Add new UI**:
1. Create `ui/my_interface.py`
2. Import `SALOMEChatbot` from `core`
3. Call `bot.ask()` and format results
4. No core logic changes needed!

**Modify RAG logic**:
1. Edit `core/salome_chatbot.py`
2. All UIs automatically use new logic

## Troubleshooting

**ChromaDB not found**:
```bash
cd ../extraction
python process_multi_module_salome_docs.py
```

**LiteLLM connection failed**:
```bash
# Check LiteLLM is running
curl http://localhost:8080/v1/models

# Or specify different URL
python chatbot.py --litellm-url http://your-server:8080
```

**Import errors**:
```bash
pip install -r ../requirements.txt
```

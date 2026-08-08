# Benchmark Suite

Tools for testing and comparing RAG and agentic modes across every layer of the pipeline: chatbot hyperparameters (temperature, number of chunks, reranker, ...), LLM/reranker model choice, and the extraction pipeline that builds the corpus (chunking, quality filtering, embedding model).

## Quick Start

```bash
# Full test
python benchmark.py

# Run benchmark with 5 workers
python benchmark.py --workers 5

# Validate the setup
python benchmark.py --help

# Quick test (2 questions)
python benchmark.py --limit 2

# Limit chatbot timeout (default no timeout)
python benchmark.py --timeout 60

# Compare existing benchmarks
python benchmark.py --compare

# Analyze results
python analyze_results.py benchmark_YYYYMMDD_HHMMSS.json

# Analyze latest results and plot graphs
python analyze_results.py --latest --graphs
```

## Configuration

Benchmark configurations are defined in `benchmark_config.json`. Copy `benchmark_config.example.json` to create your configuration.

Example structure:

```json
{
  "modes": ["rag", "agentic"],
  "rag_hyperparams": {
    "k_standard": [40],
    "k_deep_dive": [60],
    "temperature": [0.3, 0.7],
    "max_tokens": [2000],
    "reranker_enabled": [true, false],
    "reranker_model": ["BAAI/bge-reranker-v2-m3"],
    "top_n": [5, 10],
    "deep_dive": [false],
    "hyde_enabled": [false],
    "bm25_enabled": [true, false],
    "title_boost_enabled": [true],
    "k_retrieve": [80],
    "deep_dive_batch_size": [10],
    "expansion_char_budget": [60000]
  },
  "agentic_hyperparams": {
    "temperature": [0.7],
    "max_tokens": [2000],
    "max_steps": [4, 6, 8],
    "max_chars_per_page": [8000]
  },
  "model_hyperparams": {
    "model": ["mistralai-small"]
  },
  "extraction_hyperparams": {}
}
```

Every combination of the values above is run: each mode gets its own full cross-product over its own hyperparams (`rag_hyperparams` for rag, `agentic_hyperparams` for agentic — the two are never combined with each other), and that in turn is cross-produced with `model_hyperparams` and `extraction_hyperparams`. Each extra value in any list multiplies the total run count — prefer varying one or two dimensions at a time. Omit a key entirely to leave that setting at whatever `chatbot/config.json` (or `extraction/config.json`) already has (`chatbot.py`/`process_docs.py` only override what's explicitly passed).

RAG hyperparameters: `k_standard`/`k_deep_dive` (RAG pool size — only one applies per run, picked by that run's own `deep_dive` value; `chatbot.py --k` also exists as a flat override but isn't needed here), `temperature`, `max_tokens` (LLM generation length), `reranker_enabled`, `reranker_model` (only has an effect when `reranker_enabled` is true), `top_n` (docs kept after reranking), `deep_dive` (comprehensive-answer mode), `hyde_enabled`, `bm25_enabled` and `title_boost_enabled` (retrieval-pipeline toggles — see `chatbot/config.json`'s comments for what each does), `k_retrieve` (per-channel retrieval depth before RRF fusion — decoupled from `k_standard`/`k_deep_dive`, the fused pool size; ignored in deep dive mode), `deep_dive_batch_size` (only applies when `deep_dive` is true), `expansion_char_budget` (shared char budget for expanding surviving docs to their full section).

Agentic hyperparameters: `temperature`, `max_tokens`, `max_steps` (the CodeAgent step budget), `max_chars_per_page` (how much of each page it reads).

### Model choice

`model_hyperparams` is optional and applies on top of **every** mode/hyperparameter combination above — the answering LLM choice is independent of rag vs agentic. Fields: `model`, `base_url`, `api_key`. Since values are cross-produced independently, mixing several `(base_url, api_key)` pairs with several `model` values can pair a model name with the wrong endpoint's key — keep `base_url`/`api_key` to a single value each (or run separate benchmarks per provider) unless every combination is actually valid. Leave `{}` to always use `chatbot/config.json`'s `llm` block unchanged.

### Extraction parameters

`extraction_hyperparams` is optional and, like `model_hyperparams`, applies on top of every mode/model/hyperparameter combination above. Fields: `use_token_chunking`, `chunking.max_tokens`, `chunking.overlap_tokens`, `chunking.char_chunk_size`, `chunking.char_overlap`, `quality.min_score`, `quality.min_word_count`, `quality.substantial_word_count`, `embedding.model`, `embedding.type`, `embedding.base_url`, `embedding.api_key` (dotted keys map to `extraction/config.json`'s nested blocks).

Unlike everything else in this file, each distinct combination here costs a full corpus re-extraction (`extraction/process_docs.py` — minutes, no early-exit) the first time it's seen. `evaluator/base/extraction_cache.py` caches builds by a hash of the combination under `evaluator/extraction_cache/<hash>_extracted/` and reuses one build across every mode/model/hyperparameter combination that shares it (built once, not once per run) — but grid search still runs **every** combination in the grid, with no smart sampling like `evaluator/optimizer`'s Bayesian search. Keep this to 1-2 values per field, or the run count explodes on top of an already expensive rebuild. `--extraction-timeout` (default: no limit) bounds how long a single build may take. Leave `{}` to always use `extraction/config.json`'s build unchanged.

## Scripts

- **benchmark.py** - Run benchmarks
  - `--limit N` - Limit to N questions
  - `--workers N` - Parallel threads for evaluating questions, -1 for all CPUs
  - `--timeout N` - Timeout in seconds for each chatbot request
  - `--extraction-timeout N` - Timeout in seconds for building one extraction variant
  - `--compare` - Compare previous results

- **analyze_results.py** - Analyze results
  - `--latest` - Latest result
  - `--list` - List all results
  - `--graphs` - Generate analysis graphs (requires matplotlib)

## Benchmark output

The outputs are saved in a `benchmark_results` folder (created if it does not exist) as a JSON file named `benchmark_YYYYMMDD_HHMMSS.json`, in order to ensure traceability.

This JSON file contains, for each configuration, all questions along with their corresponding expected answers, chatbot answers, scores, and explanations, in the following format:

```json
{
  "metadata": {
    "timestamp": "YYYYMMDD_HHMMSS",
    "benchmark_config": {
      "modes": ["rag", "agentic"],
      "rag_hyperparams": { "k": [3, 5], "temperature": [0.3, 0.7] },
      "agentic_hyperparams": { "temperature": [0.3, 0.7] }
    },
    "total_questions": 13,
    "workers": 5,
    "timeout_seconds": null
  },
  "results": [
    {
      "mode": "rag",
      "configuration": {
        "k_standard": 40,
        "temperature": 0.3,
        "reranker_enabled": true,
        "top_n": 5,
        "deep_dive": false,
        "model": "mistralai-small"
      },
      "extraction": null,
      "questions": [
        {
          "question_id": 1,
          "question": "Question 1?",
          "tags": ["Type", "Module"],
          "reference_answer": "Expected Answer 1.",
          "generated_answer": "Chatbot Answer 1.",
          "evaluations": {
            "correctness": {
              "score": 5.0,
              "explanation": "Explanation of the correctness score."
            },
            "relevance": {
              "score": 8.0,
              "explanation": "Explanation of the relevance score."
            },
            "groundedness": {
              "score": 2.0,
              "explanation": "Explanation of the groundedness score."
            },
            "retrieval_relevance": {
              "score": 10.0,
              "explanation": "Explanation of the retrieval relevance score."
            }
          },
          "request_time": 24.658
        }
      ],
      "average_scores": {
        "correctness": 5.0,
        "relevance": 8.0,
        "groundedness": 2.0,
        "retrieval_relevance": 10.0
      },
      "average_request_time": 24.658
    }
  ]
}
```

`metadata.benchmark_config` is the full content of `benchmark_config.json` used for that run — kept alongside the results so a given `benchmark_*.json` file is self-describing (which modes and hyperparameter grid produced it) without needing to cross-reference a separate config file.

`configuration` merges the mode's own hyperparameters with any `model_hyperparams` combination for that run (both end up as chatbot.py CLI flags either way). `extraction` is `null` unless `extraction_hyperparams` was set, in which case it's that run's extraction combination (the same combination is shared, not repeated, across every mode/model/hyperparameter run built from the same extraction variant).

## Analysis Graphs

Use `--graphs` to generate visualizations:

- **Quality vs time** (`01`): Scatter plot revealing the trade-off between overall response quality and execution speed across different tested configurations
- **Metric distribution** (`02`): Box plots showing the score dispersion for each evaluation metric (correctness, relevance, groundedness, retrieval) across configurations
- **Performance heatmap** (`03`): Colored matrix providing a visual overview of average scores achieved by each configuration across all evaluated metrics
- **Scores by question config** (`04`): Grouped bar chart comparing the overall performance of each configuration on the first 15 benchmark questions, each bar labeled with its exact score
- **Scores by type config** (`05`): Box plots organized by question type tag (`Cpp`, `Py`, `Methodology`, `wrong`) to identify strengths and weaknesses of configurations according to the kind of question asked
- **Scores by module config** (`06`): Box plots organized by Uranie module tag (e.g. `Sampler`, `Sensitivity`, `DataServer`) to identify strengths and weaknesses of configurations according to question topic
- **Response time by config** (`07`): Bar chart displaying the average response times for each tested configuration
- **Response time by question config** (`08`): Grouped bar chart detailing response time variations according to questions and configurations, each bar labeled with its exact time
- **Scores by difficulty config** (`09`): Box plots organized by difficulty tag (`easy`/`medium`/`hard`) to see how each configuration holds up as questions get harder
- **Scores by language config** (`10`): Box plots organized by language tag (`Fr`/`En`) to check for language-dependent performance gaps

Difficulty and language tags are recognized by fixed values (`easy`/`medium`/`hard`, `Fr`/`En`). Every question in `dataset.json` (see `evaluator/base/README.md`) also carries exactly two more tags, always in the order `[Type, Module]` (e.g. `["Py", "Sampler"]`) — these are read positionally, not from a fixed value list, so any new type or module tag is picked up automatically as long as this ordering convention is kept.

Graphs are saved in the `graphs/` folder.

## Metrics

1. Correctness - Factual accuracy (0-10)
2. Relevance - Overall relevance (0-10)
3. Groundedness - Grounding in documents (0-10)
4. Retrieval Relevance - Relevance of retrieved documents (0-10)

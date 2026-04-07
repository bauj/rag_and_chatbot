# Benchmark Suite

Tools for testing and comparing RAG and agentic modes playing on the hyperparameters (temperature, number of chunks, re-ranker, ...).

## Quick Start

```bash
# Validate the setup
python3 run_benchmark.py --help

# Quick test (2 questions)
python3 run_benchmark.py --limit 2

# Full test (13 questions/config, ~30 min/config)
python3 run_benchmark.py

# Analyze results
python3 analyze_results.py --latest
```

## Configuration

Benchmark configurations are defined in `benchmark_config.json`. Copy `benchmark_config.example.json` to create your configuration.

Example structure:

```json
{
  "rag_hyperparams": {
    "k": [3, 5, 10],
    "temperature": [0.3, 0.7],
    "reranker_enabled": [true, false],
    "top_n": [5, 10]
  },
  "agentic_hyperparams": {
    "temperature": [0.7]
  }
}
```

## Scripts

- **run_benchmark.py** - Run benchmarks
  - `--modes rag|agentic` - Modes to test (default: both)
  - `--limit N` - Limit to N questions
  - `--compare` - Compare previous results

- **analyze_results.py** - Analyze results
  - `--latest` - Latest result
  - `--list` - List all results
  - `--graphs` - Generate analysis graphs (requires matplotlib)

- **benchmark.py** - Core module

## Analysis Graphs

Use `--graphs` to generate visualizations:

- **Scores per question**: Bar charts of average scores per question and metric
- **Metric distributions**: Histograms of score distributions
- **Scores by tag**: Average scores grouped by question tag

Graphs are saved in the `graphs/` folder.

## Metrics

1. Correctness - Factual accuracy (0-10)
2. Relevance - Overall relevance (0-10)
3. Groundedness - Grounding in documents (0-10)
4. Retrieval Relevance - Relevance of retrieved documents (0-10)

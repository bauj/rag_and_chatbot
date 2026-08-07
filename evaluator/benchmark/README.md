# Benchmark Suite

Tools for testing and comparing RAG and agentic modes playing on the hyperparameters (temperature, number of chunks, re-ranker, ...).

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

Every combination of the values above is run (a full cross-product), so each extra value in a list multiplies the total run count — prefer varying one or two dimensions at a time. `hyde_enabled`, `bm25_enabled` and `title_boost_enabled` are RAG retrieval-pipeline toggles (see `chatbot/config.json`'s comments for what each does); `max_steps` is the agentic mode's CodeAgent step budget, and `max_chars_per_page` is how much of each page it reads.

## Scripts

- **benchmark.py** - Run benchmarks
  - `--limit N` - Limit to N questions
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
        "k": 3,
        "temperature": 0.3,
        "reranker_enabled": true,
        "top_n": 5,
        "deep_dive": false
      },
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

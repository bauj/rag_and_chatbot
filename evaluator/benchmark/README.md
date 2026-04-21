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
    "modes": [
      "rag",
      "agentic"
    ],
    "total_questions": 13,
    "workers": 5
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
          }
        }
      ]
    }
  ]
}
```

## Analysis Graphs

Use `--graphs` to generate visualizations:

- **Quality vs time**: Scatter plot revealing the trade-off between overall response quality and execution speed across different tested configurations
- **Metric distribution**: Box plots showing the score dispersion for each evaluation metric (correctness, relevance, groundedness, retrieval) across configurations
- **Performance heatmap**: Colored matrix providing a visual overview of average scores achieved by each configuration across all evaluated metrics
- **Scores by question config**: Grouped bar chart comparing the overall performance of each configuration on the first 15 benchmark questions
- **Scores by tag config**: Box plots organized by question categories (tags) to identify strengths and weaknesses of configurations according to question types
- **Response time by config**: Bar chart displaying the average response times for each tested configuration
- **Response time by question config**: Grouped bar chart detailing response time variations according to questions and configurations

Graphs are saved in the `graphs/` folder.

## Metrics

1. Correctness - Factual accuracy (0-10)
2. Relevance - Overall relevance (0-10)
3. Groundedness - Grounding in documents (0-10)
4. Retrieval Relevance - Relevance of retrieved documents (0-10)

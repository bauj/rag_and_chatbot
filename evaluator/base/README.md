# Base Evaluator

Base module for evaluating RAG and agentic chatbot responses.

## Configuration

The `config.json` file defines evaluation parameters:

- `chatbot_path`: Path to the chatbot directory (relative to `base/` folder)
- `evaluator_options`: Additional options for the evaluator

Copy `config.example.json` to create your configuration.

## Usage

```bash
# Full test
python evaluator.py

# Quick test (2 questions)
python evaluator.py --limit 2

# Parallel mode (5 workers)
python evaluator.py --workers 5

# Limit chatbot timeout (default no timeout)
python evaluator.py --timeout 60
```

## Metrics

- **Correctness**: Factual accuracy (0-10)
- **Relevance**: Overall relevance (0-10)
- **Groundedness**: Grounding in documents (0-10)
- **Retrieval Relevance**: Relevance of retrieved documents (0-10)

## Dataset

The test question dataset is defined in `dataset.json`. Each question can include tags for the theme:

```json
[
  {
    "inputs": {"question": "How to use setRange() with a percentage in TOATDesign in C++?"},
    "outputs": {"answer": "..."},
    "tags": ["Cpp", "Sampler"]
  }
]
```

You can define your own tags, but here are some examples: `Cpp`, `Py`, `Methodology`, `Sampler`, `Sensitivity`, `Optimizer`, `Calibration`, `DataServer`, `Launcher`, `Relauncher`, etc.

Copy `dataset.example.json` to create your dataset.

The results are saved in `evaluation_results.json`. The file starts with a `summary` object (the same averages printed to the terminal as "EVALUATION SUMMARY - AVERAGE SCORES"), followed by `results`: the scores for each question along with a brief explanation of each grade, in the following format:

```json
{
  "summary": {
    "total_examples": 1,
    "correctness": 5.0,
    "relevance": 8.0,
    "groundedness": 2.0,
    "retrieval_relevance": 10.0,
    "overall_average": 6.25,
    "average_request_time_seconds": 24.65836753399344
  },
  "results": [
    {
      "question": "Question 1?",
      "expected_answer": "Expected Answer 1.",
      "rag_answer": "Chatbot Answer 1.",
      "request_time": 24.65836753399344,
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
```

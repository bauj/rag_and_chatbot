# Base Evaluator

Base module for evaluating RAG and agentic chatbot responses.

## Configuration

The `config.json` file defines evaluation parameters:

- `chatbot_path`: Path to the chatbot directory (relative to `base/` folder)
- `evaluator_options`: Additional options for the evaluator

Copy `config.example.json` to create your configuration.

## Usage

```bash
python evaluator.py
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

Available tags: `Cpp`, `Py`, `Methodology`, `Sampler`, `Sensitivity`, `Optimizer`, `Calibration`, `DataServer`, `Launcher`, `Relauncher`, etc.

Copy `dataset.example.json` to create your dataset.

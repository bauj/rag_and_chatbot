# Benchmark Suite

Suite d'outils pour tester et comparer les modes RAG et agentic.

## Démarrage rapide

```bash
# Valider le setup
python3 run_benchmark.py --help

# Quick test (2 questions)
python3 run_benchmark.py --limit 2

# Test complet (13 questions, ~30 min)
python3 run_benchmark.py

# Analyser les résultats
python3 analyze_results.py --latest
```

## Scripts

- **run_benchmark.py** - Lancer les benchmarks
  - `--modes rag|agentic` - Modes à tester (défaut: les deux)
  - `--limit N` - Limiter à N questions
  - `--compare` - Comparer les résultats précédents
  
- **analyze_results.py** - Analyser les résultats
  - `--latest` - Dernier résultat
  - `--list` - Lister tous les résultats
  
- **benchmark.py** - Module core

## Configurations testées

**RAG (16 configs):**
- k: 3, 5, 10
- temperature: 0.3, 0.7
- reranker_enabled: True, False
- top_n: 5, 10

**Agentic (1 config)**

## Résultats

Sauvegardés dans `benchmark_results/benchmark_YYYYMMDD_HHMMSS.json`

## Métriques

1. Correctness - Précision factuelle (0-10)
2. Relevance - Pertinence globale (0-10)
3. Groundedness - Ancrage dans les documents (0-10)
4. Retrieval Relevance - Pertinence des documents (0-10)

#!/usr/bin/env python3
"""
Benchmark script for testing different modes and hyperparameters
Tests RAG and agentic modes with multiple configurations
"""

import sys
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_CONFIG_FILE = SCRIPT_DIR / "benchmark_config.json"
BENCHMARK_CONFIG_EXAMPLE_FILE = SCRIPT_DIR / "benchmark_config.example.json"
# Anchored to the script's own directory rather than the current working
# directory, so results always land next to benchmark.py regardless of where
# `python benchmark.py` (or `python evaluator/benchmark/benchmark.py`) is run from.
BENCHMARK_RESULTS_DIR = SCRIPT_DIR / "benchmark_results"

_SUPPORTED_MODES = {"rag", "agentic"}

def _validate_benchmark_config(config_data: Dict[str, Any]) -> None:
    """Fail fast with a clear message if benchmark_config.json is missing keys
    that the rest of this module indexes directly (config_data["modes"], etc.),
    instead of a bare KeyError deep inside run_benchmark()."""
    missing = [
        key for key in ("modes", "rag_hyperparams", "agentic_hyperparams")
        if key not in config_data
    ]
    if missing:
        raise ValueError(
            f"Benchmark config is missing required key(s): {', '.join(missing)}. "
            f"See {BENCHMARK_CONFIG_EXAMPLE_FILE} for the expected structure."
        )

    unknown_modes = [m for m in config_data["modes"] if m not in _SUPPORTED_MODES]
    if unknown_modes:
        raise ValueError(
            f"Unsupported mode(s) in benchmark config: {unknown_modes}. "
            f"Supported modes are: {sorted(_SUPPORTED_MODES)}."
        )

    if "temperature" not in config_data["agentic_hyperparams"]:
        raise ValueError(
            "benchmark config's 'agentic_hyperparams' must include a 'temperature' list."
        )

def load_benchmark_config() -> Dict[str, Any]:
    """Load benchmark parameters from JSON configuration."""
    config_path = BENCHMARK_CONFIG_FILE
    if not config_path.exists():
        raise FileNotFoundError(
            f"Benchmark config file not found: {config_path}. "
            f"Create it from {BENCHMARK_CONFIG_EXAMPLE_FILE} and rerun."
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Benchmark config file is invalid JSON: {config_path}: {exc}"
        ) from exc

    _validate_benchmark_config(config_data)
    return config_data

# Import from evaluator base package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluator.base.evaluator import (
    _call_chatbot,
    _validate_config,
    run_evaluation,
    _load_examples,
    EVAL_METRICS,
)

def generate_rag_configs(hyperparams: Dict) -> List[Dict]:
    """Generate all RAG hyperparameter combinations"""
    keys = list(hyperparams.keys())
    values = [hyperparams[k] for k in keys]
    
    configs = []
    for combo in product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)
    
    return configs

def calculate_average_scores(questions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Return average score for each evaluation metric."""
    averages = {}
    for metric in EVAL_METRICS:
        scores = [q["evaluations"][metric].get("score", 0) for q in questions]
        averages[metric] = sum(scores) / len(scores) if scores else 0.0
    return averages

def run_benchmark(
    dataset: List[Dict],
    limit_questions: int = None,
    verbose: bool = False,
    max_workers: int = 1,
    timeout_seconds: int = None,
):
    """Run full benchmark with all configurations"""

    _validate_config()

    config_data = load_benchmark_config()
    modes = config_data["modes"]

    # Create output directory
    output_path = BENCHMARK_RESULTS_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    # Timestamp for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {
        "metadata": {
            "timestamp": timestamp,
            "benchmark_config": config_data,
            "total_questions": len(dataset) if limit_questions is None else min(limit_questions, len(dataset)),
            "workers": max_workers,
            "timeout_seconds": timeout_seconds,
        },
        "results": []
    }

    if limit_questions is not None:
        dataset = dataset[:limit_questions]

    rag_configs = generate_rag_configs(config_data["rag_hyperparams"])
    agentic_configs = [
        {"temperature": temp}
        for temp in config_data["agentic_hyperparams"]["temperature"]
    ]

    mode_configs = {
        "rag": rag_configs,
        "agentic": agentic_configs,
    }

    total_runs = sum(
        len(mode_configs[mode]) * len(dataset)
        for mode in modes
    )

    print(f"\n{'='*80}")
    print("BENCHMARK CONFIGURATION")
    print(f"{'='*80}")
    print(f"Questions: {len(dataset)}")
    for mode in modes:
        print(
            f"{mode.upper()}: {len(mode_configs[mode])} configurations × {len(dataset)} questions = "
            f"{len(mode_configs[mode]) * len(dataset)} runs"
        )
    if max_workers > 1:
        print(f"Workers: {max_workers}")
    print(f"{'─'*80}")
    print(f"Total: {total_runs} runs")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    progress_lock = threading.Lock()
    progress_state = {"count": 0}

    def _run_question(
        question_id: int,
        example: Dict[str, Any],
        mode: str,
        config: Dict[str, Any],
        timeout_seconds: int = None,
    ) -> Dict[str, Any]:
        question = example["inputs"]["question"]
        reference_answer = example["outputs"]["answer"]

        answer_dict = _call_chatbot(question, mode, timeout_seconds, config=config)
        # Metrics run sequentially here on purpose: this function itself runs inside a
        # worker thread of the per-question pool below. Submitting more work to that
        # same bounded pool from within one of its own workers can deadlock once all
        # workers are occupied waiting on sub-tasks that have no free thread to run on.
        evaluations = run_evaluation(question, answer_dict, reference_answer)
        request_time = answer_dict.get("request_time", 0.0)

        with progress_lock:
            progress_state["count"] += 1
            run_number = progress_state["count"]

        if verbose:
            scores_str = " | ".join([
                f"{k}: {v.get('score', 0):.1f}"
                for k, v in evaluations.items()
            ])
            print(f"  [{run_number}/{total_runs}] Q{question_id}: {question[:60]}... [{scores_str}]")

        return {
            "question_id": question_id,
            "question": question,
            "tags": example.get("tags", []),
            "reference_answer": reference_answer,
            "generated_answer": answer_dict.get("answer", ""),
            "evaluations": evaluations,
            "request_time": request_time,
        }

    def _run_mode(mode: str, configs: List[Dict[str, Any]]):
        print(f"\n{'='*80}")
        print(f"TESTING {mode.upper()} MODE")
        print(f"{'='*80}\n")

        for config_idx, config in enumerate(configs, 1):
            print(f"\n{mode.upper()} Configuration {config_idx}/{len(configs)}: {config}")
            print("-" * 80)

            if max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(_run_question, q_idx, example, mode, config, timeout_seconds): q_idx
                        for q_idx, example in enumerate(dataset, 1)
                    }

                    questions = []
                    for future in as_completed(futures):
                        questions.append(future.result())

                    questions.sort(key=lambda x: x["question_id"])
            else:
                questions = [
                    _run_question(q_idx, example, mode, config, timeout_seconds)
                    for q_idx, example in enumerate(dataset, 1)
                ]

            avg_scores = calculate_average_scores(questions)
            avg_request_time = (
                sum(q.get("request_time", 0.0) for q in questions) / len(questions)
                if questions else 0.0
            )
            config_results = {
                "mode": mode,
                "configuration": config,
                "questions": questions,
                "average_scores": avg_scores,
                "average_request_time": avg_request_time,
            }
            all_results["results"].append(config_results)

            print(f"  Average scores: {' | '.join([f'{k}: {v:.1f}' for k, v in avg_scores.items()])}")
            print(f"  Average request time: {avg_request_time:.3f}s")

    for mode in modes:
        _run_mode(mode, mode_configs[mode])

    results_file = output_path / f"benchmark_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}\n")

    print_benchmark_summary(all_results)
    return all_results

def print_benchmark_summary(results: Dict):
    """Print summary of benchmark results"""
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}\n")
    
    for run in results["results"]:
        mode = run["mode"]
        config = run["configuration"]
        avg_scores = run["average_scores"]
        
        config_str = ", ".join([f"{k}={v}" for k, v in config.items()]) if config else "default"
        print(f"Mode: {mode.upper()} | Config: {config_str}")
        print(f"  Correctness:         {avg_scores.get('correctness', 0):.1f}/10")
        print(f"  Relevance:           {avg_scores.get('relevance', 0):.1f}/10")
        print(f"  Groundedness:        {avg_scores.get('groundedness', 0):.1f}/10")
        print(f"  Retrieval Relevance: {avg_scores.get('retrieval_relevance', 0):.1f}/10")
        
        overall = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0
        print(f"  Overall:             {overall:.1f}/10\n")

def compare_results(results_dir: str = str(BENCHMARK_RESULTS_DIR)):
    """Compare multiple benchmark runs"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"No results found in {results_dir}")
        return
    
    print(f"\n{'='*80}")
    print("COMPARING BENCHMARK RESULTS")
    print(f"{'='*80}\n")
    
    result_files = sorted(results_path.glob("benchmark_*.json"))
    
    if not result_files:
        print("No benchmark results found")
        return
    
    all_data = []
    for rf in result_files:
        with open(rf, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.append((rf.name, data))
    
    print(f"Found {len(all_data)} result files\n")
    
    # Create comparison table
    for filename, data in all_data:
        print(f"File: {filename}")
        print(f"Timestamp: {data['metadata']['timestamp']}")
        print(f"Total questions: {data['metadata']['total_questions']}\n")
        
        for run in data["results"]:
            mode = run["mode"]
            config = run["configuration"]
            avg_scores = run["average_scores"]
            
            config_str = ", ".join([f"{k}={v}" for k, v in config.items()]) if config else "default"
            print(f"  {mode.upper():8} | {config_str:40} | ", end="")
            
            overall = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0
            print(f"Overall: {overall:.1f}")
        
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RAG and agentic modes")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions to test")
    parser.add_argument("--compare", action="store_true",
                        help="Compare existing results instead of running benchmark")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker threads for parallel execution, one per question (use -1 for all CPUs)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout in seconds for each chatbot request (default: no timeout)")

    args = parser.parse_args()
    workers = args.workers
    if workers == -1:
        workers = os.cpu_count() or 1
    elif workers < -1:
        parser.error("--workers must be -1 or a positive integer")

    if args.compare:
        compare_results()
    else:
        examples = _load_examples()
        run_benchmark(
            examples,
            limit_questions=args.limit,
            verbose=args.verbose,
            max_workers=workers,
            timeout_seconds=args.timeout,
        )
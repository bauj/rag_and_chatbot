#!/usr/bin/env python3
"""
Benchmark script for testing different modes and hyperparameters
Tests RAG and agentic modes with multiple configurations
"""

import subprocess
import sys
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from itertools import product

# Import from evaluator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluator import (
    MistralLLM, 
    LLM_API_URL, 
    LLM_MODEL, 
    LLM_API_KEY,
    SSL_CERTIF,
    CHATBOT_DIR,
    correctness,
    relevance,
    groundedness,
    retrieval_relevance,
    _sanitize_eval_result,
    _load_examples
)

class BenchmarkConfig:
    """Configuration for benchmark runs"""
    def __init__(self):
        self.modes = ["rag", "agentic"]
        self.rag_hyperparams = {
            "k": [3, 5, 10],  # Number of chunks to retrieve
            "temperature": [0.3, 0.7],
            "reranker_enabled": [True, False],
            "top_n": [5, 10],  # Top-N for reranker
            "deep_dive": [False]
        }
        self.agentic_hyperparams = {
            "temperature": [0.3, 0.7],
        }


def call_chatbot_json(question: str, mode: str, **kwargs) -> Dict[str, Any]:
    """Call chatbot.py with given parameters and return JSON result"""
    cmd = [
        sys.executable,
        "chatbot.py",
        "--mode", mode,
        "--question", question,
        "--json-output"
    ]
    
    # Add hyperparameters (if chatbot.py supports them in future)
    # For now, we just pass mode and question
    
    try:
        result = subprocess.run(
            cmd,
            cwd=CHATBOT_DIR,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            return {
                "answer": f"Error: {result.stderr}",
                "documents": [],
                "error": result.stderr
            }
        
        # Parse JSON from output
        stdout = result.stdout.strip()
        json_start = stdout.find('{')
        if json_start >= 0:
            json_str = stdout[json_start:]
            json_end = json_str.rfind('}')
            if json_end >= 0:
                json_str = json_str[:json_end + 1]
                chatbot_result = json.loads(json_str)
                
                answer = chatbot_result.get("answer", "No answer")
                sources = chatbot_result.get("sources", [])
                
                # Convert sources to document objects
                documents = []
                for source in sources:
                    class Document:
                        def __init__(self, content, metadata):
                            self.page_content = content
                            self.metadata = metadata
                    
                    doc = Document(
                        content=source.get("content", ""),
                        metadata={
                            "title": source.get("title", "Unknown"),
                            "module": source.get("module", "Unknown"),
                            "doc_category": source.get("doc_category", "Unknown"),
                            "doc_type": source.get("doc_type", "Unknown"),
                            "url": source.get("url", "")
                        }
                    )
                    documents.append(doc)
                
                return {
                    "answer": answer,
                    "documents": documents
                }
    except subprocess.TimeoutExpired:
        return {
            "answer": "Timeout",
            "documents": [],
            "error": "Chatbot request timed out"
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "documents": [],
            "error": str(e)
        }
    
    return {
        "answer": "Failed to parse response",
        "documents": [],
        "error": "JSON parsing failed"
    }


def run_evaluation(question: str, answer_dict: Dict, reference_answer: str) -> Dict[str, Dict]:
    """Run all evaluators on a single answer"""
    evaluations = {}
    
    inputs = {"question": question}
    
    # Correctness
    try:
        correct = correctness(inputs, answer_dict, {"answer": reference_answer})
        evaluations["correctness"] = _sanitize_eval_result(correct)
    except Exception as e:
        evaluations["correctness"] = {"score": 0.0, "explanation": f"Error: {str(e)}"}
    
    # Relevance
    try:
        relevant = relevance(inputs, answer_dict)
        evaluations["relevance"] = _sanitize_eval_result(relevant)
    except Exception as e:
        evaluations["relevance"] = {"score": 0.0, "explanation": f"Error: {str(e)}"}
    
    # Groundedness
    try:
        grounded = groundedness(inputs, answer_dict)
        evaluations["groundedness"] = _sanitize_eval_result(grounded)
    except Exception as e:
        evaluations["groundedness"] = {"score": 0.0, "explanation": f"Error: {str(e)}"}
    
    # Retrieval Relevance
    try:
        retrieval_rel = retrieval_relevance(inputs, answer_dict)
        evaluations["retrieval_relevance"] = _sanitize_eval_result(retrieval_rel)
    except Exception as e:
        evaluations["retrieval_relevance"] = {"score": 0.0, "explanation": f"Error: {str(e)}"}
    
    return evaluations


def generate_rag_configs(hyperparams: Dict) -> List[Dict]:
    """Generate all RAG hyperparameter combinations"""
    keys = list(hyperparams.keys())
    values = [hyperparams[k] for k in keys]
    
    configs = []
    for combo in product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)
    
    return configs


def run_benchmark(
    dataset: List[Dict],
    output_dir: str = "benchmark_results",
    modes: List[str] = None,
    limit_questions: int = None,
    verbose: bool = False
):
    """Run full benchmark with all configurations"""
    
    if modes is None:
        modes = ["rag", "agentic"]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Timestamp for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_results = {
        "metadata": {
            "timestamp": timestamp,
            "modes": modes,
            "total_questions": len(dataset) if limit_questions is None else min(limit_questions, len(dataset))
        },
        "results": []
    }
    
    # Limit dataset if requested
    if limit_questions:
        dataset = dataset[:limit_questions]
    
    # RAG configurations
    config = BenchmarkConfig()
    rag_configs = generate_rag_configs(config.rag_hyperparams)
    agentic_configs = [
        {"temperature": temp}
        for temp in config.agentic_hyperparams.get("temperature", [0.3, 0.7])
    ]
    
    rag_runs = len(rag_configs) * len(dataset) if "rag" in modes else 0
    agentic_runs = len(agentic_configs) * len(dataset) if "agentic" in modes else 0
    total_runs = rag_runs + agentic_runs
    
    print(f"\n{'='*80}")
    print("BENCHMARK CONFIGURATION")
    print(f"{'='*80}")
    print(f"Questions: {len(dataset)}")
    if "rag" in modes:
        print(f"RAG: {len(rag_configs)} configurations × {len(dataset)} questions = {rag_runs} runs")
    if "agentic" in modes:
        print(f"Agentic: {len(agentic_configs)} configurations × {len(dataset)} questions = {agentic_runs} runs")
    print(f"{'─'*80}")
    print(f"Total: {total_runs} runs")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")
    
    current_run = 0
    
    # Test RAG mode with different configurations
    if "rag" in modes:
        print(f"\n{'='*80}")
        print("TESTING RAG MODE")
        print(f"{'='*80}\n")
        
        for config_idx, rag_config in enumerate(rag_configs, 1):
            print(f"\nRAG Configuration {config_idx}/{len(rag_configs)}: {rag_config}")
            print("-" * 80)
            
            config_results = {
                "mode": "rag",
                "configuration": rag_config,
                "questions": []
            }
            
            for q_idx, example in enumerate(dataset, 1):
                current_run += 1
                question = example["inputs"]["question"]
                reference_answer = example["outputs"]["answer"]
                
                print(f"  [{current_run}/{total_runs}] Q{q_idx}: {question[:60]}...", end=" ", flush=True)
                
                # Call chatbot in RAG mode
                answer_dict = call_chatbot_json(question, "rag")
                
                # Run evaluations
                evaluations = run_evaluation(question, answer_dict, reference_answer)
                
                # Store results
                question_result = {
                    "question_id": q_idx,
                    "question": question,
                    "reference_answer": reference_answer,
                    "generated_answer": answer_dict.get("answer", ""),
                    "evaluations": evaluations
                }
                config_results["questions"].append(question_result)
                
                scores_str = " | ".join([
                    f"{k}: {v.get('score', 0):.1f}"
                    for k, v in evaluations.items()
                ])
                print(f"[{scores_str}]")
            
            # Calculate average scores for this configuration
            avg_scores = {}
            for metric in ["correctness", "relevance", "groundedness", "retrieval_relevance"]:
                scores = [q["evaluations"][metric].get("score", 0) for q in config_results["questions"]]
                avg_scores[metric] = sum(scores) / len(scores) if scores else 0
            
            config_results["average_scores"] = avg_scores
            all_results["results"].append(config_results)
            
            print(f"  Average scores: {' | '.join([f'{k}: {v:.1f}' for k, v in avg_scores.items()])}")
    
    # Test agentic mode with different configurations
    if "agentic" in modes:
        print(f"\n{'='*80}")
        print("TESTING AGENTIC MODE")
        print(f"{'='*80}\n")
        
        # Generate agentic configurations
        config = BenchmarkConfig()
        agentic_configs = [
            {"temperature": temp}
            for temp in config.agentic_hyperparams.get("temperature", [0.3, 0.7])
        ]
        
        for config_idx, agentic_config in enumerate(agentic_configs, 1):
            print(f"\nAgentic Configuration {config_idx}/{len(agentic_configs)}: {agentic_config}")
            print("-" * 80)
            
            config_results = {
                "mode": "agentic",
                "configuration": agentic_config,
                "questions": []
            }
            
            for q_idx, example in enumerate(dataset, 1):
                current_run += 1
                question = example["inputs"]["question"]
                reference_answer = example["outputs"]["answer"]
                
                print(f"  [{current_run}/{total_runs}] Q{q_idx}: {question[:60]}...", end=" ", flush=True)
                
                # Call chatbot in agentic mode
                answer_dict = call_chatbot_json(question, "agentic")
                
                # Run evaluations
                evaluations = run_evaluation(question, answer_dict, reference_answer)
                
                # Store results
                question_result = {
                    "question_id": q_idx,
                    "question": question,
                    "reference_answer": reference_answer,
                    "generated_answer": answer_dict.get("answer", ""),
                    "evaluations": evaluations
                }
                config_results["questions"].append(question_result)
                
                scores_str = " | ".join([
                    f"{k}: {v.get('score', 0):.1f}"
                    for k, v in evaluations.items()
                ])
                print(f"[{scores_str}]")
            
            # Calculate average scores for this agentic configuration
            avg_scores = {}
            for metric in ["correctness", "relevance", "groundedness", "retrieval_relevance"]:
                scores = [q["evaluations"][metric].get("score", 0) for q in config_results["questions"]]
                avg_scores[metric] = sum(scores) / len(scores) if scores else 0
            
            config_results["average_scores"] = avg_scores
            all_results["results"].append(config_results)
            
            print(f"  Average scores: {' | '.join([f'{k}: {v:.1f}' for k, v in avg_scores.items()])}")
    
    # Save results
    results_file = output_path / f"benchmark_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}\n")
    
    # Print summary
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


def compare_results(results_dir: str = "benchmark_results"):
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
    parser.add_argument("--modes", nargs="+", default=["rag", "agentic"],
                        help="Modes to test (rag, agentic)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions to test")
    parser.add_argument("--output-dir", default="benchmark_results",
                        help="Output directory for results")
    parser.add_argument("--compare", action="store_true",
                        help="Compare existing results instead of running benchmark")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_results(args.output_dir)
    else:
        examples = _load_examples()
        run_benchmark(
            examples,
            output_dir=args.output_dir,
            modes=args.modes,
            limit_questions=args.limit,
            verbose=args.verbose
        )

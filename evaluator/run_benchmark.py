#!/usr/bin/env python3
"""
Helper script to run benchmarks with easy configuration
"""

import argparse
import sys
from pathlib import Path

# Add evaluator directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import run_benchmark, compare_results, _load_examples


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark suite for RAG and agentic chatbot modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark on both modes with all 11 questions
  python3 run_benchmark.py
  
  # Test only agentic mode with first 3 questions
  python3 run_benchmark.py --modes agentic --limit 3
  
  # Test only RAG mode
  python3 run_benchmark.py --modes rag
  
  # Compare previous benchmark results
  python3 run_benchmark.py --compare
  
  # Quick test with fewer questions
  python3 run_benchmark.py --limit 2 --modes rag agentic
        """
    )
    
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["rag", "agentic"],
        default=["rag", "agentic"],
        help="Which modes to test (default: both)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to test (useful for quick tests)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare existing benchmark results instead of running new benchmark"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_results(args.output_dir)
    else:
        try:
            # Load examples and run benchmark
            examples = _load_examples()
            
            if args.limit:
                print(f"Testing with {args.limit} questions (out of {len(examples)})")
            else:
                print(f"Testing with all {len(examples)} questions")
            
            run_benchmark(
                examples,
                output_dir=args.output_dir,
                modes=args.modes,
                limit_questions=args.limit,
                verbose=args.verbose
            )
        except (FileNotFoundError, ValueError) as err:
            print(f"Error: {err}")
            sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analysis script for benchmark results
Generates detailed comparison tables and insights
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class BenchmarkAnalyzer:
    """Analyze and compare benchmark results"""
    
    def __init__(self, results_file: str):
        """Load benchmark results from file"""
        with open(results_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.results_file = results_file
    
    def print_header(self):
        """Print header with metadata"""
        print(f"\n{'='*100}")
        print("BENCHMARK ANALYSIS")
        print(f"{'='*100}\n")
        print(f"File: {self.results_file}")
        print(f"Timestamp: {self.data['metadata']['timestamp']}")
        print(f"Total questions: {self.data['metadata']['total_questions']}")
        print(f"Modes tested: {', '.join(self.data['metadata']['modes'])}\n")
    
    def print_summary_table(self):
        """Print summary table of all configurations"""
        print(f"{'='*100}")
        print("SUMMARY TABLE - AVERAGE SCORES BY CONFIGURATION")
        print(f"{'='*100}\n")
        
        # Prepare data
        configs_data = []
        for run in self.data["results"]:
            mode = run["mode"].upper()
            config = run["configuration"]
            scores = run["average_scores"]
            overall = sum(scores.values()) / len(scores) if scores else 0
            
            config_str = self._format_config(config)
            configs_data.append({
                "mode": mode,
                "config": config_str,
                "correctness": scores.get("correctness", 0),
                "relevance": scores.get("relevance", 0),
                "groundedness": scores.get("groundedness", 0),
                "retrieval_relevance": scores.get("retrieval_relevance", 0),
                "overall": overall
            })
        
        # Print table header
        print(f"{'Mode':<10} {'Configuration':<60} {'Correct':<8} {'Relev':<8} {'Ground':<8} {'Retriev':<8} {'Overall':<8}")
        print("-" * 100)
        
        # Print rows
        for data in configs_data:
            print(
                f"{data['mode']:<10} "
                f"{data['config']:<60} "
                f"{data['correctness']:<8.1f} "
                f"{data['relevance']:<8.1f} "
                f"{data['groundedness']:<8.1f} "
                f"{data['retrieval_relevance']:<8.1f} "
                f"{data['overall']:<8.1f}"
            )
        
        print()
    
    def print_best_worst(self):
        """Print best and worst configurations"""
        print(f"{'='*100}")
        print("BEST & WORST CONFIGURATIONS")
        print(f"{'='*100}\n")
        
        configs_data = []
        for run in self.data["results"]:
            mode = run["mode"]
            config = run["configuration"]
            scores = run["average_scores"]
            overall = sum(scores.values()) / len(scores) if scores else 0
            
            config_str = self._format_config(config)
            configs_data.append({
                "mode": mode,
                "config": config_str,
                "overall": overall,
                "scores": scores
            })
        
        # Sort by overall score
        sorted_configs = sorted(configs_data, key=lambda x: x["overall"], reverse=True)
        
        print("TOP 3 CONFIGURATIONS:\n")
        for i, config in enumerate(sorted_configs[:3], 1):
            print(f"{i}. {config['mode'].upper()} - {config['config']}")
            print(f"   Overall: {config['overall']:.1f}/10")
            print(f"   Correctness: {config['scores'].get('correctness', 0):.1f} | ", end="")
            print(f"Relevance: {config['scores'].get('relevance', 0):.1f} | ", end="")
            print(f"Groundedness: {config['scores'].get('groundedness', 0):.1f} | ", end="")
            print(f"Retrieval: {config['scores'].get('retrieval_relevance', 0):.1f}\n")
        
        print("BOTTOM 3 CONFIGURATIONS:\n")
        for i, config in enumerate(sorted_configs[-3:], 1):
            print(f"{i}. {config['mode'].upper()} - {config['config']}")
            print(f"   Overall: {config['overall']:.1f}/10")
            print(f"   Correctness: {config['scores'].get('correctness', 0):.1f} | ", end="")
            print(f"Relevance: {config['scores'].get('relevance', 0):.1f} | ", end="")
            print(f"Groundedness: {config['scores'].get('groundedness', 0):.1f} | ", end="")
            print(f"Retrieval: {config['scores'].get('retrieval_relevance', 0):.1f}\n")
    
    def print_per_question_analysis(self):
        """Print detailed per-question analysis"""
        print(f"{'='*100}")
        print("PER-QUESTION ANALYSIS")
        print(f"{'='*100}\n")
        
        # Group results by question
        questions_results = {}
        for run in self.data["results"]:
            mode = run["mode"]
            config = run["configuration"]
            config_str = self._format_config(config)
            
            for question_data in run["questions"]:
                q_id = question_data["question_id"]
                if q_id not in questions_results:
                    questions_results[q_id] = {
                        "question": question_data["question"],
                        "configs": []
                    }
                
                scores = question_data["evaluations"]
                overall = sum(s.get("score", 0) for s in scores.values()) / len(scores) if scores else 0
                
                questions_results[q_id]["configs"].append({
                    "mode": mode,
                    "config": config_str,
                    "scores": scores,
                    "overall": overall
                })
        
        # Print for each question
        for q_id in sorted(questions_results.keys()):
            q_data = questions_results[q_id]
            print(f"Question {q_id}: {q_data['question'][:70]}...")
            print("-" * 100)
            
            # Find best config for this question
            best_config = max(q_data["configs"], key=lambda x: x["overall"])
            worst_config = min(q_data["configs"], key=lambda x: x["overall"])
            
            print(f"  Best:  {best_config['mode'].upper()} - {best_config['config']}")
            print(f"         Overall: {best_config['overall']:.1f}, Scores: {self._format_scores(best_config['scores'])}")
            print(f"  Worst: {worst_config['mode'].upper()} - {worst_config['config']}")
            print(f"         Overall: {worst_config['overall']:.1f}, Scores: {self._format_scores(worst_config['scores'])}\n")
    
    def print_mode_comparison(self):
        """Compare modes across all questions"""
        print(f"{'='*100}")
        print("MODE COMPARISON (AVERAGED ACROSS ALL QUESTIONS)")
        print(f"{'='*100}\n")
        
        mode_stats = {}
        for run in self.data["results"]:
            mode = run["mode"]
            if mode not in mode_stats:
                mode_stats[mode] = {
                    "scores": {"correctness": 0, "relevance": 0, "groundedness": 0, "retrieval_relevance": 0},
                    "count": 0
                }
            
            avg = run["average_scores"]
            for metric in mode_stats[mode]["scores"].keys():
                mode_stats[mode]["scores"][metric] += avg.get(metric, 0)
            mode_stats[mode]["count"] += 1
        
        # Calculate averages
        for mode in mode_stats:
            count = mode_stats[mode]["count"]
            for metric in mode_stats[mode]["scores"]:
                mode_stats[mode]["scores"][metric] /= count
        
        # Print comparison
        print(f"{'Mode':<10} {'Correctness':<15} {'Relevance':<15} {'Groundedness':<15} {'Retrieval':<15}")
        print("-" * 100)
        
        for mode in sorted(mode_stats.keys()):
            scores = mode_stats[mode]["scores"]
            print(
                f"{mode.upper():<10} "
                f"{scores['correctness']:<15.1f} "
                f"{scores['relevance']:<15.1f} "
                f"{scores['groundedness']:<15.1f} "
                f"{scores['retrieval_relevance']:<15.1f}"
            )
        
        print()
    
    def print_metric_statistics(self):
        """Print statistics for each metric"""
        print(f"{'='*100}")
        print("METRIC STATISTICS ACROSS ALL CONFIGURATIONS")
        print(f"{'='*100}\n")
        
        metrics = ["correctness", "relevance", "groundedness", "retrieval_relevance"]
        
        for metric in metrics:
            scores = []
            for run in self.data["results"]:
                score = run["average_scores"].get(metric, 0)
                scores.append(score)
            
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                avg_score = sum(scores) / len(scores)
                
                print(f"{metric.upper()}:")
                print(f"  Min:     {min_score:.1f}")
                print(f"  Max:     {max_score:.1f}")
                print(f"  Average: {avg_score:.1f}")
                print(f"  Range:   {max_score - min_score:.1f}\n")
    
    @staticmethod
    def _format_config(config: Dict) -> str:
        """Format configuration dictionary as string"""
        if not config:
            return "default"
        return ", ".join([f"{k}={v}" for k, v in config.items()])
    
    @staticmethod
    def _format_scores(scores: Dict) -> str:
        """Format scores dictionary as string"""
        return ", ".join([
            f"{k.replace('_', ' ').title()}: {v.get('score', 0):.1f}"
            for k, v in scores.items()
        ])
    
    def generate_full_report(self, include_graphs: bool = False):
        """Generate complete analysis report"""
        self.print_header()
        self.print_summary_table()
        self.print_best_worst()
        self.print_mode_comparison()
        self.print_metric_statistics()
        self.print_per_question_analysis()
        
        if include_graphs:
            self.generate_graphs()
        
        print(f"{'='*100}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*100}\n")
    
    def generate_graphs(self, output_dir: str = "graphs"):
        """Generate various analysis graphs"""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available. Install with: pip install matplotlib")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\nGenerating graphs in {output_path}...")
        
        self._graph_scores_by_question(output_path)
        self._graph_metric_distributions(output_path)
        self._graph_scores_by_tag(output_path)
        
        print("Graphs generated successfully!")
    
    def _graph_scores_by_question(self, output_path: Path):
        """Generate bar chart of average scores per question"""
        questions_data = {}
        
        for run in self.data["results"]:
            for q_data in run["questions"]:
                q_id = q_data["question_id"]
                if q_id not in questions_data:
                    questions_data[q_id] = {
                        "question": q_data["question"][:50] + "..." if len(q_data["question"]) > 50 else q_data["question"],
                        "scores": {"correctness": [], "relevance": [], "groundedness": [], "retrieval_relevance": []}
                    }
                
                for metric, eval_data in q_data["evaluations"].items():
                    score = eval_data.get("score", 0)
                    questions_data[q_id]["scores"][metric].append(score)
        
        # Calculate averages
        question_ids = sorted(questions_data.keys())
        metrics = ["correctness", "relevance", "groundedness", "retrieval_relevance"]
        
        fig, ax = plt.subplots(figsize=(15, 8))
        x = range(len(question_ids))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            scores = [sum(questions_data[q]["scores"][metric]) / len(questions_data[q]["scores"][metric]) 
                     for q in question_ids]
            ax.bar([xi + i*width for xi in x], scores, width, label=metric.title())
        
        ax.set_xlabel('Question ID')
        ax.set_ylabel('Average Score')
        ax.set_title('Average Scores by Question and Metric')
        ax.set_xticks([xi + width*1.5 for xi in x])
        ax.set_xticklabels(question_ids)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "scores_by_question.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _graph_metric_distributions(self, output_path: Path):
        """Generate histograms of score distributions per metric"""
        metrics = ["correctness", "relevance", "groundedness", "retrieval_relevance"]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            scores = []
            for run in self.data["results"]:
                for q_data in run["questions"]:
                    score = q_data["evaluations"].get(metric, {}).get("score", 0)
                    scores.append(score)
            
            axes[i].hist(scores, bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{metric.title()} Distribution')
            axes[i].set_xlabel('Score')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "metric_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _graph_scores_by_tag(self, output_path: Path):
        """Generate bar chart of average scores by tag"""
        tag_data = {}
        
        for run in self.data["results"]:
            for q_data in run["questions"]:
                tags = q_data.get("tags", [])
                if not tags:
                    tags = ["untagged"]
                
                for tag in tags:
                    if tag not in tag_data:
                        tag_data[tag] = {"correctness": [], "relevance": [], "groundedness": [], "retrieval_relevance": []}
                    
                    for metric, eval_data in q_data["evaluations"].items():
                        score = eval_data.get("score", 0)
                        tag_data[tag][metric].append(score)
        
        if not tag_data:
            return
        
        # Calculate averages
        tags = sorted(tag_data.keys())
        metrics = ["correctness", "relevance", "groundedness", "retrieval_relevance"]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(tags))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            scores = [sum(tag_data[tag][metric]) / len(tag_data[tag][metric]) if tag_data[tag][metric] else 0
                     for tag in tags]
            ax.bar([xi + i*width for xi in x], scores, width, label=metric.title())
        
        ax.set_xlabel('Tag')
        ax.set_ylabel('Average Score')
        ax.set_title('Average Scores by Tag and Metric')
        ax.set_xticks([xi + width*1.5 for xi in x])
        ax.set_xticklabels(tags, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "scores_by_tag.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("results_file", nargs="?", help="Results JSON file to analyze")
    parser.add_argument("--list", action="store_true", help="List available results files")
    parser.add_argument("--latest", action="store_true", help="Analyze latest results file")
    parser.add_argument("--graphs", action="store_true", help="Generate analysis graphs (requires matplotlib)")
    
    args = parser.parse_args()
    
    # Find results files
    results_dir = Path("benchmark_results")
    if not results_dir.exists():
        print("No benchmark_results directory found")
        return
    
    result_files = sorted(results_dir.glob("benchmark_*.json"), reverse=True)
    
    if args.list:
        print("\nAvailable benchmark results:\n")
        for i, rf in enumerate(result_files, 1):
            print(f"{i}. {rf.name}")
        return
    
    # Determine which file to analyze
    if args.latest:
        if not result_files:
            print("No results files found")
            return
        results_file = result_files[0]
    elif args.results_file:
        results_file = Path(args.results_file)
        if not results_file.exists():
            results_file = results_dir / args.results_file
            if not results_file.exists():
                print(f"Results file not found: {args.results_file}")
                return
    else:
        if not result_files:
            print("No results files found. Run benchmark first.")
            return
        print("Available results files:")
        for i, rf in enumerate(result_files, 1):
            print(f"{i}. {rf.name}")
        
        choice = input("\nSelect a file to analyze (number): ").strip()
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(result_files):
                results_file = result_files[choice_idx]
            else:
                print("Invalid choice")
                return
        except ValueError:
            print("Invalid input")
            return
    
    # Analyze
    analyzer = BenchmarkAnalyzer(str(results_file))
    analyzer.generate_full_report(include_graphs=args.graphs)


if __name__ == "__main__":
    main()

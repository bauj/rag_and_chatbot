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
import statistics

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class BenchmarkAnalyzer:
    """Analyze and compare benchmark results"""

    # Tags that describe question difficulty / language rather than topic.
    # Anything not in these sets is treated as a topic tag (Sampler, Sensitivity, ...).
    DIFFICULTY_TAGS = {"easy", "medium", "hard"}
    LANGUAGE_TAGS = {"Fr", "En"}

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
    
    def print_response_time_analysis(self):
        """Print response time analysis by configuration and question"""
        print(f"{'='*100}")
        print("RESPONSE TIME ANALYSIS")
        print(f"{'='*100}\n")
        
        # Summary by configuration
        print("AVERAGE RESPONSE TIME BY CONFIGURATION:\n")
        print(f"{'Mode':<10} {'Configuration':<60} {'Avg Time (s)':<15} {'Min (s)':<12} {'Max (s)':<12}")
        print("-" * 100)
        
        for run in self.data["results"]:
            mode = run["mode"].upper()
            config = self._format_config(run["configuration"])
            avg_time = run.get("average_request_time", 0)
            
            # Calculate min/max for questions
            times = [q.get("request_time", 0) for q in run["questions"]]
            min_time = min(times) if times else 0
            max_time = max(times) if times else 0
            
            print(
                f"{mode:<10} "
                f"{config:<60} "
                f"{avg_time:<15.4f} "
                f"{min_time:<12.4f} "
                f"{max_time:<12.4f}"
            )
        
        print()
    
    def print_response_time_by_question(self):
        """Print response times grouped by question"""
        print(f"{'='*100}")
        print("RESPONSE TIME BY QUESTION")
        print(f"{'='*100}\n")
        
        # Group by question
        questions_times = {}
        for run in self.data["results"]:
            mode = run["mode"]
            config = self._format_config(run["configuration"])
            
            for q_data in run["questions"]:
                q_id = q_data["question_id"]
                if q_id not in questions_times:
                    questions_times[q_id] = {
                        "question": q_data["question"][:70],
                        "times": []
                    }
                
                request_time = q_data.get("request_time", 0)
                questions_times[q_id]["times"].append({
                    "mode": mode,
                    "config": config,
                    "time": request_time
                })
        
        # Print for each question
        for q_id in sorted(questions_times.keys()):
            q_data = questions_times[q_id]
            times_list = q_data["times"]
            
            avg_time = sum(t["time"] for t in times_list) / len(times_list) if times_list else 0
            min_time = min(t["time"] for t in times_list) if times_list else 0
            max_time = max(t["time"] for t in times_list) if times_list else 0
            
            print(f"Q{q_id}: {q_data['question']}")
            print(f"  Average: {avg_time:.4f}s | Min: {min_time:.4f}s | Max: {max_time:.4f}s")
            
            # Show fastest and slowest configs
            fastest = min(times_list, key=lambda x: x["time"])
            slowest = max(times_list, key=lambda x: x["time"])
            print(f"  Fastest: {fastest['mode'].upper()} ({fastest['time']:.4f}s)")
            print(f"  Slowest: {slowest['mode'].upper()} ({slowest['time']:.4f}s)\n")
    
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
    
    @staticmethod
    def _normalize_tags(raw_tags):
        """Normalize raw tag data into a list of tag strings"""
        if isinstance(raw_tags, str):
            tags = [raw_tags]
        elif raw_tags is None:
            tags = []
        elif isinstance(raw_tags, (list, tuple)):
            tags = [str(tag) for tag in raw_tags if tag is not None]
        else:
            tags = [str(raw_tags)]

        if not tags:
            tags = ["untagged"]
        return tags

    @classmethod
    def _classify_tags(cls, tags: List[str]) -> Dict[str, List[str]]:
        """Split a normalized tag list into topic / difficulty / language buckets."""
        buckets = {"topic": [], "difficulty": [], "language": []}
        for tag in tags:
            if tag in cls.DIFFICULTY_TAGS:
                buckets["difficulty"].append(tag)
            elif tag in cls.LANGUAGE_TAGS:
                buckets["language"].append(tag)
            else:
                buckets["topic"].append(tag)
        return buckets

    def _get_config_labels(self):
        """Generate short and full config labels for all runs"""
        config_labels_short = {}
        config_labels_full = {}
        config_counter = {}
        all_configs = []
        
        for run in self.data["results"]:
            mode = run["mode"]
            config = self._format_config(run["configuration"])
            config_label = f"{mode.upper()}\n{config}"
            
            if mode not in config_counter:
                config_counter[mode] = 0
            config_counter[mode] += 1
            short_label = f"{mode.upper()[:3]}-{config_counter[mode]}"
            
            all_configs.append(short_label)
            config_labels_short[config_label] = short_label
            config_labels_full[short_label] = config_label
        
        return sorted(list(set(all_configs))), config_labels_short, config_labels_full
    
    def _get_colors(self, n_colors):
        """Get color palette for n_colors items"""
        return plt.cm.Set3(range(n_colors))
    
    def _save_graph(self, fig, output_path, filename):
        """Save graph with consistent settings"""
        plt.tight_layout()
        plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_full_report(self, include_graphs: bool = False):
        """Generate complete analysis report"""
        self.print_header()
        self.print_summary_table()
        self.print_best_worst()
        self.print_mode_comparison()
        self.print_metric_statistics()
        self.print_response_time_analysis()
        self.print_response_time_by_question()
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
        
        # Core analysis graphs
        print("  Generating: Quality vs Response Time...")
        self._graph_quality_vs_time(output_path)
        
        print("  Generating: Metric Distribution by Config...")
        self._graph_metric_distributions(output_path)
        
        print("  Generating: Overall Performance Heatmap...")
        self._graph_performance_heatmap(output_path)
        
        print("  Generating: Scores by Question and Config...")
        self._graph_scores_by_question_and_config(output_path)
        
        print("  Generating: Scores by Tag and Config...")
        self._graph_scores_by_tag(output_path)

        print("  Generating: Response Time by Config...")
        self._graph_response_time_by_config(output_path)

        print("  Generating: Response Time by Question and Config...")
        self._graph_response_time_by_question(output_path)

        print("  Generating: Scores by Difficulty and Config...")
        self._graph_scores_by_difficulty(output_path)

        print("  Generating: Scores by Language and Config...")
        self._graph_scores_by_language(output_path)

        print("Graphs generated successfully!")
    
    def _graph_scores_by_question_and_config(self, output_path: Path):
        """Generate grouped bar chart of overall scores per question and config"""
        question_config_scores = {}
        all_configs, config_labels_short, config_labels_full = self._get_config_labels()
        
        for run in self.data["results"]:
            mode = run["mode"]
            config = self._format_config(run["configuration"])
            config_label = f"{mode.upper()}\n{config}"
            
            for q_data in run["questions"]:
                q_id = q_data["question_id"]
                if q_id not in question_config_scores:
                    question_config_scores[q_id] = {}
                
                # Calculate overall score for this question-config
                scores = [eval_data.get("score", 0) for eval_data in q_data["evaluations"].values()]
                overall_score = sum(scores) / len(scores) if scores else 0
                question_config_scores[q_id][config_label] = overall_score
        
        if not question_config_scores or not all_configs:
            return

        q_ids = sorted(question_config_scores.keys())[:15]  # Limit to first 15 questions for readability

        # Prepare plot
        fig, ax = plt.subplots(figsize=(16, 7))
        x = range(len(q_ids))
        width = 0.8 / len(all_configs)
        colors = self._get_colors(len(all_configs))
        
        for config_idx, short_label in enumerate(all_configs):
            config_label = config_labels_full[short_label]
            scores = [question_config_scores[q_id].get(config_label, 0) for q_id in q_ids]
            offset = width * (config_idx - len(all_configs)/2 + 0.5)
            ax.bar([xi + offset for xi in x], scores, width, label=short_label, 
                   color=colors[config_idx], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Question ID')
        ax.set_ylabel('Overall Score')
        ax.set_title('Overall Score by Question and Configuration')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Q{q_id}' for q_id in q_ids])
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 10)
        
        self._save_graph(fig, output_path, "04_scores_by_question_config.png")
    
    def _graph_metric_distributions(self, output_path: Path):
        """Generate box plots of score distributions per metric, distinguishing by configuration"""
        metrics = ["correctness", "relevance", "groundedness", "retrieval_relevance"]
        
        # Collect data by metric and configuration
        config_data = {}
        all_configs, config_labels_short, _ = self._get_config_labels()

        if not all_configs:
            return

        for run in self.data["results"]:
            mode = run["mode"]
            config = self._format_config(run["configuration"])
            config_label = f"{mode.upper()}\n{config}"
            short_label = config_labels_short[config_label]

            for q_data in run["questions"]:
                for metric in metrics:
                    if metric not in config_data:
                        config_data[metric] = {}
                    if short_label not in config_data[metric]:
                        config_data[metric][short_label] = []
                    
                    score = q_data["evaluations"].get(metric, {}).get("score", 0)
                    config_data[metric][short_label].append(score)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        colors = self._get_colors(len(all_configs))
        
        for i, metric in enumerate(metrics):
            # Prepare data for box plot
            data_to_plot = [config_data.get(metric, {}).get(config, []) for config in all_configs]
            
            bp = axes[i].boxplot(data_to_plot, tick_labels=all_configs, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[i].set_title(f'{metric.title()} Distribution by Configuration', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Score')
            axes[i].grid(True, alpha=0.3, axis='y')
            axes[i].tick_params(axis='x', rotation=0)
        
        self._save_graph(fig, output_path, "02_metric_distributions.png")
    
    def _graph_scores_by_dimension(self, output_path: Path, get_labels, dimension_name: str, filename: str):
        """Generate subplots of scores by an arbitrary tag dimension (topic, difficulty, language),
        with box plots for each config grouped by label. `get_labels(q_data)` returns the list of
        labels along that dimension for one question (e.g. its topic tags, or its difficulty tag)."""
        metrics = ["correctness", "relevance", "groundedness", "retrieval_relevance"]

        # Collect data by label, config, and metric
        label_config_data = {}
        all_configs, config_labels_short, _ = self._get_config_labels()

        for run in self.data["results"]:
            mode = run["mode"]
            config = self._format_config(run["configuration"])
            config_label = f"{mode.upper()}\n{config}"
            short_label = config_labels_short[config_label]

            for q_data in run["questions"]:
                labels = get_labels(q_data)

                for label in labels:
                    if label not in label_config_data:
                        label_config_data[label] = {}
                    if short_label not in label_config_data[label]:
                        label_config_data[label][short_label] = {
                            "correctness": [],
                            "relevance": [],
                            "groundedness": [],
                            "retrieval_relevance": []
                        }

                    for metric, eval_data in q_data["evaluations"].items():
                        score = eval_data.get("score", 0)
                        label_config_data[label][short_label][metric].append(score)

        if not label_config_data:
            return

        labels_sorted = sorted(label_config_data.keys())

        # Create subplots for each metric
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        colors = self._get_colors(len(all_configs))

        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]

            # Prepare data: for each label, collect all config data
            box_data = []
            box_labels = []
            box_colors = []

            for label_idx, label in enumerate(labels_sorted):
                for config_idx, config in enumerate(all_configs):
                    label_config_scores = label_config_data[label].get(config, {}).get(metric, [])
                    if label_config_scores:
                        box_data.append(label_config_scores)
                        box_labels.append(config)
                        box_colors.append(colors[config_idx])

                # Add spacing between labels
                if label_idx < len(labels_sorted) - 1:
                    box_data.append([])
                    box_labels.append('')
                    box_colors.append('white')

            # Create box plot
            bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.6,
                           positions=range(len(box_data)))

            # Color the boxes
            for patch, color in zip(bp['boxes'], box_colors):
                if patch.get_facecolor() != (1, 1, 1, 1):  # Skip spacing boxes
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

            # Customize box plot appearance
            for whisker in bp['whiskers']:
                whisker.set(linewidth=1, alpha=0.7)
            for cap in bp['caps']:
                cap.set(linewidth=1, alpha=0.7)
            for median in bp['medians']:
                median.set(linewidth=2, color='red')

            # Add label separators with headers
            ax.set_ylabel('Score')
            ax.set_title(f'{metric.title()} by {dimension_name} and Configuration', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45, labelsize=8)

            # Add label headers on top
            label_positions = []
            pos = 0
            for label_idx, label in enumerate(labels_sorted):
                start_pos = pos
                pos += len(all_configs)
                end_pos = pos - 1
                label_positions.append((start_pos, end_pos, label))
                if label_idx < len(labels_sorted) - 1:
                    pos += 1

            # Draw vertical lines between labels and annotate label groups
            for label_idx, (start, end, label) in enumerate(label_positions):
                if label_idx > 0:
                    ax.axvline(x=start - 0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
                center = (start + end) / 2
                ax.text(center, 10.5, label, ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

            ax.set_xlabel(f'Configuration grouped by {dimension_name.lower()}')
            ax.set_ylim(0, 11)

        self._save_graph(fig, output_path, filename)

    def _graph_scores_by_tag(self, output_path: Path):
        """Scores broken down by topic tag (e.g. Sampler, Sensitivity) and configuration."""
        def topic_labels(q_data):
            tags = self._normalize_tags(q_data.get("tags", []))
            topics = self._classify_tags(tags)["topic"]
            return topics or ["untagged"]

        self._graph_scores_by_dimension(output_path, topic_labels, "Topic", "05_scores_by_tag_config.png")

    def _graph_scores_by_difficulty(self, output_path: Path):
        """Scores broken down by difficulty tag (easy/medium/hard) and configuration."""
        def difficulty_labels(q_data):
            tags = self._normalize_tags(q_data.get("tags", []))
            difficulties = self._classify_tags(tags)["difficulty"]
            return difficulties or ["unspecified"]

        self._graph_scores_by_dimension(output_path, difficulty_labels, "Difficulty", "08_scores_by_difficulty_config.png")

    def _graph_scores_by_language(self, output_path: Path):
        """Scores broken down by language tag (Fr/En) and configuration."""
        def language_labels(q_data):
            tags = self._normalize_tags(q_data.get("tags", []))
            languages = self._classify_tags(tags)["language"]
            return languages or ["unspecified"]

        self._graph_scores_by_dimension(output_path, language_labels, "Language", "09_scores_by_language_config.png")
    
    def _graph_response_time_by_config(self, output_path: Path):
        """Generate bar chart of response time by configuration"""
        config_times = {}
        all_configs, config_labels_short, _ = self._get_config_labels()

        for run in self.data["results"]:
            mode = run["mode"]
            config = self._format_config(run["configuration"])
            config_label = f"{mode.upper()}\n{config}"
            short_label = config_labels_short[config_label]
            avg_time = run.get("average_request_time", 0)
            config_times[short_label] = avg_time
        
        if not config_times:
            return
        
        times = [config_times.get(c, 0) for c in all_configs]
        colors = self._get_colors(len(all_configs))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(all_configs)), times, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Response Time (seconds)')
        ax.set_title('Average Response Time by Configuration')
        ax.set_xticks(range(len(all_configs)))
        ax.set_xticklabels(all_configs, rotation=0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, time) in enumerate(zip(bars, times)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{time:.4f}s', ha='center', va='bottom', fontsize=9)
        
        self._save_graph(fig, output_path, "06_response_time_by_config.png")
    
    def _graph_response_time_by_question(self, output_path: Path):
        """Generate bar chart of response time by question, nuanced by configuration"""
        # Group times by question and configuration
        question_config_times = {}
        all_configs, config_labels_short, _ = self._get_config_labels()

        for run in self.data["results"]:
            mode = run["mode"]
            config = self._format_config(run["configuration"])
            config_label = f"{mode.upper()}\n{config}"
            short_label = config_labels_short[config_label]

            for q_data in run["questions"]:
                q_id = q_data["question_id"]
                request_time = q_data.get("request_time", 0)
                
                if q_id not in question_config_times:
                    question_config_times[q_id] = {}
                question_config_times[q_id][short_label] = request_time
        
        if not question_config_times:
            return
        
        # Sort questions and limit to first 10 for readability
        q_ids = sorted(question_config_times.keys())[:10]
        
        # Prepare data for grouped bar chart
        x = range(len(q_ids))
        width = 0.8 / len(all_configs)
        colors = self._get_colors(len(all_configs))
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot bars for each configuration
        for i, config in enumerate(all_configs):
            times = [question_config_times[q_id].get(config, 0) for q_id in q_ids]
            offset = width * (i - len(all_configs)/2 + 0.5)
            ax.bar([xi + offset for xi in x], times, width, label=config, color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Question ID')
        ax.set_ylabel('Response Time (seconds)')
        ax.set_title('Response Time by Question and Configuration')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Q{q_id}' for q_id in q_ids])
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        self._save_graph(fig, output_path, "07_response_time_by_question_config.png")
    
    def _graph_performance_heatmap(self, output_path: Path):
        """Generate heatmap of average scores by config and metric"""
        import numpy as np
        
        config_metrics = {}
        all_configs, config_labels_short, _ = self._get_config_labels()

        for run in self.data["results"]:
            mode = run["mode"]
            config = self._format_config(run["configuration"])
            config_label = f"{mode.upper()}\n{config}"
            short_label = config_labels_short[config_label]

            if short_label not in config_metrics:
                config_metrics[short_label] = {}
            
            # Get average scores for each metric
            avg_scores = run.get("average_scores", {})
            for metric, score in avg_scores.items():
                config_metrics[short_label][metric] = score
        
        if not config_metrics:
            return
        
        # Prepare data matrix
        metrics = ["correctness", "relevance", "groundedness", "retrieval_relevance"]
        data = np.array([[config_metrics[config].get(metric, 0) for metric in metrics] for config in all_configs])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=10)
        
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(all_configs)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_yticklabels(all_configs)
        
        # Add text annotations
        for i in range(len(all_configs)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.1f}', ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Performance Heatmap: Scores by Configuration and Metric', fontweight='bold')
        plt.colorbar(im, ax=ax, label='Score')
        self._save_graph(fig, output_path, "03_performance_heatmap.png")

    def _graph_quality_vs_time(self, output_path: Path):
        """Generate scatter plot of quality (overall score) vs response time"""
        config_data = {}
        all_configs, config_labels_short, _ = self._get_config_labels()

        for run in self.data["results"]:
            mode = run["mode"]
            config = self._format_config(run["configuration"])
            config_label = f"{mode.upper()}\n{config}"
            short_label = config_labels_short[config_label]

            avg_scores = run.get("average_scores", {})
            overall_score = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0
            avg_time = run.get("average_request_time", 0)
            
            config_data[short_label] = {"quality": overall_score, "time": avg_time}
        
        if not config_data:
            return
        
        qualities = [config_data[c]["quality"] for c in all_configs]
        times = [config_data[c]["time"] for c in all_configs]
        colors = self._get_colors(len(all_configs))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(times, qualities, s=300, c=range(len(all_configs)), cmap='Set3', alpha=0.7, edgecolors='black', linewidth=2)
        
        # Add labels for each point
        for i, (config, time, quality) in enumerate(zip(all_configs, times, qualities)):
            ax.annotate(config, (time, quality), fontsize=9, fontweight='bold', 
                       xytext=(5, 5), textcoords='offset points', ha='left')
        
        ax.set_xlabel('Average Response Time (seconds)', fontsize=12)
        ax.set_ylabel('Overall Quality Score', fontsize=12)
        ax.set_title('Quality vs Response Time Trade-off', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 10)
        
        # Add ideal zone annotation
        ax.axhspan(8, 10, alpha=0.1, color='green', label='High Quality Zone')
        ax.axvspan(0, min(times) * 1.5 if times else 1, alpha=0.1, color='blue', label='Fast Response Zone')
        ax.legend(loc='lower left')
        
        self._save_graph(fig, output_path, "01_quality_vs_time.png")

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

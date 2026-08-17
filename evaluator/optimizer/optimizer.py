#!/usr/bin/env python3
"""
Hyperparameter optimizer for the RAG/agentic chatbot.

Replaces the exhaustive grid search of evaluator/benchmark with a sample-efficient
Bayesian search (Optuna, TPE sampler) suited to an expensive (minutes-to-hours per
configuration) and stochastic (LLM-graded) objective: the average score of the N
dataset questions across the 4 evaluation metrics (correctness, relevance,
groundedness, retrieval_relevance).

Two techniques address the specifics of this problem on top of plain Bayesian
optimization:
- Pruning: a configuration's running average is checked periodically while it is
  being evaluated, and evaluation is aborted early if it is clearly worse than
  other trials at the same point — the single biggest lever for cutting total
  runtime, since most of the cost is running the N questions through the chatbot
  and grading LLM.
- Final validation: because scores are stochastic, the top candidates found by the
  search are re-evaluated on the full dataset a few more times after optimization,
  to report a mean +/- standard deviation and avoid crowning a noisy outlier.
"""

import sys
import json
import os
import time
import random
import argparse
import statistics
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "optuna is required to run the optimizer. Install it with "
        "'pip install -r requirements.txt' (or 'pip install optuna') in the "
        "project's virtual environment."
    ) from exc

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: this script never opens an interactive window
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# TPESampler(multivariate=True, group=True) below is a deliberate, informed choice
# (it's Optuna's documented pattern for a conditional/branching search space — see
# _suggest_trial_config) rather than an accidental one, so silence just these two
# known ExperimentalWarnings instead of leaving them to clutter every run's output.
# optuna.visualization.matplotlib's plot_* functions are similarly still marked
# experimental despite being stable since Optuna 2.2 — same reasoning for silencing.
warnings.filterwarnings(
    "ignore",
    category=optuna.exceptions.ExperimentalWarning,
    message=r"Argument ``(multivariate|group)``",
)
warnings.filterwarnings(
    "ignore",
    category=optuna.exceptions.ExperimentalWarning,
    message=r".*plot_param_importances is experimental",
)

SCRIPT_DIR = Path(__file__).resolve().parent
OPTIMIZER_CONFIG_FILE = SCRIPT_DIR / "optimizer_config.json"
OPTIMIZER_CONFIG_EXAMPLE_FILE = SCRIPT_DIR / "optimizer_config.example.json"
# Anchored to the script's own directory rather than the current working
# directory, so results/studies always land next to optimizer.py regardless of
# where `python optimizer.py` is run from.
RESULTS_DIR = SCRIPT_DIR / "optimizer_results"
STUDIES_DIR = SCRIPT_DIR / "studies"
GRAPHS_DIR = SCRIPT_DIR / "graphs"

# Accepted (COMPLETE) vs refused (PRUNED/FAIL) — used consistently across every plot.
_STATE_COLOR_ACCEPTED = "#2ca02c"
_STATE_COLOR_REFUSED = "#d62728"

_SUPPORTED_MODES = {"rag", "agentic"}
_SUPPORTED_PARAM_TYPES = {"int", "float", "bool", "categorical"}
# Sub-spaces sampled unconditionally every trial (unlike "rag"/"agentic", which are
# mutually exclusive branches of "mode") — model choice and extraction params apply
# regardless of which mode a trial ends up using.
_FLAT_SUBSPACE_KEYS = ("model", "extraction")

# Import from evaluator base package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluator.base.evaluator import (
    _call_chatbot,
    _validate_config,
    run_evaluation,
    _load_examples,
    EVAL_METRICS,
)
from evaluator.base import extraction_cache
from evaluator.base.extraction_cache import TUNABLE_EXTRACTION_FIELDS


def _validate_sub_space(label: str, sub_space: Dict[str, Any], fixed_params: Dict[str, Any]) -> None:
    """Shared per-parameter validation for one search_space sub-space (a mode's
    rag/agentic block, or the flat 'model'/'extraction' blocks) — every entry
    needs a supported type, 'low'/'high' for int/float, 'choices' for categorical."""
    if "mode" in sub_space:
        raise ValueError(f"search_space['{label}'] must not define a 'mode' parameter — that name is reserved.")

    for name, spec in sub_space.items():
        ptype = spec.get("type")
        if ptype not in _SUPPORTED_PARAM_TYPES:
            raise ValueError(
                f"search_space['{label}']['{name}'].type must be one of "
                f"{sorted(_SUPPORTED_PARAM_TYPES)}, got {ptype!r}."
            )
        if ptype in ("int", "float") and ("low" not in spec or "high" not in spec):
            raise ValueError(
                f"search_space['{label}']['{name}'] of type '{ptype}' needs both 'low' and 'high'."
            )
        if ptype == "categorical" and not spec.get("choices"):
            raise ValueError(
                f"search_space['{label}']['{name}'] of type 'categorical' needs a non-empty 'choices' list."
            )

    overlap = set(fixed_params) & set(sub_space)
    if overlap:
        raise ValueError(
            f"fixed_params['{label}'] and search_space['{label}'] both define {sorted(overlap)} — "
            "a hyperparameter cannot be both fixed and optimized."
        )


def _validate_optimizer_config(config_data: Dict[str, Any]) -> None:
    """Fail fast with a clear message if optimizer_config.json is missing or
    misusing keys the rest of this module indexes directly, instead of a bare
    KeyError/AttributeError deep inside a multi-hour run."""
    if "search_space" not in config_data:
        raise ValueError(
            "Optimizer config is missing required key: search_space. "
            f"See {OPTIMIZER_CONFIG_EXAMPLE_FILE} for the expected structure."
        )

    search_space = config_data["search_space"]
    mode_spec = search_space.get("mode")
    if not mode_spec or not mode_spec.get("choices"):
        raise ValueError(
            "optimizer config's search_space must have a 'mode' entry with a "
            f"non-empty 'choices' list (e.g. {{\"choices\": [\"rag\", \"agentic\"]}}). "
            f"See {OPTIMIZER_CONFIG_EXAMPLE_FILE} for the expected structure."
        )

    unknown_modes = set(mode_spec["choices"]) - _SUPPORTED_MODES
    if unknown_modes:
        raise ValueError(
            f"Unsupported mode(s) in search_space['mode']['choices']: {sorted(unknown_modes)}. "
            f"Supported modes are: {sorted(_SUPPORTED_MODES)}."
        )

    fixed_params_by_key = config_data.get("fixed_params", {})

    for mode in mode_spec["choices"]:
        sub_space = search_space.get(mode)
        if not sub_space:
            raise ValueError(
                f"search_space['mode']['choices'] includes '{mode}' but search_space has no "
                f"(non-empty) '{mode}' entry with that mode's hyperparameters."
            )
        _validate_sub_space(mode, sub_space, fixed_params_by_key.get(mode, {}))

    # 'model' and 'extraction' are optional, flat (mode-independent) sub-spaces:
    # sampled on every trial regardless of which mode ('rag'/'agentic') it uses.
    if "model" in search_space:
        _validate_sub_space("model", search_space["model"], fixed_params_by_key.get("model", {}))

    if "extraction" in search_space:
        extraction_space = search_space["extraction"]
        _validate_sub_space("extraction", extraction_space, fixed_params_by_key.get("extraction", {}))
        unknown_fields = set(extraction_space) - set(TUNABLE_EXTRACTION_FIELDS)
        if unknown_fields:
            raise ValueError(
                f"search_space['extraction'] has unsupported field(s): {sorted(unknown_fields)}. "
                f"Supported: {list(TUNABLE_EXTRACTION_FIELDS)}."
            )
        unknown_fixed = set(fixed_params_by_key.get("extraction", {})) - set(TUNABLE_EXTRACTION_FIELDS)
        if unknown_fixed:
            raise ValueError(
                f"fixed_params['extraction'] has unsupported field(s): {sorted(unknown_fixed)}. "
                f"Supported: {list(TUNABLE_EXTRACTION_FIELDS)}."
            )


def load_optimizer_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load optimizer parameters from JSON configuration."""
    path = Path(config_path) if config_path else OPTIMIZER_CONFIG_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Optimizer config file not found: {path}. "
            f"Create it from {OPTIMIZER_CONFIG_EXAMPLE_FILE} and rerun."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Optimizer config file is invalid JSON: {path}: {exc}") from exc

    _validate_optimizer_config(config_data)
    return config_data


def _suggest_value(trial: "optuna.trial.Trial", key: str, spec: Dict[str, Any]) -> Any:
    ptype = spec["type"]
    if ptype == "int":
        return trial.suggest_int(key, spec["low"], spec["high"], step=spec.get("step", 1))
    if ptype == "float":
        return trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))
    if ptype == "bool":
        return trial.suggest_categorical(key, [True, False])
    if ptype == "categorical":
        return trial.suggest_categorical(key, spec["choices"])
    raise ValueError(f"Unsupported param type {ptype!r} for {key!r}.")


def _suggest_sub_space(
    trial: "optuna.trial.Trial",
    prefix: str,
    sub_space: Dict[str, Any],
    fixed: Dict[str, Any],
) -> Dict[str, Any]:
    """Sample every parameter in one search_space sub-space, each prefixed with
    `prefix` so it's tracked as a distinct Optuna parameter from any same-named
    parameter in another sub-space (e.g. both 'rag' and 'agentic' have a
    'temperature', with potentially different ranges)."""
    params = dict(fixed)
    for name, spec in sub_space.items():
        params[name] = _suggest_value(trial, f"{prefix}__{name}", spec)
    return params


def _suggest_trial_config(
    trial: "optuna.trial.Trial",
    search_space: Dict[str, Any],
    fixed_params_by_key: Dict[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    """Sample a full trial configuration: (mode, chatbot hyperparameters,
    extraction overrides).

    'mode' is itself an optimized categorical choice between 'rag' and 'agentic';
    only the sub search space matching the sampled mode is then sampled from, so a
    trial never suggests parameters that don't apply to it. This is Optuna's
    standard pattern for a conditional/branching search space (a branch variable
    followed by branch-specific parameters) — TPESampler(multivariate=True,
    group=True) is built to handle exactly this.

    'model' and 'extraction' (both optional in search_space) are sampled
    unconditionally on every trial regardless of mode — an LLM/reranker choice or
    an extraction variant applies whether the trial is rag or agentic. 'model'
    params (model/base_url/api_key) are merged into the returned hyperparameter
    dict alongside the mode-specific ones (both end up as chatbot.py CLI flags via
    _call_chatbot); 'extraction' params are returned separately since they drive
    an extraction_cache.get_or_build() lookup instead of a CLI flag.
    """
    mode = trial.suggest_categorical("mode", search_space["mode"]["choices"])
    params = _suggest_sub_space(trial, mode, search_space[mode], fixed_params_by_key.get(mode, {}))

    if "model" in search_space:
        params.update(_suggest_sub_space(trial, "model", search_space["model"], fixed_params_by_key.get("model", {})))

    extraction_overrides = None
    if "extraction" in search_space:
        extraction_overrides = _suggest_sub_space(
            trial, "extraction", search_space["extraction"], fixed_params_by_key.get("extraction", {})
        )

    return mode, params, extraction_overrides


def _params_from_trial(
    t: "optuna.trial.FrozenTrial",
    config_data: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    """Reconstruct (mode, hyperparameters, extraction overrides) for an
    already-sampled trial, e.g. to re-run its exact configuration during final
    validation."""
    search_space = config_data["search_space"]
    fixed_params_by_key = config_data.get("fixed_params", {})
    mode = t.params["mode"]

    def _collect(prefix: str, fixed: Dict[str, Any]) -> Dict[str, Any]:
        collected = dict(fixed)
        marker = f"{prefix}__"
        for k, v in t.params.items():
            if k.startswith(marker):
                collected[k[len(marker):]] = v
        return collected

    params = _collect(mode, fixed_params_by_key.get(mode, {}))
    if "model" in search_space:
        params.update(_collect("model", fixed_params_by_key.get("model", {})))

    extraction_overrides = None
    if "extraction" in search_space:
        extraction_overrides = _collect("extraction", fixed_params_by_key.get("extraction", {}))

    return mode, params, extraction_overrides


def _run_question(
    question_id: int,
    example: Dict[str, Any],
    mode: str,
    config: Dict[str, Any],
    chatbot_timeout: Optional[int],
    chatbot_config_path: Optional[str] = None,
) -> Tuple[int, Dict[str, float], float, float]:
    """Run one dataset question through the chatbot and the 4 grading metrics.

    Returns (question_id, per_metric_scores, question_average_score, request_time).
    """
    question = example["inputs"]["question"]
    reference_answer = example["outputs"]["answer"]

    answer_dict = _call_chatbot(question, mode, chatbot_timeout, config=config,
                                 chatbot_config_path=chatbot_config_path)
    # Metrics run sequentially here on purpose: this function runs inside a worker
    # thread of the per-question pool below. Submitting more work to that same
    # bounded pool from within one of its own workers can deadlock once all
    # workers are occupied waiting on sub-tasks that have no free thread to run on.
    evaluations = run_evaluation(question, answer_dict, reference_answer)
    request_time = answer_dict.get("request_time", 0.0)

    metric_scores = {m: evaluations[m].get("score", 0.0) for m in EVAL_METRICS}
    question_average = sum(metric_scores.values()) / len(metric_scores)
    return question_id, metric_scores, question_average, request_time


def evaluate_config_on_dataset(
    config: Dict[str, Any],
    dataset: List[Dict[str, Any]],
    mode: str,
    chatbot_timeout: Optional[int],
    workers: int,
    trial: Optional["optuna.trial.Trial"] = None,
    min_questions_before_report: int = 0,
    interval_questions: int = 1,
    chatbot_config_path: Optional[str] = None,
) -> Tuple[List[float], Dict[str, float], float]:
    """Run a hyperparameter configuration over the whole dataset.

    If `trial` is given, the running average score is periodically reported to
    Optuna (via trial.report) once at least `min_questions_before_report`
    questions have completed, checked every `interval_questions` questions.
    optuna.TrialPruned is raised if Optuna decides the trial should be cut short.

    Returns (per_question_average_scores, per_metric_averages, avg_request_time).
    """
    n = len(dataset)
    question_scores: List[Optional[float]] = [None] * n
    request_times: List[float] = [0.0] * n
    metric_totals = {m: 0.0 for m in EVAL_METRICS}
    completed = 0

    def _record(question_id: int, metric_scores: Dict[str, float], q_avg: float, req_time: float) -> None:
        question_scores[question_id - 1] = q_avg
        request_times[question_id - 1] = req_time
        for m, v in metric_scores.items():
            metric_totals[m] += v

    def _maybe_report(step: int) -> None:
        if trial is None or step < min_questions_before_report:
            return
        if step % interval_questions != 0 and step != n:
            return
        running_mean = sum(s for s in question_scores if s is not None) / step
        trial.report(running_mean, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Processed in fixed-size chunks (rather than one giant future map) so
            # pruning is checked between chunks instead of only after every single
            # question has finished across the whole dataset.
            for chunk_start in range(0, n, workers):
                chunk = list(enumerate(dataset[chunk_start:chunk_start + workers], start=chunk_start + 1))
                futures = {
                    executor.submit(_run_question, q_id, example, mode, config, chatbot_timeout,
                                     chatbot_config_path): q_id
                    for q_id, example in chunk
                }
                for future in as_completed(futures):
                    q_id, metric_scores, q_avg, req_time = future.result()
                    _record(q_id, metric_scores, q_avg, req_time)
                completed += len(chunk)
                _maybe_report(completed)
    else:
        for q_id, example in enumerate(dataset, 1):
            _, metric_scores, q_avg, req_time = _run_question(q_id, example, mode, config, chatbot_timeout,
                                                                chatbot_config_path)
            _record(q_id, metric_scores, q_avg, req_time)
            completed += 1
            _maybe_report(completed)

    per_metric_avg = {m: metric_totals[m] / n for m in EVAL_METRICS}
    avg_request_time = sum(request_times) / n if n else 0.0
    return question_scores, per_metric_avg, avg_request_time


def objective(
    trial: "optuna.trial.Trial",
    config_data: Dict[str, Any],
    dataset: List[Dict[str, Any]],
) -> float:
    mode, params, extraction_overrides = _suggest_trial_config(
        trial, config_data["search_space"], config_data.get("fixed_params", {})
    )

    pruning_cfg = config_data.get("pruning", {})
    pruning_enabled = pruning_cfg.get("enabled", True)
    workers = config_data.get("workers", 1)
    chatbot_timeout = config_data.get("chatbot_timeout_seconds")

    print(f"\n[Trial {trial.number}] starting: mode={mode} params={params}"
          + (f" extraction={extraction_overrides}" if extraction_overrides is not None else ""))
    start = time.perf_counter()
    try:
        chatbot_config_path = None
        if extraction_overrides is not None:
            # Cost note: unlike pruning below, there is no early-exit here — a trial
            # that's the first to sample a never-before-built extraction combo pays
            # the full rebuild cost (minutes) regardless of how the trial eventually
            # scores. Subsequent trials sampling the SAME combo reuse the cache.
            build_start = time.perf_counter()
            chatbot_config_path = extraction_cache.get_or_build(
                extraction_overrides, timeout_seconds=config_data.get("extraction_timeout_seconds")
            )
            trial.set_user_attr("extraction_build_seconds", time.perf_counter() - build_start)

        _, per_metric_avg, avg_request_time = evaluate_config_on_dataset(
            params,
            dataset,
            mode,
            chatbot_timeout,
            workers,
            trial=trial if pruning_enabled else None,
            min_questions_before_report=pruning_cfg.get("min_questions_before_pruning", 20),
            interval_questions=pruning_cfg.get("interval_questions", 5),
            chatbot_config_path=chatbot_config_path,
        )
    finally:
        # Recorded even on optuna.TrialPruned (raised from inside the call above)
        # so pruned trials also report how long they ran for, not just completed ones.
        trial.set_user_attr("elapsed_seconds", time.perf_counter() - start)
    overall = sum(per_metric_avg.values()) / len(per_metric_avg)
    trial.set_user_attr("mode", mode)
    if extraction_overrides is not None:
        trial.set_user_attr("extraction_overrides", extraction_overrides)
    trial.set_user_attr("per_metric_avg", per_metric_avg)
    trial.set_user_attr("avg_request_time", avg_request_time)
    trial.set_user_attr("n_questions", len(dataset))
    return overall


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    seconds = round(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _make_trial_callback(config_data: Dict[str, Any]):
    """Build an Optuna callback that prints each finished trial with its mode
    resolved and its params de-prefixed (trial.params holds raw 'mode__name'
    keys plus 'mode' itself — see _suggest_trial_config)."""

    def _callback(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        elapsed = _format_duration(trial.user_attrs.get("elapsed_seconds"))
        if trial.state == optuna.trial.TrialState.COMPLETE:
            mode, params, _ = _params_from_trial(trial, config_data)
            is_best = study.best_trial.number == trial.number
            marker = " *NEW BEST*" if is_best else ""
            print(f"[Trial {trial.number}] COMPLETE in {elapsed} value={trial.value:.3f}{marker} mode={mode} params={params}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            mode, params, _ = _params_from_trial(trial, config_data)
            last_step = max(trial.intermediate_values) if trial.intermediate_values else None
            partial = trial.intermediate_values.get(last_step, float("nan")) if last_step is not None else float("nan")
            print(
                f"[Trial {trial.number}] PRUNED in {elapsed} at question {last_step} "
                f"(running average was {partial:.3f}) mode={mode} params={params}"
            )
        elif trial.state == optuna.trial.TrialState.FAIL:
            print(f"[Trial {trial.number}] FAILED in {elapsed} params={trial.params}")

    return _callback


def run_final_validation(
    top_trials: List["optuna.trial.FrozenTrial"],
    dataset: List[Dict[str, Any]],
    config_data: Dict[str, Any],
    repeats: int,
) -> List[Dict[str, Any]]:
    """Re-evaluate the top candidates on the full dataset several more times
    (no pruning) to report mean +/- stdev and guard against a noisy outlier
    winning purely by luck during the search."""
    workers = config_data.get("workers", 1)
    chatbot_timeout = config_data.get("chatbot_timeout_seconds")

    validated = []
    for rank, t in enumerate(top_trials, 1):
        mode, params, extraction_overrides = _params_from_trial(t, config_data)
        print(f"\nValidating candidate {rank}/{len(top_trials)} (trial {t.number}, search value={t.value:.3f}): "
              f"mode={mode} params={params}"
              + (f" extraction={extraction_overrides}" if extraction_overrides is not None else ""))

        chatbot_config_path = None
        if extraction_overrides is not None:
            # Cache hit in the common case (this exact combo was already built during
            # the search that produced this trial) — only a real rebuild if a study's
            # storage was reused without its extraction_cache/ (e.g. copied elsewhere).
            chatbot_config_path = extraction_cache.get_or_build(
                extraction_overrides, timeout_seconds=config_data.get("extraction_timeout_seconds")
            )

        run_values = []
        per_metric_runs = []
        for r in range(repeats):
            print(f"  Repeat {r + 1}/{repeats}...")
            repeat_start = time.perf_counter()
            _, per_metric_avg, _ = evaluate_config_on_dataset(
                params, dataset, mode, chatbot_timeout, workers, chatbot_config_path=chatbot_config_path
            )
            overall = sum(per_metric_avg.values()) / len(per_metric_avg)
            run_values.append(overall)
            per_metric_runs.append(per_metric_avg)
            print(f"    -> {overall:.3f} (in {_format_duration(time.perf_counter() - repeat_start)})")

        mean_v = statistics.mean(run_values)
        std_v = statistics.stdev(run_values) if len(run_values) > 1 else 0.0
        validated.append({
            "trial_number": t.number,
            "mode": mode,
            "params": params,
            "extraction_overrides": extraction_overrides,
            "search_value": t.value,
            "validation_runs": run_values,
            "validation_mean": mean_v,
            "validation_std": std_v,
            "per_metric_runs": per_metric_runs,
        })

    validated.sort(key=lambda r: r["validation_mean"], reverse=True)
    return validated


def _serialize_trial(t: "optuna.trial.FrozenTrial") -> Dict[str, Any]:
    return {
        "number": t.number,
        "state": t.state.name,
        "value": t.value,
        "params": t.params,
        "user_attrs": t.user_attrs,
        "intermediate_values": {str(k): v for k, v in t.intermediate_values.items()},
    }


def _default_storage(study_name: str) -> str:
    STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{STUDIES_DIR / study_name}.db"


def _trial_color(state: "optuna.trial.TrialState") -> str:
    """Green for an accepted (COMPLETE) trial, red for anything refused (PRUNED/FAIL)."""
    return _STATE_COLOR_ACCEPTED if state == optuna.trial.TrialState.COMPLETE else _STATE_COLOR_REFUSED


def _trial_display_value(t: "optuna.trial.FrozenTrial") -> Optional[float]:
    """The value to plot for a trial: its final value if it completed, otherwise
    its last reported running average before being pruned (None for FAIL, which
    has neither)."""
    if t.value is not None:
        return t.value
    if t.intermediate_values:
        return t.intermediate_values[max(t.intermediate_values)]
    return None


def _plot_optimization_history(study: "optuna.Study", ax) -> None:
    """Objective value per trial (green=accepted, red=refused) plus the running
    best-so-far curve — the standard first plot for any hyperparameter search,
    and directly answers 'is this still improving or has it plateaued'."""
    trials = sorted(study.trials, key=lambda t: t.number)
    best_so_far = []
    best = None
    any_point = False
    for t in trials:
        y = _trial_display_value(t)
        if y is None:
            continue
        any_point = True
        ax.scatter(t.number, y, color=_trial_color(t.state), s=28, zorder=3)
        if t.state == optuna.trial.TrialState.COMPLETE:
            best = y if best is None else max(best, y)
        if best is not None:
            best_so_far.append((t.number, best))

    if best_so_far:
        xs, ys = zip(*best_so_far)
        ax.plot(xs, ys, color="#1f77b4", linewidth=2, zorder=2, label="best so far")

    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective value")
    ax.set_title("Optimization history")
    if not any_point:
        ax.text(0.5, 0.5, "No trial produced a value yet", ha="center", va="center", transform=ax.transAxes)
        return

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_STATE_COLOR_ACCEPTED,
               markersize=8, label="Accepted (complete)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_STATE_COLOR_REFUSED,
               markersize=8, label="Refused (pruned/failed)"),
    ]
    if best_so_far:
        handles.append(Line2D([0], [0], color="#1f77b4", linewidth=2, label="Best so far"))
    ax.legend(handles=handles, loc="best", fontsize=8)


def _plot_param_evolution(study: "optuna.Study", path: Path, exclude_mode: bool = False) -> None:
    """One small subplot per Optuna-level parameter (trial number vs sampled
    value, green=accepted/red=refused), so a glance shows whether the sampler is
    converging on a region for each knob or still spread out. Uses the raw
    Optuna param names (e.g. 'rag__k_standard', 'extraction__embedding.model')
    rather than de-prefixed ones — each is already an unambiguous decision
    Optuna actually made, which is what this plot is about."""
    trials = [t for t in study.trials if t.params]
    if not trials:
        return

    param_names = sorted({k for t in trials for k in t.params if not (exclude_mode and k == "mode")})
    if not param_names:
        return

    ncols = min(3, len(param_names))
    nrows = -(-len(param_names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows), squeeze=False)
    flat_axes = [ax for row in axes for ax in row]

    for ax, name in zip(flat_axes, param_names):
        xs, raw_vals, colors = [], [], []
        for t in trials:
            if name not in t.params:
                continue
            xs.append(t.number)
            raw_vals.append(t.params[name])
            colors.append(_trial_color(t.state))

        if all(isinstance(v, bool) for v in raw_vals):
            ys = [int(v) for v in raw_vals]
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["False", "True"])
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw_vals):
            ys = raw_vals
        else:
            categories = sorted({v for v in raw_vals}, key=str)
            codes = {c: i for i, c in enumerate(categories)}
            ys = [codes[v] for v in raw_vals]
            ax.set_yticks(list(codes.values()))
            ax.set_yticklabels(list(codes.keys()))

        ax.scatter(xs, ys, c=colors, s=22, zorder=3)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)

    for ax in flat_axes[len(param_names):]:
        ax.axis("off")

    fig.suptitle("Parameter evolution across trials (green = accepted, red = refused)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_plots(study: "optuna.Study", output_dir: Path) -> Optional[Path]:
    """Save the small set of plots that matter for a hyperparameter search:
    optimization history, parameter importances, and per-parameter evolution.
    No-op (with a message) if matplotlib isn't installed."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available -- skipping plots. Install with: pip install matplotlib")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_optimization_history(study, ax)
    fig.tight_layout()
    fig.savefig(output_dir / "01_optimization_history.png", dpi=150)
    plt.close(fig)

    # "mode" is a real Optuna parameter even when the config pins a single choice
    # (search_space.mode.choices == [X]) -- not worth a bar/subplot of its own then,
    # since it never varies.
    single_mode = len({t.params.get("mode") for t in study.trials if "mode" in t.params}) <= 1

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) >= 2:
        try:
            importance_params = None
            if single_mode:
                importance_params = sorted({k for t in study.trials for k in t.params if k != "mode"}) or None
            imp_ax = optuna.visualization.matplotlib.plot_param_importances(study, params=importance_params)
            imp_ax.figure.tight_layout()
            imp_ax.figure.savefig(output_dir / "02_param_importances.png", dpi=150)
            plt.close(imp_ax.figure)
        except (ValueError, RuntimeError) as e:
            # e.g. every completed trial shares the exact same params (nothing to
            # attribute importance to) -- not worth failing the whole run over.
            print(f"Skipped parameter importance plot: {e}")
    else:
        print("Skipped parameter importance plot: needs at least 2 completed trials.")

    _plot_param_evolution(study, output_dir / "03_param_evolution.png", exclude_mode=single_mode)

    print(f"Plots saved to: {output_dir}/")
    return output_dir


def run_optimizer(config_data: Dict[str, Any]) -> Dict[str, Any]:
    _validate_config()

    # -1 means "use all CPUs" (same convention as evaluator/benchmark's --workers),
    # resolved once here so every reader of config_data["workers"] below (objective,
    # run_final_validation) sees the same concrete integer.
    workers_cfg = config_data.get("workers", 1)
    if workers_cfg == -1:
        workers_cfg = os.cpu_count() or 1
    elif workers_cfg < -1:
        raise ValueError(f"'workers' must be -1 or a positive integer, got {workers_cfg}.")
    config_data["workers"] = workers_cfg

    n_jobs_cfg = config_data.get("n_jobs", 1)
    effective_n_jobs = (os.cpu_count() or 1) if n_jobs_cfg == -1 else n_jobs_cfg
    effective_concurrency = workers_cfg * effective_n_jobs
    if effective_concurrency > 4:
        # Unlike a lightweight thread, each "worker" here spawns a full chatbot.py
        # subprocess that reloads the embedding model, the Chroma vectorstore and
        # (if reranker_enabled) the BAAI/bge-reranker-v2-m3 cross-encoder — several
        # GB of RAM by itself — from scratch. High workers x n_jobs has previously
        # driven this process into swap and gotten it OOM-killed mid-trial (visible
        # afterwards as a trial stuck in RUNNING state in the study database), which
        # looks like the optimizer "hanging" even though nothing in this script
        # deadlocks on its own. Concurrent processes also all open the SAME Chroma
        # persist_directory, which is a further contention point under high concurrency.
        print(
            f"WARNING: workers={workers_cfg} x n_jobs={n_jobs_cfg} means up to "
            f"{effective_concurrency} chatbot.py subprocesses running at once, each "
            "reloading its own models (the RAG reranker alone is several GB). This "
            "can exhaust RAM and make the run appear to hang rather than actually "
            "deadlocking. Consider a modest workers value (e.g. 2-4) unless you've "
            "confirmed the machine has enough RAM for that many concurrent model loads."
        )

    dataset = _load_examples()
    questions_limit = config_data.get("questions_limit")
    if questions_limit is not None:
        dataset = dataset[:questions_limit]

    # dataset.json is grouped by module/topic rather than shuffled, so a pruned
    # trial that only sees the first N questions would see a biased, unrepresentative
    # sample. Shuffle once with a fixed seed and reuse that same order for every
    # trial, so intermediate values stay comparable across trials at a given step.
    shuffle_seed = config_data.get("dataset_shuffle_seed")
    if shuffle_seed is not None:
        dataset = list(dataset)
        random.Random(shuffle_seed).shuffle(dataset)

    study_name = config_data.get("study_name", "chatbot_hyperparameter_optimization")
    storage = config_data.get("storage") or _default_storage(study_name)

    pruning_cfg = config_data.get("pruning", {})
    if pruning_cfg.get("enabled", True):
        pruner = optuna.pruners.PercentilePruner(
            percentile=pruning_cfg.get("percentile", 25.0),
            n_startup_trials=pruning_cfg.get("n_warmup_trials", 5),
            n_warmup_steps=pruning_cfg.get("min_questions_before_pruning", 20),
            interval_steps=pruning_cfg.get("interval_questions", 5),
        )
    else:
        pruner = optuna.pruners.NopPruner()

    sampler = optuna.samplers.TPESampler(
        seed=config_data.get("sampler_seed"),
        multivariate=True,
        group=True,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    n_trials = config_data.get("n_trials", 30)
    already_run = len([t for t in study.trials if t.state != optuna.trial.TrialState.RUNNING])
    stuck_running = len(study.trials) - already_run
    next_trial_number = len(study.trials)

    print(f"\n{'=' * 80}")
    print("OPTIMIZER CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"Workers per trial: {workers_cfg}")
    search_space = config_data["search_space"]
    fixed_params_by_key = config_data.get("fixed_params", {})
    print(f"Modes considered: {search_space['mode']['choices']} (mode itself is optimized)")
    print(f"Questions per trial: {len(dataset)}")
    for m in search_space["mode"]["choices"]:
        print(f"  [{m}] search space: {json.dumps(search_space[m])}")
        if fixed_params_by_key.get(m):
            print(f"  [{m}] fixed params: {fixed_params_by_key[m]}")
    if "model" in search_space:
        print(f"  [model, every trial] search space: {json.dumps(search_space['model'])}")
        if fixed_params_by_key.get("model"):
            print(f"  [model, every trial] fixed params: {fixed_params_by_key['model']}")
    if "extraction" in search_space:
        print(f"  [extraction, every trial] search space: {json.dumps(search_space['extraction'])}")
        if fixed_params_by_key.get("extraction"):
            print(f"  [extraction, every trial] fixed params: {fixed_params_by_key['extraction']}")
        n_warmup_trials = pruning_cfg.get("n_warmup_trials", 5) if pruning_cfg.get("enabled", True) else 0
        print(
            f"WARNING: extraction hyperparameters are being searched. Unlike other params, a new "
            f"combination costs a full corpus re-extraction (minutes) with no early-exit, cached "
            f"afterwards under {extraction_cache.CACHE_ROOT}/ (~hundreds of MB each). TPE's random "
            f"startup phase (~{n_warmup_trials or 'several'} trials) can each land on a distinct "
            "combination, so keep this sub-space small (2-3 values per dimension)."
        )
    print(f"Sampler: TPE (seed={config_data.get('sampler_seed')})")
    print(f"Pruning: {'enabled' if pruning_cfg.get('enabled', True) else 'disabled'}")
    print(f"Study: {study_name} ({storage})")
    if already_run or stuck_running:
        # Trial numbers are cumulative for the whole study (persisted in `storage`),
        # not per invocation — so on a resumed study the first trial printed below
        # is expected to start at #{next_trial_number}, not #0.
        note = f"Resuming existing study: {already_run} trial(s) already recorded"
        if stuck_running:
            note += f" ({stuck_running} left stuck in RUNNING state by an interrupted previous session — Optuna will just skip over them and start fresh trials)"
        print(note)
        print(f"New trials in this session will be numbered starting at #{next_trial_number} (not #0) — trial numbers are cumulative for this study.")
    print(f"Trials to run this session: {n_trials}")
    if config_data.get("n_jobs", 1) != 1:
        print(
            f"n_jobs={config_data.get('n_jobs')}: trials run concurrently, so their 'starting'/'COMPLETE'/'PRUNED' "
            "lines below may print out of numeric order — that's expected, not a bug."
        )
    print(f"{'=' * 80}\n")

    study.optimize(
        lambda trial: objective(trial, config_data, dataset),
        n_trials=n_trials,
        timeout=config_data.get("timeout_seconds"),
        n_jobs=config_data.get("n_jobs", 1),
        callbacks=[_make_trial_callback(config_data)],
        gc_after_trial=True,
    )

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: t.value, reverse=True)

    try:
        best_trial = study.best_trial
    except ValueError:
        # No trial completed (e.g. everything failed or was pruned before finishing) —
        # study.best_trial raises rather than returning None in that case.
        best_trial = None

    print(f"\n{'=' * 80}")
    print("SEARCH COMPLETE")
    print(f"{'=' * 80}")
    if best_trial is not None:
        best_mode, best_params, best_extraction = _params_from_trial(best_trial, config_data)
        print(f"Best trial: #{best_trial.number} value={best_trial.value:.3f}")
        print(f"Mode: {best_mode}")
        print(f"Params: {best_params}")
        if best_extraction is not None:
            print(f"Extraction: {best_extraction}")
    else:
        print("No trial completed successfully.")

    validation_results = None
    final_validation_cfg = config_data.get("final_validation", {})
    if final_validation_cfg.get("enabled", True) and completed_trials:
        top_k = final_validation_cfg.get("top_k", 3)
        repeats = final_validation_cfg.get("repeats", 3)
        top_trials = completed_trials[:top_k]
        print(f"\n{'=' * 80}")
        print(f"FINAL VALIDATION (top {len(top_trials)} candidates, {repeats} repeats each)")
        print(f"{'=' * 80}")
        validation_results = run_final_validation(top_trials, dataset, config_data, repeats)

        print(f"\n{'=' * 80}")
        print("VALIDATED RANKING")
        print(f"{'=' * 80}")
        for rank, r in enumerate(validation_results, 1):
            extraction_suffix = f" extraction={r['extraction_overrides']}" if r.get("extraction_overrides") is not None else ""
            print(
                f"{rank}. trial #{r['trial_number']} [{r['mode']}]: "
                f"{r['validation_mean']:.3f} +/- {r['validation_std']:.3f} "
                f"(search value was {r['search_value']:.3f}) params={r['params']}{extraction_suffix}"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "metadata": {
            "timestamp": timestamp,
            "optimizer_config": config_data,
            "study_name": study_name,
            "storage": storage,
            "n_questions": len(dataset),
        },
        "best_trial": _serialize_trial(best_trial) if best_trial is not None else None,
        "trials": [_serialize_trial(t) for t in study.trials],
        "final_validation": validation_results,
    }
    results_file = RESULTS_DIR / f"optimizer_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {results_file}")

    if config_data.get("plots", {}).get("enabled", True):
        generate_plots(study, GRAPHS_DIR / study_name)

    print(f"\n{'=' * 80}")
    print("RECOMMENDED CONFIGURATION")
    print(f"{'=' * 80}")
    if validation_results:
        best = validation_results[0]
        recommended = {"mode": best["mode"], "params": best["params"]}
        if best.get("extraction_overrides") is not None:
            recommended["extraction"] = best["extraction_overrides"]
        print(json.dumps(recommended, indent=2, ensure_ascii=False))
    elif best_trial is not None:
        recommended = {"mode": best_mode, "params": best_params}
        if best_extraction is not None:
            recommended["extraction"] = best_extraction
        print(json.dumps(recommended, indent=2, ensure_ascii=False))
    else:
        print("No trial completed — nothing to recommend.")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimizer for the RAG/agentic chatbot"
    )
    parser.add_argument("--config", type=str, default=None,
                         help="Path to optimizer config JSON (default: optimizer_config.json)")
    parser.add_argument("--n-trials", type=int, default=None,
                         help="Number of trials to run this session (overrides config)")
    parser.add_argument("--timeout", type=int, default=None,
                         help="Wall-clock budget in seconds for the whole study (overrides config)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Limit number of dataset questions per trial (overrides config)")
    parser.add_argument("--workers", type=int, default=None,
                         help="Parallel threads for evaluating questions within a single trial, "
                              "-1 for all CPUs (overrides config)")
    parser.add_argument("--chatbot-timeout", type=int, default=None,
                         help="Timeout in seconds for each chatbot request (overrides config)")
    parser.add_argument("--extraction-timeout", type=int, default=None,
                         help="Timeout in seconds for building an extraction variant, "
                              "default no limit (overrides config)")
    parser.add_argument("--study-name", type=str, default=None,
                         help="Optuna study name (overrides config)")
    parser.add_argument("--storage", type=str, default=None,
                         help="Optuna storage URL, e.g. sqlite:///studies/foo.db (overrides config)")
    parser.add_argument("--no-pruning", action="store_true",
                         help="Disable pruning regardless of config")
    parser.add_argument("--no-validation", action="store_true",
                         help="Skip the final validation pass regardless of config")
    parser.add_argument("--no-plots", action="store_true",
                         help="Skip generating plots regardless of config")

    args = parser.parse_args()

    config_data = load_optimizer_config(Path(args.config) if args.config else None)

    if args.n_trials is not None:
        config_data["n_trials"] = args.n_trials
    if args.timeout is not None:
        config_data["timeout_seconds"] = args.timeout
    if args.limit is not None:
        config_data["questions_limit"] = args.limit
    if args.workers is not None:
        config_data["workers"] = args.workers
    if args.chatbot_timeout is not None:
        config_data["chatbot_timeout_seconds"] = args.chatbot_timeout
    if args.extraction_timeout is not None:
        config_data["extraction_timeout_seconds"] = args.extraction_timeout
    if args.study_name is not None:
        config_data["study_name"] = args.study_name
    if args.storage is not None:
        config_data["storage"] = args.storage
    if args.no_pruning:
        config_data.setdefault("pruning", {})["enabled"] = False
    if args.no_validation:
        config_data.setdefault("final_validation", {})["enabled"] = False
    if args.no_plots:
        config_data.setdefault("plots", {})["enabled"] = False

    run_optimizer(config_data)


if __name__ == "__main__":
    main()

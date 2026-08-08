# Hyperparameter Optimizer

Finds the chatbot mode and hyperparameters that maximize the average score over the question dataset (see `evaluator/base/README.md`), instead of exhaustively testing every combination like `evaluator/benchmark`.

## Why not grid search

Each configuration takes minutes to hours to evaluate (N chatbot calls + 4N LLM-graded metrics), and the hyperparameters mix booleans, integers and floats across two modes. A full cross-product over even a modest range per parameter (as `evaluator/benchmark` does) quickly becomes too expensive to run to completion.

This optimizer uses **Bayesian optimization** (via [Optuna](https://optuna.org/), TPE sampler) instead: each trial's result informs where the next trial samples, so it converges on good regions of the search space in far fewer evaluations than grid or random search — the standard approach for optimizing an expensive, black-box, mixed-type objective. Two additions specifically target the two properties that make this problem harder than a typical hyperparameter search:

- **The objective is expensive** (each trial = N chatbot calls + 4N grading calls). *Pruning* checks a trial's running average score periodically while it's still evaluating, and aborts it early if it's clearly worse than other trials at the same point — so a bad configuration is abandoned after ~20 questions instead of burning through all questions. This is the single biggest lever for cutting total runtime.

- **Each score is stochastic** (LLM-graded). Averaging over N questions + 4N metrics already smooths out a lot of that noise within one trial, but the *best* trial found by the search could still be a lucky outlier. A *final validation* pass re-evaluates the top few candidates a few more times on the full dataset (no pruning) and reports mean ± standard deviation, so the recommended configuration is a genuinely robust winner, not noise.

## Setup

```bash
cp optimizer_config.example.json optimizer_config.json
```

Edit `optimizer_config.json`:

- `search_space.mode.choices`: which chatbot mode(s) to consider — `["rag", "agentic"]` to let the optimizer pick whichever mode scores higher, or a single-element list (e.g. `["rag"]`) to pin one mode.

- `search_space.rag` / `search_space.agentic`: the hyperparameters to optimize for each mode and their range (`int`/`float` need `low`/`high`; `bool` needs nothing else; `categorical` needs a `choices` list, e.g. for model names). Only the sub-space matching a trial's sampled mode is ever used by that trial.
  - `rag`: `k_standard`, `k_deep_dive`, `top_n`, `temperature`, `max_tokens`, `reranker_enabled`, `reranker_model`, `deep_dive`, `hyde_enabled`, `bm25_enabled`, `title_boost_enabled`, `k_retrieve`, `deep_dive_batch_size`, `expansion_char_budget`. `k_standard`/`k_deep_dive` are the RAG pool size, used depending on that trial's own `deep_dive` value (`chatbot.py --k` also exists as a flat override of whichever applies, but isn't needed here since only one of the two ever takes effect per trial anyway). `k_retrieve` (per-channel retrieval depth before RRF fusion) is ignored in deep dive mode; `deep_dive_batch_size` only applies when `deep_dive` is true; `reranker_model` only has an effect when `reranker_enabled` is true.
  - `agentic`: `temperature`, `max_tokens`, `max_steps` (CodeAgent step budget), `max_chars_per_page`.
  - This mirrors exactly what `chatbot.py`'s CLI flags support — see `chatbot.py --help` for the authoritative list if `chatbot/core/config.py` changes again.

- `search_space.model` (optional): LLM choice, sampled on **every** trial regardless of mode — `model`, `base_url`, `api_key`. See [Model choice](#model-choice) below before searching over `base_url`/`api_key`.

- `search_space.extraction` (optional): corpus/embedding pipeline choice, also sampled on every trial regardless of mode. See [Extraction parameters](#extraction-parameters) below — this one is expensive, read it before enabling.

- `n_trials`, `pruning`, `final_validation`: see the inline `_comment_*` keys in `optimizer_config.example.json` for what each knob does.

The evaluator's own config (`evaluator/base/config.json` — LLM endpoint, chatbot path) is reused as-is; there's nothing optimizer-specific to set up there.

### Model choice

`search_space.model` is a flat sub-space (like `search_space.extraction`), sampled once per trial independent of `mode` — the answering LLM applies whether that trial ends up rag or agentic. `model` (the LLM name) is the common case: comparing several model names available on the same endpoint. `base_url`/`api_key` pick an entirely different provider; since every field in a sub-space is sampled independently, mixing multiple `(base_url, api_key)` values with multiple `model` values can pair a model name with the wrong endpoint's key. If you want to compare providers, either keep `base_url`/`api_key` to one value each (pin them via `fixed_params.model` while only `model` varies), or run separate studies per provider.

`reranker_model` (which cross-encoder re-ranks candidates) lives in `search_space.rag` instead, not here — it's rag-only and only takes effect when `reranker_enabled` is true.

### Extraction parameters

`search_space.extraction` sweeps the parameters of `extraction/process_docs.py` — chunking strategy, quality filtering, embedding model — the pipeline that builds the ChromaDB (rag) and `page_index.json` (agentic) the chatbot actually reads from. This is fundamentally more expensive than every other parameter here: changing the chatbot's `k` or `temperature` is a different CLI flag on the next subprocess call (seconds); changing the corpus's chunk size means re-running extraction over the whole corpus (minutes, no early-exit) before a single question can be asked.

To keep this tractable, `evaluator/base/extraction_cache.py` builds each distinct combination **at most once**: a trial resolves its sampled extraction params to a hash key, checks `evaluator/extraction_cache/<hash>_extracted/` for an existing build, and only invokes `process_docs.py` on a cache miss — a per-key file lock (`fcntl.flock`) serializes concurrent trials (`n_jobs > 1`) that land on the same never-before-seen combination instead of racing to build it twice. Once built, a matching chatbot config is generated alongside it (same `chromadb_path`/`embedding.model` as what was just built — a mismatch here silently returns wrong retrieval results, see `chatbot/config.json`'s own comment on this) and passed to every chatbot call in that trial via `--config`.

What this buys you: repeated trials that land on the same combination (common as TPE converges on a promising region) are essentially free after the first. What it doesn't buy you: there is still no early-exit from the build itself — pruning can only cut short the *per-question* evaluation loop that follows, not the extraction run. Optuna's TPE sampler also spends its first several trials (`n_startup_trials`, default ~10) sampling randomly before it has enough data to be smart, so a wide `search_space.extraction` can mean that many distinct — expensive — combinations get built in the opening trials alone. Keep each field to 2-3 values.

Fields: `use_token_chunking`, `chunking.max_tokens`, `chunking.overlap_tokens`, `chunking.char_chunk_size`, `chunking.char_overlap`, `quality.min_score`, `quality.min_word_count`, `quality.substantial_word_count`, `embedding.model`, `embedding.type`, `embedding.base_url`, `embedding.api_key` (dotted keys map to `extraction/config.json`'s nested blocks). `embedding.model`/`embedding.type` here are what determines the matching chatbot-side embedding config — don't also put `embedding.model` in a raw chatbot override elsewhere, extraction is the only place it's safe to vary. Leave `extraction_timeout_seconds` (top-level config key, or `--extraction-timeout`) at `null` unless you want a hard ceiling on one build.

### Concurrency and memory

`workers` and `n_jobs` both default to safe values (1) for a reason: unlike a typical hyperparameter search where a "worker" is a lightweight thread, each unit of concurrency here launches a full `chatbot.py` **subprocess** that reloads the embedding model, the Chroma vectorstore, and — if the sampled config has `reranker_enabled: true` — the `BAAI/bge-reranker-v2-m3` cross-encoder, which alone is several GB of RAM. `workers x n_jobs` concurrent chatbot calls can need roughly that many times several GB.

Pushing either past a modest value (say, `workers=4`) without checking available RAM first has driven this process into swap and gotten it OOM-killed mid-run in practice — which looks like the optimizer "hanging" (nothing in this script deadlocks on its own; `ThreadPoolExecutor`s are always scoped with `with`, so they don't leak). A killed run also leaves its currently-in-progress trial stuck in `RUNNING` state in the study database forever (Optuna doesn't retroactively mark it failed) — harmless (a resumed study just starts new trials at the next number and ignores it), but a giveaway that a previous run died mid-trial rather than actually finishing. `-1` (all CPUs) is supported for `workers` for parity with `evaluator/benchmark`'s `--workers`, but given the per-call cost here, prefer a number you've checked the machine can afford over `-1`.

### Fixing a parameter instead of optimizing it

To pin a hyperparameter to a specific value, move it out of `search_space` and into `fixed_params` under the same mode — it's then passed to every trial's chatbot call unchanged instead of being sampled. The same applies to `mode` itself: giving `search_space.mode.choices` a single element pins the mode.

For example, to fix mode `"rag"` with `k=6` while still optimizing the rest of the RAG hyperparameters:

```json
{
  "search_space": {
    "mode": {"choices": ["rag"]},
    "rag": {
      "top_n": {"type": "int", "low": 3, "high": 20},
      "temperature": {"type": "float", "low": 0.0, "high": 1.0},
      "reranker_enabled": {"type": "bool"},
      "deep_dive": {"type": "bool"},
      "hyde_enabled": {"type": "bool"},
      "bm25_enabled": {"type": "bool"},
      "title_boost_enabled": {"type": "bool"}
    },
    "agentic": { ... }
  },
  "fixed_params": {
    "rag": {"k": 6},
    "agentic": {}
  }
}
```

Note that `k` was removed from `search_space.rag` — a parameter can't appear in both `search_space` and `fixed_params` for the same mode (validation rejects the config if it does), since that would make it ambiguous whether it should be sampled or fixed. `search_space.agentic` still needs to be present and valid even though `agentic` is excluded from `mode.choices` here — it's simply never sampled, no trial will use it.

## Usage

```bash
# Run with the settings in optimizer_config.json
python optimizer.py

# Override a few settings from the command line
python optimizer.py --n-trials 50 --workers 4

# Quick smoke test on 3 questions to check everything is wired correctly
python optimizer.py --n-trials 2 --limit 3 --no-validation

# Resume: rerun the same command later (same study_name/storage) to add more
# trials to a study that was interrupted or that you want to extend
python optimizer.py --n-trials 20
```

Key flags (all mirror a `optimizer_config.json` key and override it): `--n-trials`, `--timeout`, `--limit`, `--workers`, `--chatbot-timeout`, `--extraction-timeout`, `--study-name`, `--storage`, `--no-pruning`, `--no-validation`, `--config <path>`.

## How a trial is scored

Each trial first samples `mode` (`"rag"` or `"agentic"`, from `search_space.mode.choices`), then samples hyperparameters from that mode's sub-space only — `mode` is just another optimized categorical choice, so the search can end up recommending either mode, not just tune one of them in isolation. Internally this is Optuna's standard pattern for a conditional/branching search space (a branch variable followed by branch-specific parameters); `TPESampler(multivariate=True, group=True)` is built to handle it, and each parameter is tracked per-branch (e.g. `rag`'s `temperature` and `agentic`'s `temperature` are distinct Optuna parameters) so their ranges can differ freely.

For the sampled configuration, every dataset question is answered by the chatbot and graded on the 4 metrics; the trial's value is the mean of all `4N` scores (equivalently, the mean of the 4 per-metric averages — the same "overall average" reported by `evaluator/base` and `evaluator/benchmark`). Optuna's direction is `maximize`. Trials of both modes are compared on the same footing, both during pruning and in the final ranking.

`dataset.json` is grouped by module/topic rather than shuffled, so a pruned trial that only got through the first N questions would see a biased, unrepresentative sample. The dataset is shuffled once (`dataset_shuffle_seed`) and that same order is reused for every trial, so a trial's running average at question 20 is comparable to every other trial's running average at question 20.

## Output

Each run writes `optimizer_results/optimizer_YYYYMMDD_HHMMSS.json`:

```json
{
  "metadata": { "timestamp": "...", "optimizer_config": { ... }, "n_questions": 94 },
  "best_trial": { "number": 12, "value": 7.8, "params": { "mode": "rag", "rag__k": 7, "rag__temperature": 0.4, ... } },
  "trials": [ { "number": 0, "state": "COMPLETE", "value": 6.9, "params": { ... }, "user_attrs": { "mode": "rag", "per_metric_avg": { ... }, "avg_request_time": 12.3 } } ],
  "final_validation": [
    {
      "trial_number": 12,
      "mode": "rag",
      "params": { "k_standard": 45, "temperature": 0.4, ... },
      "extraction_overrides": { "chunking.char_chunk_size": 900 },
      "search_value": 7.8,
      "validation_runs": [7.6, 7.9, 7.7],
      "validation_mean": 7.73,
      "validation_std": 0.15
    }
  ]
}
```

`trials[].params` holds Optuna's raw, per-branch-prefixed keys (`"mode"` plus `"<mode>__<name>"`/`"model__<name>"`/`"extraction__<name>"` for each sampled parameter) — that's what makes a `temperature` sampled under `rag` a genuinely distinct parameter from one sampled under `agentic`. `final_validation[].params` is the de-prefixed, ready-to-use hyperparameter dict for that candidate's mode (`rag`/`agentic` params merged with any `model` ones); `extraction_overrides` is `null` unless `search_space.extraction` was in play, in which case it's the dict passed to `extraction_cache.get_or_build()`.

`final_validation` (when enabled) is sorted by `validation_mean` descending — its first entry is the recommended configuration, also printed at the end of the run.

Optuna's study itself is persisted separately in `studies/<study_name>.db` (SQLite), which is what makes resuming/extending a study possible across sessions.

## Metrics

Same 4 metrics as `evaluator/base` and `evaluator/benchmark`:

1. Correctness - Factual accuracy (0-10)
2. Relevance - Overall relevance (0-10)
3. Groundedness - Grounding in documents (0-10)
4. Retrieval Relevance - Relevance of retrieved documents (0-10)

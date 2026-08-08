"""
On-demand, cached builds of extraction-parameter variants, shared by
evaluator/benchmark and evaluator/optimizer so both can sweep/optimize the
extraction pipeline (chunking, quality filtering, embedding model) the same
way they already sweep chatbot-side hyperparameters.

Unlike a chatbot query (seconds), rebuilding the corpus (extraction/process_docs.py)
takes minutes and produces a ~hundreds-of-MB ChromaDB, so every distinct
combination of extraction hyperparameters is built at most once and cached by
a hash of its parameter values — see get_or_build().
"""

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from evaluator.base.evaluator import CHATBOT_DIR, EXTRACTION_DIR, _kill_process_tree

# evaluator/base/extraction_cache.py -> evaluator/ (shared by benchmark and optimizer,
# so a variant built while benchmarking is reused by the optimizer and vice versa).
# The "_extracted" suffix matches an existing .gitignore pattern (*_extracted/).
CACHE_ROOT = Path(__file__).resolve().parent.parent / "extraction_cache"

# Every field extraction/config.json exposes that actually changes retrieval quality.
# Deliberately excludes project_name/modules/output_dir: those pick *which* docs are
# indexed, not a quality knob, and output_dir is always overridden to the cache dir.
TUNABLE_EXTRACTION_FIELDS = (
    "use_token_chunking",
    "chunking.max_tokens",
    "chunking.overlap_tokens",
    "chunking.char_chunk_size",
    "chunking.char_overlap",
    "quality.min_score",
    "quality.min_word_count",
    "quality.substantial_word_count",
    "embedding.model",
    "embedding.type",
    "embedding.base_url",
    "embedding.api_key",
)


def _set_dotted(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _strip_comments(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _hash_key(overrides: Dict[str, Any]) -> str:
    canonical = json.dumps(overrides, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _validate_overrides(overrides: Dict[str, Any]) -> None:
    unknown = set(overrides) - set(TUNABLE_EXTRACTION_FIELDS)
    if unknown:
        raise ValueError(
            f"Unknown extraction hyperparameter(s): {sorted(unknown)}. "
            f"Supported: {list(TUNABLE_EXTRACTION_FIELDS)}."
        )


def _build(overrides: Dict[str, Any], cache_dir: Path, timeout_seconds: Optional[float]) -> None:
    """Run extraction/process_docs.py with `overrides` merged in, writing its
    output straight into cache_dir. Raises on failure/timeout."""
    extraction_dir = Path(EXTRACTION_DIR)
    base_config_path = extraction_dir / "config.json"
    if not base_config_path.exists():
        raise FileNotFoundError(
            f"Extraction base config not found: {base_config_path}. "
            "Copy extraction/config.example.json to extraction/config.json first."
        )

    extraction_cfg = _strip_comments(json.loads(base_config_path.read_text(encoding="utf-8")))
    for dotted_key, value in overrides.items():
        _set_dotted(extraction_cfg, dotted_key, value)
    extraction_cfg["output_dir"] = str(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_config_path = tempfile.mkstemp(suffix=".json", dir=str(extraction_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tf:
            json.dump(extraction_cfg, tf)

        cmd = [sys.executable, "process_docs.py", "--config", temp_config_path]
        print(f"[extraction_cache] Building extraction variant {cache_dir.name} "
              f"({', '.join(f'{k}={v}' for k, v in overrides.items())}) ...")
        # start_new_session=True + _kill_process_tree on timeout: same reasoning as
        # _call_chatbot's own subprocess handling (evaluator/base/evaluator.py) — a
        # killed build shouldn't leave orphaned descendants still holding CPU/memory.
        proc = subprocess.Popen(cmd, cwd=str(extraction_dir), start_new_session=True)
        try:
            proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            proc.communicate()
            raise RuntimeError(
                f"Extraction build for variant {cache_dir.name} timed out after {timeout_seconds}s"
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Extraction build for variant {cache_dir.name} failed (exit code {proc.returncode})"
            )
    finally:
        os.unlink(temp_config_path)

    if not (cache_dir / "chromadb").exists():
        raise RuntimeError(
            f"Extraction build for variant {cache_dir.name} did not produce a chromadb/ directory"
        )
    print(f"[extraction_cache] Built variant {cache_dir.name}")


def _write_chatbot_config(overrides: Dict[str, Any], cache_dir: Path, out_path: Path) -> None:
    """Derive a chatbot config for this variant: same as the project's base
    chatbot/config.json, but pointed at this variant's chromadb/page_index and
    with embedding.model/type patched to match what was just built (mismatched
    embeddings between extraction and chatbot silently return wrong results —
    see chatbot/config.json's own comment on this)."""
    base_config_path = Path(CHATBOT_DIR) / "config.json"
    if not base_config_path.exists():
        raise FileNotFoundError(f"Chatbot base config not found: {base_config_path}.")

    chatbot_cfg = _strip_comments(json.loads(base_config_path.read_text(encoding="utf-8")))
    chatbot_cfg["chromadb_path"] = str((cache_dir / "chromadb").resolve())

    embedding_overrides = {
        field[len("embedding."):]: value
        for field, value in overrides.items()
        if field.startswith("embedding.")
    }
    if embedding_overrides:
        embedding_cfg = dict(chatbot_cfg.get("embedding") or {})
        embedding_cfg.update(embedding_overrides)
        chatbot_cfg["embedding"] = embedding_cfg

    if chatbot_cfg.get("agentic"):
        chatbot_cfg["agentic"] = dict(chatbot_cfg["agentic"])
        chatbot_cfg["agentic"]["page_index_path"] = str((cache_dir / "page_index.json").resolve())

    out_path.write_text(json.dumps(chatbot_cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def get_or_build(overrides: Dict[str, Any], timeout_seconds: Optional[float] = None) -> str:
    """Return the path to a ready-to-use chatbot config JSON for the extraction
    variant described by `overrides` (a dict of dotted TUNABLE_EXTRACTION_FIELDS
    names to values — fields left unset keep extraction/config.json's value).

    Builds it first if this exact combination hasn't been seen before; a
    per-variant flock serializes concurrent callers (e.g. optimizer trials
    running with n_jobs > 1) onto the same build instead of racing/duplicating
    a multi-minute extraction run.
    """
    _validate_overrides(overrides)
    key = _hash_key(overrides)
    cache_dir = CACHE_ROOT / f"{key}_extracted"
    chromadb_dir = cache_dir / "chromadb"
    chatbot_config_path = cache_dir / "chatbot_config.json"

    if chromadb_dir.exists() and chatbot_config_path.exists():
        return str(chatbot_config_path)

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = CACHE_ROOT / f"{key}.lock"
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            # Re-check: another thread/process may have built it while we waited.
            if not (chromadb_dir.exists() and chatbot_config_path.exists()):
                _build(overrides, cache_dir, timeout_seconds)
                _write_chatbot_config(overrides, cache_dir, chatbot_config_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    return str(chatbot_config_path)

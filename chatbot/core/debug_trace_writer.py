"""
DebugTraceWriter — dumps a deepagents/LangGraph run's message log to structured
JSON files for easier programmatic comparison across many debug runs.

Minimal port (2026-08-27): serializes the message list, no per-step wall time —
LangGraph does not stamp per-node timing. Full-parity timing via the streaming
path is deferred to task #146.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class DebugTraceWriter:
    """
    Serializes a run's message log to JSON, one file per run.

    Usage:
        writer = DebugTraceWriter(debug_dir="debug_traces")
        path = writer.dump(result["messages"], question="How do I configure X?",
                           extra={"steps_used": 4})
    """

    def __init__(self, debug_dir: str | Path = "debug_traces"):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def dump(
        self,
        messages: List[Any],
        question: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Write the message log to a timestamped JSON file.

        Args:
            messages: The LangGraph run's message list (result["messages"]).
                      May be empty (e.g. a run that crashed before producing one).
            question: The original task/question, stored for context.
            extra: Optional dict of additional metadata to store alongside
                   the trace (e.g. {"steps_used": 4, "degraded_answer": False}).

        Returns:
            Path to the written JSON file.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = self.debug_dir / f"trace_{ts}.json"

        serialized = [self._serialize_message(i, m) for i, m in enumerate(messages)]

        record = {
            "timestamp": ts,
            "question": question,
            "metadata": extra or {},
            "total_tokens": self._total_tokens(messages),
            "messages": serialized,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str, ensure_ascii=False)

        return out_path

    @staticmethod
    def _serialize_message(index: int, msg: Any) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "index": index,
            "role": getattr(msg, "type", type(msg).__name__),
            "content": getattr(msg, "content", ""),
        }

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                {"name": tc["name"], "args": tc.get("args", {})}
                if isinstance(tc, dict)
                else {"name": getattr(tc, "name", None), "args": getattr(tc, "args", {})}
                for tc in tool_calls
            ]

        tool_name = getattr(msg, "name", None)
        if tool_name:
            entry["tool_name"] = tool_name

        usage = getattr(msg, "usage_metadata", None)
        if usage:
            entry["usage_metadata"] = usage

        return entry

    @staticmethod
    def _total_tokens(messages: List[Any]) -> Optional[int]:
        total = 0
        seen_any = False
        for msg in messages:
            usage = getattr(msg, "usage_metadata", None)
            if not usage:
                continue
            seen_any = True
            total += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        return total if seen_any else None


def load_traces(debug_dir: str | Path) -> List[Dict[str, Any]]:
    """Convenience loader: reads all trace JSON files from a directory into a list of dicts."""
    debug_dir = Path(debug_dir)
    traces = []
    for path in sorted(debug_dir.glob("trace_*.json")):
        with open(path, encoding="utf-8") as f:
            traces.append(json.load(f))
    return traces

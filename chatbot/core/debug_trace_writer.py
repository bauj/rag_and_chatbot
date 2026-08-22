"""
DebugTraceWriter — dumps smolagents agent.memory to structured JSON files
for easier programmatic comparison across many debug runs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class DebugTraceWriter:
    """
    Serializes a smolagents CodeAgent's memory steps to JSON, one file per run.

    Usage:
        writer = DebugTraceWriter(debug_dir="debug_traces")
        path = writer.dump(agent, question="How do I configure X?", extra={"steps_used": 4})
    """

    def __init__(self, debug_dir: str | Path = "debug_traces"):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def dump(
        self,
        agent,
        question: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Write agent.memory to a timestamped JSON file.

        Args:
            agent: A smolagents CodeAgent (or anything with a `.memory.steps` list).
            question: The original task/question, stored for context.
            extra: Optional dict of additional metadata to store alongside
                   the trace (e.g. {"steps_used": 4, "degraded_answer": False}).

        Returns:
            Path to the written JSON file.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = self.debug_dir / f"trace_{ts}.json"

        steps = [self._serialize_step(i, step) for i, step in enumerate(agent.memory.steps)]

        record = {
            "timestamp": ts,
            "question": question,
            "metadata": extra or {},
            "total_duration_seconds": self._total_duration(steps),
            "total_tokens": self._total_tokens(steps),
            "steps": steps,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str, ensure_ascii=False)

        return out_path

    @staticmethod
    def _serialize_step(index: int, step: Any) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "index": index,
            "step_type": type(step).__name__,
        }

        model_output = getattr(step, "model_output", None)
        if model_output:
            entry["model_output"] = model_output

        tool_calls = getattr(step, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                {"name": tc.name, "arguments": tc.arguments} for tc in tool_calls
            ]

        observations = getattr(step, "observations", None)
        if observations:
            entry["observations"] = observations

        error = getattr(step, "error", None)
        if error:
            entry["error"] = str(error)

        # Some step types (e.g. PlanningStep) carry a plan instead of tool calls
        plan = getattr(step, "plan", None)
        if plan:
            entry["plan"] = plan

        # smolagents' ActionStep carries timing as a Timing object (.start_time/
        # .end_time/.duration), not flat attrs on the step itself — earlier
        # versions of this method looked for input_token_count/output_token_count/
        # duration directly on the step, which never existed, so every trace
        # silently recorded no timing/token data at all.
        timing = getattr(step, "timing", None)
        if timing is not None:
            entry["start_time"] = timing.start_time
            entry["end_time"] = timing.end_time
            entry["duration_seconds"] = timing.duration

        token_usage = getattr(step, "token_usage", None)
        if token_usage is not None:
            entry["input_tokens"] = token_usage.input_tokens
            entry["output_tokens"] = token_usage.output_tokens
            entry["total_tokens"] = token_usage.total_tokens

        return entry

    @staticmethod
    def _total_duration(steps: List[Dict[str, Any]]) -> Optional[float]:
        durations = [s["duration_seconds"] for s in steps if s.get("duration_seconds") is not None]
        return sum(durations) if durations else None

    @staticmethod
    def _total_tokens(steps: List[Dict[str, Any]]) -> Optional[int]:
        totals = [s["total_tokens"] for s in steps if s.get("total_tokens") is not None]
        return sum(totals) if totals else None


def load_traces(debug_dir: str | Path) -> List[Dict[str, Any]]:
    """Convenience loader: reads all trace JSON files from a directory into a list of dicts."""
    debug_dir = Path(debug_dir)
    traces = []
    for path in sorted(debug_dir.glob("trace_*.json")):
        with open(path, encoding="utf-8") as f:
            traces.append(json.load(f))
    return traces
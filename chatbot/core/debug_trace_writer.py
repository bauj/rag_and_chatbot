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

        record = {
            "timestamp": ts,
            "question": question,
            "metadata": extra or {},
            "steps": [self._serialize_step(i, step) for i, step in enumerate(agent.memory.steps)],
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

        # Token/timing info if smolagents populated it on this step
        for attr in ("input_token_count", "output_token_count", "duration"):
            val = getattr(step, attr, None)
            if val is not None:
                entry[attr] = val

        return entry


def load_traces(debug_dir: str | Path) -> List[Dict[str, Any]]:
    """Convenience loader: reads all trace JSON files from a directory into a list of dicts."""
    debug_dir = Path(debug_dir)
    traces = []
    for path in sorted(debug_dir.glob("trace_*.json")):
        with open(path, encoding="utf-8") as f:
            traces.append(json.load(f))
    return traces
# tests/test_debug_trace_writer.py
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.debug_trace_writer import DebugTraceWriter, load_traces


def _timing(start, end):
    return SimpleNamespace(start_time=start, end_time=end, duration=(end - start) if end is not None else None)


def _token_usage(input_tokens, output_tokens):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _make_agent(steps):
    agent = MagicMock()
    agent.memory.steps = steps
    return agent


def test_dump_writes_a_json_file(tmp_path):
    writer = DebugTraceWriter(debug_dir=tmp_path)
    step = SimpleNamespace(model_output="answer", tool_calls=None, observations=None,
                            error=None, plan=None, timing=None, token_usage=None)
    out_path = writer.dump(_make_agent([step]), question="q?")
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["question"] == "q?"


def test_dump_records_per_step_timing():
    writer = DebugTraceWriter.__new__(DebugTraceWriter)  # avoid touching the filesystem
    step = SimpleNamespace(model_output=None, tool_calls=None, observations=None,
                            error=None, plan=None, timing=_timing(10.0, 12.5), token_usage=None)
    entry = writer._serialize_step(0, step)
    assert entry["start_time"] == 10.0
    assert entry["end_time"] == 12.5
    assert entry["duration_seconds"] == 2.5


def test_dump_records_per_step_token_usage():
    writer = DebugTraceWriter.__new__(DebugTraceWriter)
    step = SimpleNamespace(model_output=None, tool_calls=None, observations=None,
                            error=None, plan=None, timing=None, token_usage=_token_usage(100, 50))
    entry = writer._serialize_step(0, step)
    assert entry["input_tokens"] == 100
    assert entry["output_tokens"] == 50
    assert entry["total_tokens"] == 150


def test_dump_omits_timing_and_tokens_when_absent():
    writer = DebugTraceWriter.__new__(DebugTraceWriter)
    step = SimpleNamespace(model_output=None, tool_calls=None, observations=None,
                            error=None, plan=None, timing=None, token_usage=None)
    entry = writer._serialize_step(0, step)
    assert "duration_seconds" not in entry
    assert "total_tokens" not in entry


def test_dump_aggregates_total_duration_and_tokens(tmp_path):
    writer = DebugTraceWriter(debug_dir=tmp_path)
    steps = [
        SimpleNamespace(model_output=None, tool_calls=None, observations=None, error=None,
                         plan=None, timing=_timing(0.0, 1.0), token_usage=_token_usage(10, 5)),
        SimpleNamespace(model_output=None, tool_calls=None, observations=None, error=None,
                         plan=None, timing=_timing(1.0, 3.5), token_usage=_token_usage(20, 15)),
    ]
    out_path = writer.dump(_make_agent(steps), question="q?")
    data = json.loads(out_path.read_text())
    assert data["total_duration_seconds"] == 3.5
    assert data["total_tokens"] == 50


def test_dump_total_duration_none_when_no_step_has_timing(tmp_path):
    writer = DebugTraceWriter(debug_dir=tmp_path)
    step = SimpleNamespace(model_output=None, tool_calls=None, observations=None,
                            error=None, plan=None, timing=None, token_usage=None)
    out_path = writer.dump(_make_agent([step]), question="q?")
    data = json.loads(out_path.read_text())
    assert data["total_duration_seconds"] is None
    assert data["total_tokens"] is None


def test_load_traces_reads_all_written_files(tmp_path):
    writer = DebugTraceWriter(debug_dir=tmp_path)
    step = SimpleNamespace(model_output=None, tool_calls=None, observations=None,
                            error=None, plan=None, timing=None, token_usage=None)
    writer.dump(_make_agent([step]), question="q1")
    writer.dump(_make_agent([step]), question="q2")

    traces = load_traces(tmp_path)
    assert len(traces) == 2
    assert {t["question"] for t in traces} == {"q1", "q2"}

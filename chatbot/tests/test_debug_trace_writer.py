# tests/test_debug_trace_writer.py
import json
from types import SimpleNamespace

from core.debug_trace_writer import DebugTraceWriter, load_traces


def _human(content):
    return SimpleNamespace(type="human", content=content, tool_calls=[], name=None,
                           usage_metadata=None)


def _ai(content, tool_calls=None, usage_metadata=None):
    return SimpleNamespace(type="ai", content=content, tool_calls=tool_calls or [],
                           name=None, usage_metadata=usage_metadata)


def _tool(content, name):
    return SimpleNamespace(type="tool", content=content, tool_calls=[], name=name,
                           usage_metadata=None)


def test_dump_writes_a_json_file(tmp_path):
    writer = DebugTraceWriter(debug_dir=tmp_path)
    out_path = writer.dump([_human("q?"), _ai("an answer")], question="q?")
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["question"] == "q?"


def test_dump_serializes_one_entry_per_message_with_role_and_content(tmp_path):
    writer = DebugTraceWriter(debug_dir=tmp_path)
    messages = [_human("Question: what is X?"), _ai("X is a thing.")]
    out_path = writer.dump(messages, question="what is X?")
    data = json.loads(out_path.read_text())

    assert [m["index"] for m in data["messages"]] == [0, 1]
    assert data["messages"][0]["role"] == "human"
    assert data["messages"][0]["content"] == "Question: what is X?"
    assert data["messages"][1]["role"] == "ai"
    assert data["messages"][1]["content"] == "X is a thing."


def test_dump_records_tool_calls_from_ai_messages(tmp_path):
    writer = DebugTraceWriter(debug_dir=tmp_path)
    ai = _ai("", tool_calls=[{"name": "search_sections_tool", "args": {"query": "sketch"}, "id": "c1"}])
    out_path = writer.dump([_human("q"), ai], question="q")
    data = json.loads(out_path.read_text())

    assert data["messages"][1]["tool_calls"] == [
        {"name": "search_sections_tool", "args": {"query": "sketch"}}
    ]


def test_dump_records_tool_message_name(tmp_path):
    writer = DebugTraceWriter(debug_dir=tmp_path)
    out_path = writer.dump([_tool("page text here", name="read_page_tool")], question="q")
    data = json.loads(out_path.read_text())

    assert data["messages"][0]["role"] == "tool"
    assert data["messages"][0]["tool_name"] == "read_page_tool"


def test_dump_aggregates_total_tokens_from_usage_metadata(tmp_path):
    writer = DebugTraceWriter(debug_dir=tmp_path)
    messages = [
        _human("q"),
        _ai("partial", usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}),
        _ai("final", usage_metadata={"input_tokens": 150, "output_tokens": 30, "total_tokens": 180}),
    ]
    out_path = writer.dump(messages, question="q")
    data = json.loads(out_path.read_text())

    assert data["total_tokens"] == 300


def test_dump_total_tokens_none_when_no_usage_reported(tmp_path):
    writer = DebugTraceWriter(debug_dir=tmp_path)
    out_path = writer.dump([_human("q"), _ai("a")], question="q")
    data = json.loads(out_path.read_text())

    assert data["total_tokens"] is None


def test_dump_has_no_timing_fields(tmp_path):
    """Per-step wall time is deferred to task #146 — the minimal port carries none."""
    writer = DebugTraceWriter(debug_dir=tmp_path)
    out_path = writer.dump([_human("q"), _ai("a")], question="q")
    data = json.loads(out_path.read_text())

    assert "total_duration_seconds" not in data
    for m in data["messages"]:
        assert "duration_seconds" not in m


def test_dump_stores_extra_metadata(tmp_path):
    writer = DebugTraceWriter(debug_dir=tmp_path)
    out_path = writer.dump([_ai("a")], question="q", extra={"steps_used": 4, "degraded_answer": False})
    data = json.loads(out_path.read_text())

    assert data["metadata"]["steps_used"] == 4
    assert data["metadata"]["degraded_answer"] is False


def test_dump_handles_empty_message_list(tmp_path):
    writer = DebugTraceWriter(debug_dir=tmp_path)
    out_path = writer.dump([], question="q", extra={"error": "GraphRecursionError"})
    data = json.loads(out_path.read_text())

    assert data["messages"] == []
    assert data["total_tokens"] is None
    assert data["metadata"]["error"] == "GraphRecursionError"


def test_load_traces_reads_all_written_files(tmp_path):
    writer = DebugTraceWriter(debug_dir=tmp_path)
    writer.dump([_ai("a")], question="q1")
    writer.dump([_ai("b")], question="q2")

    traces = load_traces(tmp_path)
    assert len(traces) == 2
    assert {t["question"] for t in traces} == {"q1", "q2"}

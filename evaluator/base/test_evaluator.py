"""Tests for evaluator.py's judge-response parsing."""
import json
from unittest.mock import MagicMock

import pytest

from evaluator.base.evaluator import MistralLLM


class _FakeGrade(dict):
    """Minimal stand-in for the {score, explanation} TypedDict schema."""


def _bot_with_response(content: str) -> MistralLLM:
    bot = MistralLLM(api_url="http://fake", model="m", api_key="k")
    bot.with_structured_output(_FakeGrade)
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.raise_for_status = lambda: None
    fake_response.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    bot._post = MagicMock(return_value=fake_response)
    return bot


def test_invoke_parses_clean_json():
    bot = _bot_with_response('{"score": 8, "explanation": "fine"}')
    result = bot.invoke([{"role": "user", "content": "grade this"}])
    assert result == {"score": 8, "explanation": "fine"}


def test_invoke_parses_json_wrapped_in_markdown_fence():
    """
    Some judge endpoints accept response_format=json_schema with HTTP 200
    but still reply in a ```json ... ``` fence instead of raw JSON. Before
    the fix, this fell all the way through to {"content": <fenced text>},
    which has no "score" key and gets misread as a grading failure.
    """
    fenced = '```json\n{\n  "score": 8,\n  "explanation": "The document is relevant."\n}\n```'
    bot = _bot_with_response(fenced)
    result = bot.invoke([{"role": "user", "content": "grade this"}])
    assert result == {"score": 8, "explanation": "The document is relevant."}


def test_invoke_parses_json_wrapped_in_plain_fence():
    fenced = '```\n{"score": 5, "explanation": "ok"}\n```'
    bot = _bot_with_response(fenced)
    result = bot.invoke([{"role": "user", "content": "grade this"}])
    assert result == {"score": 5, "explanation": "ok"}


def test_invoke_falls_back_to_content_dict_when_truly_unparseable():
    bot = _bot_with_response("not json at all, no fence either")
    result = bot.invoke([{"role": "user", "content": "grade this"}])
    assert result == {"content": "not json at all, no fence either"}


def test_invoke_salvages_json_with_missing_explanation_quote():
    """
    The judge LLM occasionally drops the opening quote on "explanation"
    (task #144): {"explanation":The text says X} instead of
    {"explanation":"The text says X"}. Otherwise valid JSON.
    """
    malformed = '{"score": 8, "explanation":The text supports the answer."}'
    bot = _bot_with_response(malformed)
    result = bot.invoke([{"role": "user", "content": "grade this"}])
    assert result == {"score": 8, "explanation": "The text supports the answer."}


def test_run_evaluation_only_runs_requested_metrics(monkeypatch):
    """
    no-rag mode has no retrieved documents, so groundedness/retrieval_relevance
    are meaningless — run_evaluation's `metrics` param lets a caller skip them
    entirely rather than running (and paying for) a judge call that grades
    "no documents" every time.
    """
    import evaluator.base.evaluator as ev

    monkeypatch.setattr(ev, "correctness", MagicMock(return_value={"score": 9, "explanation": "ok"}))
    monkeypatch.setattr(ev, "relevance", MagicMock(return_value={"score": 8, "explanation": "ok"}))
    groundedness_mock = MagicMock(return_value={"score": 0, "explanation": "should not be called"})
    retrieval_relevance_mock = MagicMock(return_value={"score": 0, "explanation": "should not be called"})
    monkeypatch.setattr(ev, "groundedness", groundedness_mock)
    monkeypatch.setattr(ev, "retrieval_relevance", retrieval_relevance_mock)

    result = ev.run_evaluation(
        "q", {"answer": "a", "documents": []}, "ref",
        metrics=["correctness", "relevance"],
    )

    assert set(result.keys()) == {"correctness", "relevance"}
    groundedness_mock.assert_not_called()
    retrieval_relevance_mock.assert_not_called()


def test_run_evaluation_defaults_to_all_metrics(monkeypatch):
    import evaluator.base.evaluator as ev

    for name in ("correctness", "relevance", "groundedness", "retrieval_relevance"):
        monkeypatch.setattr(ev, name, MagicMock(return_value={"score": 5, "explanation": "ok"}))

    result = ev.run_evaluation("q", {"answer": "a", "documents": []}, "ref")

    assert set(result.keys()) == {"correctness", "relevance", "groundedness", "retrieval_relevance"}

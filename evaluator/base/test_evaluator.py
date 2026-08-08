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

# tests/test_grounding_judge.py
from unittest.mock import MagicMock

from core.grounding_judge import (
    check_grounding,
    concatenated_observations,
    llm_builtin_classifier,
    _documented_arg_counts,
)


def _agent_with_observations(*observations):
    agent = MagicMock()
    agent.memory.steps = [MagicMock(observations=obs) for obs in observations]
    return agent


# ---------------------------------------------------------------------------
# concatenated_observations
# ---------------------------------------------------------------------------

def test_concatenated_observations_joins_step_observations():
    agent = _agent_with_observations("first page text", "second page text")
    result = concatenated_observations(agent)
    assert "first page text" in result
    assert "second page text" in result


def test_concatenated_observations_skips_steps_with_no_observations():
    agent = _agent_with_observations("real text", None)
    result = concatenated_observations(agent)
    assert result == "real text"


# ---------------------------------------------------------------------------
# check_grounding — gating
# ---------------------------------------------------------------------------

def test_check_grounding_returns_empty_for_prose_with_no_code():
    agent = _agent_with_observations("some documentation")
    result = check_grounding("Just a plain-language answer, no code here.", agent)
    assert result == []


# ---------------------------------------------------------------------------
# check_grounding — Python (AST path)
# ---------------------------------------------------------------------------

def test_check_grounding_flags_python_call_not_in_observations():
    agent = _agent_with_observations("addFeature(shape, name) is defined on ModelAPI.")
    answer = "```python\nresult = deleteEverything(shape)\n```"
    result = check_grounding(answer, agent)
    assert [c["call"] for c in result] == ["deleteEverything"]


def test_check_grounding_passes_python_call_present_in_observations():
    agent = _agent_with_observations("addFeature(shape, name) is defined on ModelAPI.")
    answer = "```python\nresult = addFeature(shape, 'x')\n```"
    result = check_grounding(answer, agent)
    assert result == []


def test_check_grounding_ignores_python_builtins():
    agent = _agent_with_observations("no relevant API here")
    answer = "```python\nprint(len(str(42)))\n```"
    result = check_grounding(answer, agent)
    assert result == []


def test_check_grounding_handles_unfenced_python_that_parses():
    agent = _agent_with_observations("addFeature(shape, name) is defined on ModelAPI.")
    answer = "deleteEverything(shape)"
    result = check_grounding(answer, agent)
    assert [c["call"] for c in result] == ["deleteEverything"]


def test_check_grounding_dedupes_repeated_unknown_call():
    agent = _agent_with_observations("nothing relevant")
    answer = "```python\nfoo(1)\nfoo(2)\n```"
    result = check_grounding(answer, agent)
    assert len(result) == 1
    assert result[0]["call"] == "foo"


# ---------------------------------------------------------------------------
# check_grounding — non-Python (regex fallback path)
# ---------------------------------------------------------------------------

def test_check_grounding_flags_cpp_call_not_in_observations_via_regex_fallback():
    agent = _agent_with_observations("Feature::addFeature(shape, name) is the C++ entry point.")
    answer = "```cpp\nauto result = Feature::deleteEverything(shape);\n```"
    result = check_grounding(answer, agent)
    assert "deleteEverything" in [c["call"] for c in result]


def test_check_grounding_ignores_cpp_keywords_via_regex_fallback():
    agent = _agent_with_observations("nothing relevant")
    answer = "```cpp\nif (x) { return std::cout; }\n```"
    result = check_grounding(answer, agent)
    assert result == []


# ---------------------------------------------------------------------------
# check_grounding — llm_classify_fn filter layer
# ---------------------------------------------------------------------------

def test_check_grounding_llm_classify_fn_can_filter_a_flagged_name():
    agent = _agent_with_observations("nothing relevant")
    answer = "```python\ncustomBuiltinLikeThing(1)\n```"
    result = check_grounding(answer, agent, llm_classify_fn=lambda name: True)
    assert result == []


def test_check_grounding_llm_classify_fn_cannot_wave_through_names_already_matched():
    # A name that DOES appear in observations is never even offered to the
    # classify function — proves the filter can only remove, not add, flags.
    agent = _agent_with_observations("addFeature(shape, name) is defined on ModelAPI.")
    answer = "```python\naddFeature(shape, 'x')\n```"
    calls_seen = []

    def classify(name):
        calls_seen.append(name)
        return False

    result = check_grounding(answer, agent, llm_classify_fn=classify)
    assert result == []
    assert calls_seen == []


def test_check_grounding_llm_classify_fn_false_keeps_the_flag():
    agent = _agent_with_observations("nothing relevant")
    answer = "```python\nfoo(1)\n```"
    result = check_grounding(answer, agent, llm_classify_fn=lambda name: False)
    assert [c["call"] for c in result] == ["foo"]


# ---------------------------------------------------------------------------
# _documented_arg_counts
# ---------------------------------------------------------------------------

def test_documented_arg_counts_single_signature():
    obs = "model.addFeature(shape, name) is the call."
    assert _documented_arg_counts("addFeature", obs) == {2}


def test_documented_arg_counts_multiple_signatures():
    obs = (
        "model.addBox(Part_doc, DX, DY, DZ)\n"
        "model.addBox(Part_doc, point1, point2)\n"
        "model.addBox(Part_doc, OX, OY, OZ, DX, DY, DZ)\n"
    )
    assert _documented_arg_counts("addBox", obs) == {4, 3, 7}


def test_documented_arg_counts_ignores_nested_parens():
    obs = "model.addCylinder(Part_doc, model.selection(\"VERTEX\", \"Origin\"), axis, 20, 60)"
    # 5 top-level args: Part_doc, selection(...), axis, 20, 60 — the nested
    # selection(...) call's own comma must not inflate the count.
    assert _documented_arg_counts("addCylinder", obs) == {5}


def test_documented_arg_counts_no_occurrence_returns_empty_set():
    assert _documented_arg_counts("addBox", "nothing relevant here") == set()


def test_documented_arg_counts_matches_newline_tokenized_html_text():
    # read_page() extracts HTML via BeautifulSoup get_text(separator='\n'),
    # so real retrieved doc text has every token — including "(" and "," —
    # on its own line, not a contiguous "name(args)" string. A literal
    # "name(" match would silently never fire against real documentation.
    tokenized = "model\n.\naddBox\n(\nPart_doc\n,\n0\n,\n0\n,\n0\n,\n20\n,\n20\n,\n20\n)\n8\nmodel"
    assert _documented_arg_counts("addBox", tokenized) == {7}


# ---------------------------------------------------------------------------
# check_grounding — overload ambiguity flag
# ---------------------------------------------------------------------------

def test_check_grounding_flags_grounded_call_with_multiple_documented_overloads():
    observations = (
        "model.addBox(Part_doc, DX, DY, DZ) — by dimensions.\n"
        "model.addBox(Part_doc, OX, OY, OZ, DX, DY, DZ) — by point and dimensions, "
        "where OX/OY/OZ is the center and DX/DY/DZ are half-lengths.\n"
    )
    agent = _agent_with_observations(observations)
    answer = "```python\nBox_1 = model.addBox(doc, 0, 0, 60, 10, 10, 10)\n```"
    result = check_grounding(answer, agent)
    assert len(result) == 1
    assert result[0]["call"] == "addBox"
    assert "multiple documented signatures" in result[0]["reason"]


def test_check_grounding_does_not_flag_grounded_call_with_one_signature():
    observations = "model.addFeature(shape, name) is the only documented form."
    agent = _agent_with_observations(observations)
    answer = "```python\nmodel.addFeature(shape, 'x')\n```"
    result = check_grounding(answer, agent)
    assert result == []


def test_check_grounding_overload_flag_not_offered_to_llm_classify_fn():
    # Overload ambiguity is a structural fact about the docs, not a builtin
    # question — the classify_fn layer (which only answers "is this a
    # builtin") must never be consulted for it.
    observations = (
        "model.addBox(Part_doc, DX, DY, DZ)\n"
        "model.addBox(Part_doc, OX, OY, OZ, DX, DY, DZ)\n"
    )
    agent = _agent_with_observations(observations)
    answer = "```python\nBox_1 = model.addBox(doc, 0, 0, 60, 10, 10, 10)\n```"
    calls_seen = []

    def classify(name):
        calls_seen.append(name)
        return True  # would wave through if it were ever consulted

    result = check_grounding(answer, agent, llm_classify_fn=classify)
    assert len(result) == 1
    assert calls_seen == []


# ---------------------------------------------------------------------------
# llm_builtin_classifier
# ---------------------------------------------------------------------------

def test_llm_builtin_classifier_returns_true_on_yes_response():
    judge_model = MagicMock(return_value=MagicMock(content="YES"))
    classify = llm_builtin_classifier(judge_model)
    assert classify("malloc") is True


def test_llm_builtin_classifier_returns_false_on_no_response():
    judge_model = MagicMock(return_value=MagicMock(content="NO"))
    classify = llm_builtin_classifier(judge_model)
    assert classify("addFeature") is False


def test_llm_builtin_classifier_fails_closed_on_exception():
    def broken_model(messages):
        raise RuntimeError("endpoint down")

    classify = llm_builtin_classifier(broken_model)
    assert classify("anything") is False

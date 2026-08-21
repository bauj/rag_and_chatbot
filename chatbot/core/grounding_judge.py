import json
import re
from typing import Any, List


_JUDGE_PROMPT = """You are checking whether a piece of generated code is grounded in the \
documentation excerpts below. A call is grounded only if BOTH of these hold:

1. Name: the function/method name actually appears in the documentation excerpts.
2. Usage: the way the call is made in the code (argument count, whether arguments are \
positional or keyword, argument order, and argument types) matches how the documentation \
shows that function being called. A call that uses the right name but invents keyword \
arguments, adds/drops/reorders parameters, or passes a different kind of value than the \
documentation's example (e.g. a raw tuple where the documentation shows a selection object) \
is NOT grounded, even though the name is correct.

Do not judge business logic, variable naming, or whether the code accomplishes the user's \
goal — only whether each call is grounded per the two rules above.

Documentation excerpts:
---
{observations}
---

Code to check:
---
{code}
---

Respond ONLY with JSON, no other text, in this exact format:
{{"grounded": true}}
or, if any calls are not grounded (by name, by usage, or both):
{{"grounded": false, "ungrounded_calls": [
  {{"call": "name1", "reason": "unknown name" | "signature mismatch: <short explanation>"}}
]}}
"""


def concatenated_observations(agent) -> str:
    """Join every step's observations into one text blob for the judge to check against."""
    parts = []
    for step in getattr(agent.memory, "steps", []):
        obs = getattr(step, "observations", None)
        if obs:
            parts.append(str(obs))
    return "\n".join(parts)


def check_grounding_llm(code: str, agent, judge_model) -> List[dict]:
    """
    Ask a separate LLM call whether `code` only uses APIs present in the
    agent's retrieved observations, checking both that each call's name
    exists in the docs and that it's called the way the docs show (argument
    count/order/kind — catches a real function name used with a fabricated
    signature, not just an invented function name).

    Returns a list of {"call": str, "reason": str} dicts for anything
    flagged, empty if everything checks out or the judge call fails open.

    judge_model: any object exposing a smolagents-style __call__ that accepts
    a list of {"role": ..., "content": ...} messages and returns a response
    with a `.content` string. Pass a cheap/fast model here — this is a
    verification pass, not a generation pass, so it doesn't need to be the
    same model that wrote the code.
    """
    observations = concatenated_observations(agent)
    if not observations.strip():
        # Nothing to verify against — treat as ungrounded by default,
        # consistent with "no observations means no grounding".
        return [{"call": "*", "reason": "no observations available to verify against"}]

    prompt = _JUDGE_PROMPT.format(observations=observations, code=code)

    try:
        response = judge_model([{"role": "user", "content": prompt}])
        raw = response.content if hasattr(response, "content") else str(response)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Strip a leading ```json / ``` fence and a trailing ``` fence,
            # since models often wrap JSON output in one despite being told
            # not to.
            cleaned = re.sub(r'^```[a-zA-Z]*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```$', '', cleaned)
        result = json.loads(cleaned.strip())
    except Exception:
        # Judge call failed or returned non-JSON — fail open rather than
        # blocking a possibly-fine answer on a broken verification step.
        return []

    if result.get("grounded", True):
        return []
    return list(result.get("ungrounded_calls", []))
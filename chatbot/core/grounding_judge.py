"""
grounding_judge — deterministic check that code in an agentic answer only
calls names that actually appear in the documentation the agent retrieved.

Replaces an earlier version that asked the *same* weak/small LLM that wrote
the code to also judge whether it was grounded. Two problems with that:
literature on self-verification shows LLMs exhibit self-preference bias
judging their own output, and the judge failed open (returned "grounded")
whenever its own JSON response didn't parse — exactly the failure mode a
weak model hits most often, silently disabling the safety net for the
models it was meant to help. See project memory
project-rag-agentic-reliability-wip for the full writeup.

check_grounding() only verifies call *names* exist in the retrieved text —
not argument count/order/kind, which the old LLM judge attempted but which
needs real signature data (Doxygen memitem parsing or Sphinx autodoc, not
free-text scanning) to check without guessing; that's a follow-up, not done
here. Python code is parsed with `ast` for accurate call-name extraction;
any other language (or unfenced/unparseable code) falls back to a regex
scan for `identifier(`, which can't distinguish a call from a declaration or
macro as reliably but is fully language-agnostic.
"""

import ast
import builtins
import re
from typing import Callable, List, Optional, Tuple


_CODE_FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

_BUILTIN_NAMES = set(dir(builtins))
# Common C++ keywords/std names free-text scanning would otherwise flag as
# "unknown" on every single answer that contains any C++ at all.
_CPP_KEYWORDS = {
    "if", "else", "for", "while", "do", "switch", "case", "break", "continue",
    "return", "new", "delete", "sizeof", "static_cast", "dynamic_cast",
    "reinterpret_cast", "const_cast", "typename", "template", "namespace",
    "using", "auto", "printf", "cout", "cin", "endl", "assert",
}


def concatenated_observations(agent) -> str:
    """Join every step's observations into one text blob to check calls against."""
    parts = []
    for step in getattr(agent.memory, "steps", []):
        obs = getattr(step, "observations", None)
        if obs:
            parts.append(str(obs))
    return "\n".join(parts)


def _find_code_blocks(answer: str) -> List[Tuple[str, str]]:
    """
    Return (language, code) pairs for every fenced code block in `answer`.
    Falls back to treating the whole answer as one Python block if it has no
    fences but parses as valid standalone Python (English prose essentially
    never does) — covers a model that wrote code without wrapping it in
    markdown fences.
    """
    fenced = _CODE_FENCE_RE.findall(answer)
    if fenced:
        return [(lang.lower(), code) for lang, code in fenced]
    if _extract_python_calls(answer) is not None:
        return [("python", answer)]
    return []


def _extract_python_calls(code: str) -> Optional[List[str]]:
    """AST-based call-name extraction. Returns None if `code` isn't valid Python."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
    return names


def _extract_calls(language: str, code: str) -> List[str]:
    if language in ("python", "py", ""):
        names = _extract_python_calls(code)
        if names is not None:
            return names
    return _CALL_RE.findall(code)


def check_grounding(
    answer: str,
    agent,
    llm_classify_fn: Optional[Callable[[str], bool]] = None,
) -> List[dict]:
    """
    Extract call names from every code block in `answer` and flag any that
    appear in neither the retrieved documentation nor a static builtin
    allowlist. Returns [] if `answer` has no code blocks at all.

    llm_classify_fn: optional callable(name) -> bool, asked only about names
    that survive the static allowlist — a narrow "is this a language/stdlib
    builtin" question, not a grounding verifier. It can only shrink the
    result (filter a false positive), never wave through a name that
    genuinely doesn't appear in the docs, so a truly hallucinated project
    API is unaffected by whatever it returns. Pass None to skip this layer
    and rely on the static allowlist alone.
    """
    blocks = _find_code_blocks(answer)
    if not blocks:
        return []

    observations_lower = concatenated_observations(agent).lower()

    flagged = {}
    for language, code in blocks:
        for name in _extract_calls(language, code):
            if name in flagged:
                continue
            if name in _BUILTIN_NAMES or name in _CPP_KEYWORDS:
                continue
            if name.lower() in observations_lower:
                continue
            if llm_classify_fn is not None and llm_classify_fn(name):
                continue
            flagged[name] = {"call": name, "reason": "unknown name"}

    return list(flagged.values())


def llm_builtin_classifier(judge_model) -> Callable[[str], bool]:
    """
    Build a classify(name) -> bool function that asks `judge_model` a single
    narrow yes/no question — "is this a language/stdlib builtin" — for names
    the static allowlist in check_grounding() didn't recognize (e.g. C++
    std:: members, less-common Python builtins). See check_grounding's
    llm_classify_fn docstring for why this is safe even with a weak model:
    it only filters, never grounds.

    Fails CLOSED (returns False, i.e. "not a builtin", keeping the flag) on
    any classify error — the opposite of the old judge's fail-open, since
    there's no longer a second full-grounding LLM call whose failure should
    default to trusting the answer.
    """
    def classify(name: str) -> bool:
        prompt = (
            f"Answer with exactly one word, YES or NO: is `{name}` a "
            "standard-library or language builtin function/method name "
            "(e.g. print, len, str, malloc, std::vector, cout), as opposed "
            "to a project-specific API name?"
        )
        try:
            response = judge_model([{"role": "user", "content": prompt}])
            raw = response.content if hasattr(response, "content") else str(response)
            return raw.strip().upper().startswith("Y")
        except Exception:
            return False

    return classify

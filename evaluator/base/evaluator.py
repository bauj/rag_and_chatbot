import re
import subprocess
import sys
import json
import os
import signal
import threading
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing_extensions import Annotated, TypedDict, get_args
import requests
from typing import Any, Optional, Dict, List

############################################################################################
### Load configuration from evaluator/config.json with fallback on environment variables ###
############################################################################################

def load_evaluator_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.json"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(f"WARNING: config file not found at {config_path}. Using environment defaults.")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as cf:
            return json.load(cf)
    except Exception as e:
        print(f"WARNING: could not read config file {config_path}: {e}")
        return {}

evaluator_config = load_evaluator_config()

# Extract LLM configuration from the "llm" section
llm_config = evaluator_config.get("llm", {})

LLM_API_URL = os.getenv("MISTRAL_API_URL", llm_config.get("base_url"))
LLM_MODEL = os.getenv("MISTRAL_MODEL", llm_config.get("model"))
LLM_API_KEY = os.getenv("MISTRAL_API_KEY", llm_config.get("api_key"))
SSL_CERTIF = os.getenv("SSL_CERTIF", llm_config.get("ssl_cert_file"))

CHATBOT_DIR = os.getenv("CHATBOT_DIR", evaluator_config.get("chatbot_path"))
# Only needed by evaluator/base/extraction_cache.py (optimizer/benchmark's extraction
# hyperparameters); falls back to the same relative depth as chatbot_path's default.
EXTRACTION_DIR = os.getenv("EXTRACTION_DIR", evaluator_config.get("extraction_path", "../../extraction"))

def _validate_config() -> None:
    """Fail fast with an actionable message if required config is missing,
    instead of running the whole dataset and quietly producing all-0.0 scores
    (e.g. because config.json failed to parse and was silently replaced with {})."""
    missing = [
        name for name, value in [
            ("llm.base_url (or MISTRAL_API_URL)", LLM_API_URL),
            ("llm.model (or MISTRAL_MODEL)", LLM_MODEL),
            ("llm.api_key (or MISTRAL_API_KEY)", LLM_API_KEY),
            ("chatbot_path (or CHATBOT_DIR)", CHATBOT_DIR),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required evaluator configuration: " + ", ".join(missing) + ". "
            "Check evaluator/base/config.json (copy it from config.example.json if it "
            "doesn't exist yet) or set the corresponding environment variables."
        )

evaluator_options = evaluator_config.get("evaluator_options", {})

AGENTIC_MODE = bool(evaluator_options.get("agentic_mode", False))

# "mode" is the general form ("rag", "agentic", "no-rag" — anything chatbot.py's
# --mode accepts); "agentic_mode" is the older boolean-only knob, kept for
# backwards compatibility with existing evaluator_options.json files.
EVAL_MODE = evaluator_options.get("mode") or ("agentic" if AGENTIC_MODE else "rag")

# groundedness/retrieval_relevance grade the retrieved documents, which
# no-rag mode never has — skip them rather than spend two judge calls per
# question grading "no documents" every time.
EVAL_METRICS_FOR_MODE = ["correctness", "relevance"] if EVAL_MODE == "no-rag" else None

# Grading calls go straight over HTTP with no subprocess involved, so unlike
# the chatbot subprocess call (bounded by --timeout) a stalled connection here would
# otherwise hang forever. Bound it, and retry transient connection failures a couple
# of times before giving up — a dropped connection isn't the same as a bad answer and
# shouldn't silently show up as a fake 0.0 score.
LLM_REQUEST_TIMEOUT = float(os.getenv("MISTRAL_REQUEST_TIMEOUT", "120"))
LLM_REQUEST_RETRIES = int(os.getenv("MISTRAL_REQUEST_RETRIES", "2"))

############################################################################################
################################ Custom Mistral LLM client #################################
############################################################################################

# Maps the Python types used in the Annotated[...] grade TypedDicts to JSON Schema types.
_JSON_TYPE_BY_PY_NAME = {"str": "string", "float": "number", "int": "number", "bool": "boolean"}

def _typed_dict_to_json_schema(schema_cls: type, name: str) -> dict:
    """Convert a TypedDict of Annotated[type, ..., description] fields into an
    OpenAI-compatible JSON schema, so the grading LLM is actually constrained to
    the {score, explanation} shape instead of relying on it to follow free-text
    instructions."""
    properties = {}
    required = []
    for field_name, annotation in schema_cls.__annotations__.items():
        args = get_args(annotation)
        field_type = args[0] if args else annotation
        description = next((a for a in args[1:] if isinstance(a, str)), None)
        prop = {"type": _JSON_TYPE_BY_PY_NAME.get(getattr(field_type, "__name__", ""), "string")}
        if description:
            prop["description"] = description
        properties[field_name] = prop
        required.append(field_name)
    return {
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }

# Custom Mistral LLM client
class MistralLLM:
    def __init__(self, api_url: str, model: str, api_key: str, temperature: float = 0.7):
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.structured_output_enabled = False
        self.json_schema = None
        # Reused across calls (incl. concurrently, from the ThreadPoolExecutor in
        # main()) so requests pool and reuse TCP/TLS connections instead of
        # paying a fresh handshake for every single grading call.
        self.session = requests.Session()

    def _post(self, payload: dict, headers: dict) -> requests.Response:
        """POST to the chat completions endpoint, retrying transient failures
        (connection errors, timeouts, and 5xx server errors) with backoff. A
        transient failure isn't the same as a bad answer and shouldn't just
        propagate straight to a fake 0.0 score."""
        attempt = 0
        while True:
            try:
                response = self.session.post(
                    f"{self.api_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    verify=SSL_CERTIF if SSL_CERTIF else True,
                    timeout=LLM_REQUEST_TIMEOUT,
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt < LLM_REQUEST_RETRIES:
                    time.sleep(2 ** attempt)  # 1s, 2s, ...
                    attempt += 1
                    continue
                raise
            if response.status_code >= 500 and attempt < LLM_REQUEST_RETRIES:
                time.sleep(2 ** attempt)  # 1s, 2s, ...
                attempt += 1
                continue
            return response

    def invoke(self, messages: list) -> Any:
        """Call Mistral API and return response"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }

        # Add structured output if enabled. Prefer a real json_schema constraint
        # (forces the shape of the response) and fall back to the generic
        # json_object mode, which only guarantees valid JSON, not a given shape.
        used_json_schema = self.structured_output_enabled and self.json_schema is not None
        if self.structured_output_enabled:
            payload["response_format"] = (
                {"type": "json_schema", "json_schema": self.json_schema}
                if used_json_schema
                else {"type": "json_object"}
            )

        response = self._post(payload, headers)

        # Not every OpenAI-compatible gateway supports strict json_schema
        # response_format; some reject it outright with a 400. Fall back to the
        # more widely-supported generic json_object mode before giving up.
        if response.status_code == 400 and used_json_schema:
            payload["response_format"] = {"type": "json_object"}
            response = self._post(payload, headers)

        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Parse JSON if structured output was requested
        if self.structured_output_enabled:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
            # Some gateways accept response_format=json_schema with HTTP 200
            # but still reply with the JSON wrapped in a markdown code fence
            # instead of raw JSON. Strip ```json ... ``` / ``` ... ``` before
            # giving up, so a cosmetic wrapper doesn't get misread as a
            # missing "score" (a pipeline failure, not a real 0.0 grade).
            fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
            if fenced:
                try:
                    return json.loads(fenced.group(1))
                except json.JSONDecodeError:
                    pass
            # Some judge responses drop the opening quote on the "explanation"
            # value (e.g. `"explanation":The text...` instead of
            # `"explanation":"The text...`), which is otherwise valid JSON.
            # Patch that one known glitch and retry before giving up (task #144).
            repaired = re.sub(r'"explanation"\s*:\s*(?!")', '"explanation": "', content, count=1)
            if repaired != content:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass
            return {"content": content}

        # Return a simple object with content attribute for compatibility
        class Response:
            def __init__(self, content):
                self.content = content

        return Response(content)

    def with_structured_output(self, schema):
        """Enable structured output, constraining responses to `schema`'s shape."""
        self.structured_output_enabled = True
        self.json_schema = _typed_dict_to_json_schema(schema, schema.__name__)
        return self

def _run_structured_eval(llm: MistralLLM, instructions: str, content: str):
    """Run a structured evaluator LLM and extract the output.

    Returns: {score: float, explanation: str}

    A malformed grading response (missing/non-numeric "score") is treated as a
    pipeline error, not a real 0.0 — it's routed through the same except branch
    as network failures below, so it's never confused with a genuine low score
    assigned by the grader.
    """
    try:
        grade = llm.invoke([
            {"role": "system", "content": instructions},
            {"role": "user", "content": content},
        ])
        if not isinstance(grade, dict) or "score" not in grade:
            raise ValueError(f"Grading response did not contain a 'score' field: {grade!r}")
        score = float(grade["score"])
        explanation_text = grade.get("explanation", "")
        return {"score": score, "explanation": explanation_text}
    except Exception as e:
        return {"score": 0.0, "explanation": f"Grading error (not an actual score of 0): {e}"}

def _normalize_sources(raw_sources: list) -> list:
    """Match chatbot.py's _emit_json field mapping, so in-process results are
    shaped identically to what the subprocess path used to return via parsed
    JSON — full_content (untruncated) wins over the 200-char content preview
    that DocumentationChatbot.ask() also puts in 'content'."""
    return [
        {
            "title": s.get("title"),
            "module": s.get("module"),
            "doc_category": s.get("doc_category"),
            "doc_type": s.get("doc_type"),
            "url": s.get("url"),
            "content": s.get("full_content") or s.get("content"),
        }
        for s in raw_sources
    ]


_inprocess_chatbot = None
_inprocess_agentic_chatbot = None


def _init_inprocess_chatbot() -> None:
    """Load the chatbot core once for the whole evaluation run instead of once
    per question (main()'s dataset loop only — benchmark.py/optimizer.py still
    use _call_chatbot's subprocess path since they vary config per trial).

    Mirrors chatbot.py's own init sequence — same default config resolution,
    same skip_reranker rule for modes that don't use it — so answers are
    unchanged; only the per-question subprocess spawn + model reload goes away.
    No state carries between questions: ask()/ask_no_rag() are called fresh
    each time, exactly as before.
    """
    global _inprocess_chatbot, _inprocess_agentic_chatbot

    sys.path.insert(0, CHATBOT_DIR)
    from core import ChatbotConfig, DocumentationChatbot

    config_path = os.path.join(CHATBOT_DIR, "config.json")
    config = ChatbotConfig.load(config_path if os.path.exists(config_path) else None)

    # agentic mode never touches the reranker (its search_pages/read_page tools
    # bypass it) — same rule as chatbot.py's own skip_reranker.
    skip_reranker = EVAL_MODE in ("no-rag", "agentic")
    print(f"Loading chatbot once for the whole run (mode={EVAL_MODE})...")
    _inprocess_chatbot = DocumentationChatbot(config, skip_reranker=skip_reranker)

    if EVAL_MODE == "agentic":
        from core import AgenticChatbot
        _inprocess_agentic_chatbot = AgenticChatbot(
            config,
            vectorstore=_inprocess_chatbot.vectorstore,
            bm25_index=_inprocess_chatbot.bm25_index,
        )
    print("Chatbot ready — reused for every question in this run.")


def _ask_chatbot_inprocess(question: str, timeout_seconds: Optional[int] = None) -> dict:
    """In-process replacement for _call_chatbot, used only by main()'s dataset
    run once _init_inprocess_chatbot() has loaded the model.

    Runs the call in a thread with a soft join(timeout): unlike the subprocess
    path (which can hard-kill a hung process tree) this can't force-kill a
    stuck call — a timed-out question is reported as such and its thread is
    left to finish or die on its own in the background.
    """
    start_time = time.perf_counter()
    outcome: Dict[str, Any] = {}

    def run():
        try:
            if EVAL_MODE == "agentic":
                outcome["result"] = _inprocess_agentic_chatbot.ask(question)
            elif EVAL_MODE == "no-rag":
                outcome["result"] = _inprocess_chatbot.ask_no_rag(question)
            else:
                outcome["result"] = _inprocess_chatbot.ask(question)
        except Exception as e:
            outcome["error"] = str(e)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    elapsed = time.perf_counter() - start_time

    if thread.is_alive():
        return {"answer": "Chatbot request timed out", "error": "timeout", "documents": [], "request_time": elapsed}
    if "error" in outcome:
        return {"answer": f"Error: {outcome['error']}", "error": outcome["error"], "documents": [], "request_time": elapsed}

    result = outcome["result"]
    answer = result.get("answer")
    error = result.get("error")
    return {
        # keep a real None answer as None so downstream can render the error;
        # only substitute the sentinel when the key is genuinely absent.
        "answer": answer if ("answer" in result or answer is not None) else "No answer returned",
        "error": error,
        "documents": _build_documents(_normalize_sources(result.get("sources", []))),
        "request_time": elapsed,
        "filters": result.get("filters", {}),
    }


def _build_documents(sources: list) -> list:
    """Convert raw source dicts into document-like objects."""
    documents = []
    for source in sources:
        class Document:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata

        doc = Document(
            content=source.get("content", ""),
            metadata={
                "title": source.get("title", "Unknown"),
                "module": source.get("module", "Unknown"),
                "doc_category": source.get("doc_category", "Unknown"),
                "doc_type": source.get("doc_type", "Unknown"),
                "url": source.get("url", "")
            }
        )
        documents.append(doc)
    return documents

def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill proc and any descendants it may have spawned, not just its own PID.

    proc is started with start_new_session=True (its own process group), so
    killing that whole group reaches anything it forked off — unlike proc.kill(),
    which only signals proc itself and would leave descendants as orphans still
    holding memory/CPU (or a lock on the shared Chroma persist_directory).
    """
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass  # already exited on its own

def _call_chatbot(question: str, mode: str = None, timeout_seconds: int = None,
                   config: Optional[Dict[str, Any]] = None,
                   chatbot_config_path: Optional[str] = None) -> dict:
    """Call chatbot.py and return parsed answer/documents.

    `config` carries hyperparameter overrides and is translated into the
    matching chatbot.py CLI flags:
    - both modes: temperature, max_tokens, model, base_url, api_key
    - rag mode: k, k_standard, k_deep_dive, reranker_enabled, top_n, deep_dive,
      hyde_enabled, bm25_enabled, title_boost_enabled, k_retrieve,
      deep_dive_batch_size, expansion_char_budget, reranker_model, reranker_type
    - agentic mode: max_steps, max_chars_per_page

    `chatbot_config_path`, if given, is passed as chatbot.py's --config — used
    to point at an extraction-parameter variant's own chatbot config (matching
    chromadb_path/embedding model), see evaluator/base/extraction_cache.py.
    Hyperparameter flags above still layer on top of it normally.
    """
    start_time = time.perf_counter()
    cmd = [sys.executable, "chatbot.py", "--question", question, "--json-output"]
    if chatbot_config_path:
        cmd.extend(["--config", chatbot_config_path])
    if mode:
        cmd.extend(["--mode", mode])

    if config:
        if config.get("k") is not None:
            cmd.extend(["--k", str(config["k"])])
        if config.get("k_standard") is not None:
            cmd.extend(["--k-standard", str(config["k_standard"])])
        if config.get("k_deep_dive") is not None:
            cmd.extend(["--k-deep-dive", str(config["k_deep_dive"])])
        if config.get("temperature") is not None:
            cmd.extend(["--temperature", str(config["temperature"])])
        if config.get("max_tokens") is not None:
            cmd.extend(["--max-tokens", str(config["max_tokens"])])
        if config.get("model") is not None:
            cmd.extend(["--model", str(config["model"])])
        if config.get("base_url") is not None:
            cmd.extend(["--base-url", str(config["base_url"])])
        if config.get("api_key") is not None:
            cmd.extend(["--api-key", str(config["api_key"])])
        if config.get("reranker_model") is not None:
            cmd.extend(["--reranker-model", str(config["reranker_model"])])
        if config.get("reranker_type") is not None:
            cmd.extend(["--reranker-type", str(config["reranker_type"])])
        if config.get("top_n") is not None:
            cmd.extend(["--top-n", str(config["top_n"])])
        if config.get("reranker_enabled") is False:
            cmd.append("--no-rerank")
        if config.get("deep_dive"):
            cmd.append("--deep-dive")
        if config.get("hyde_enabled") is False:
            cmd.append("--no-hyde")
        if config.get("bm25_enabled") is False:
            cmd.append("--no-bm25")
        if config.get("title_boost_enabled") is False:
            cmd.append("--no-title-boost")
        if config.get("k_retrieve") is not None:
            cmd.extend(["--k-retrieve", str(config["k_retrieve"])])
        if config.get("deep_dive_batch_size") is not None:
            cmd.extend(["--deep-dive-batch-size", str(config["deep_dive_batch_size"])])
        if config.get("expansion_char_budget") is not None:
            cmd.extend(["--expansion-char-budget", str(config["expansion_char_budget"])])
        if config.get("max_steps") is not None:
            cmd.extend(["--max-steps", str(config["max_steps"])])
        if config.get("max_chars_per_page") is not None:
            cmd.extend(["--max-chars-per-page", str(config["max_chars_per_page"])])

    try:
        # Popen (rather than subprocess.run) so a timeout can be handled with
        # _kill_process_tree below: run()'s own timeout handling only kills the
        # immediate chatbot.py PID, which would leave any processes it spawned
        # still running — burning memory/CPU and potentially still holding a lock
        # on the shared Chroma persist_directory, which could then stall other
        # chatbot calls too.
        proc = subprocess.Popen(
            cmd,
            cwd=CHATBOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            proc.communicate()  # drain pipes / reap the now-dead process
            raise

        elapsed = time.perf_counter() - start_time
        if returncode != 0:
            return {
                "answer": f"Error calling chatbot: {stderr}",
                "documents": [],
                "request_time": elapsed,
            }

        stdout = stdout.strip()
        json_start = stdout.find('{')
        if json_start < 0:
            raise ValueError("No JSON found in chatbot output")

        json_str = stdout[json_start:]
        json_end = json_str.rfind('}')
        if json_end < 0:
            raise ValueError("Invalid JSON format")

        chatbot_result = json.loads(json_str[:json_end + 1])
        answer = chatbot_result.get("answer", "No answer returned")
        sources = chatbot_result.get("sources", [])
        return {
            "answer": answer,
            "documents": _build_documents(sources),
            "request_time": elapsed,
            "filters": chatbot_result.get("filters", {}),
        }

    except (json.JSONDecodeError, ValueError) as e:
        elapsed = time.perf_counter() - start_time
        return {
            "answer": f"Error parsing chatbot response: {str(e)}",
            "documents": [],
            "request_time": elapsed,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start_time
        return {
            "answer": "Chatbot request timed out",
            "documents": [],
            "request_time": elapsed,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return {
            "answer": f"Error: {str(e)}",
            "documents": [],
            "request_time": elapsed,
        }

def run_evaluation(question: str, answer_dict: Dict, reference_answer: str,
                    executor: Optional[ThreadPoolExecutor] = None,
                    metrics: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Run the given evaluators (default: all of EVAL_METRICS) on a single answer.

    `metrics` lets a caller skip metrics that are meaningless for a given
    answer — e.g. no-rag mode has no retrieved documents, so grading
    groundedness/retrieval_relevance would just judge "no documents" every
    time, at the cost of two judge calls per question for nothing.
    """
    metrics = metrics if metrics is not None else EVAL_METRICS
    evaluations = {}
    inputs = {"question": question}

    def _eval_metric(metric_name: str) -> Dict[str, Any]:
        try:
            if metric_name == "correctness":
                result = correctness(inputs, answer_dict, {"answer": reference_answer})
            elif metric_name == "relevance":
                result = relevance(inputs, answer_dict)
            elif metric_name == "groundedness":
                result = groundedness(inputs, answer_dict)
            elif metric_name == "retrieval_relevance":
                result = retrieval_relevance(inputs, answer_dict)
            else:
                return {"score": 0.0, "explanation": f"Unknown metric: {metric_name}"}

            return _sanitize_eval_result(result)
        except Exception as e:
            return {"score": 0.0, "explanation": f"Error: {str(e)}"}

    if executor is not None:
        futures = {
            executor.submit(_eval_metric, metric): metric
            for metric in metrics
        }
        raw_results = {}
        for future in as_completed(futures):
            metric_name = futures[future]
            raw_results[metric_name] = future.result()
        for metric in metrics:
            evaluations[metric] = raw_results.get(metric, {"score": 0.0, "explanation": "No result returned."})
    else:
        for metric in metrics:
            evaluations[metric] = _eval_metric(metric)

    return evaluations

############################################################################################
################################## Shared grading LLM #######################################
############################################################################################

# All four metrics below grade on the same {score, explanation} shape, so they
# share a single schema and a single Mistral client instead of four identical
# ones — the prompts (system instructions) are what actually differ per metric.
class Grade(TypedDict):
    explanation: Annotated[str, ..., "A single paragraph in English justifying the score based on the criteria"]
    score: Annotated[float, ..., "A number between 0 and 10"]

grader_llm = MistralLLM(
    api_url=LLM_API_URL,
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    temperature=0
).with_structured_output(Grade)

############################################################################################
####################################### Correctness ########################################
############################################################################################

# Grade prompt
correctness_instructions = """You are an expert evaluator grading the factual accuracy of an ANSWER produced by a RAG chatbot, by comparing it against a REFERENCE ANSWER. You will be given a QUESTION, the REFERENCE ANSWER, and the ANSWER TO GRADE.

Evaluate ONLY factual accuracy relative to the REFERENCE ANSWER:

(1) Grade based solely on factual accuracy — do not penalize differences in wording, structure, tone, or level of detail.
(2) Flag any internal contradiction within the ANSWER TO GRADE.
(3) Additional information beyond the REFERENCE ANSWER is acceptable, provided it is factually accurate and does not conflict with the REFERENCE ANSWER.

Use this rubric strictly. Pick the anchor whose description best matches the ANSWER TO GRADE; use an intermediate value (e.g. 7, 3) when it falls between two anchors:

- 10: Every factual claim matches the REFERENCE ANSWER. No contradictions, no invented facts.
- 8: All essential facts are correct; at most one minor, non-critical imprecision or omission.
- 6: The core answer is correct but is missing a fact a user would consider important, or contains one minor inaccuracy.
- 4: Mixed accuracy — some correct facts are present, but at least one significant claim is wrong or missing, materially changing the answer.
- 2: Mostly incorrect — only marginal or tangential facts are correct; the main claim is wrong or absent.
- 0: Entirely incorrect, contradicts the REFERENCE ANSWER, or does not answer the question at all.

Before assigning a score, identify the specific facts in the ANSWER TO GRADE that match or conflict with the REFERENCE ANSWER — your explanation should cite them.

Your output MUST be in JSON format with only two keys:

- "score": a number between 0 and 10 (use intermediate values when the answer falls between two rubric anchors)
- "explanation": a single paragraph written in English, citing the specific facts that justified the score."""

# Evaluator
def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Score the answer against the reference output."""
    answers = (
        f"QUESTION: {inputs['question']}\n"
        f"REFERENCE ANSWER: {reference_outputs['answer']}\n"
        f"ANSWER TO GRADE: {outputs['answer']}"
    )
    return _run_structured_eval(grader_llm, correctness_instructions, answers)

############################################################################################
######################################## Relevance #########################################
############################################################################################

# Grade prompt
relevance_instructions = """You are an expert evaluator grading whether an ANSWER directly and efficiently addresses a QUESTION, independent of whether the answer is factually correct. You will be given a QUESTION and an ANSWER.

Judge only:

(1) Does the ANSWER address what was actually asked?
(2) Is it free of irrelevant padding, off-topic content, or an unjustified refusal to answer?

Use this rubric strictly. Pick the anchor whose description best matches the ANSWER; use an intermediate value when it falls between two anchors:

- 10: Directly and completely addresses the question; no irrelevant content.
- 8: Addresses the question well, with only minor irrelevant or redundant content.
- 6: Partially addresses the question — covers part of what was asked but misses or sidesteps another part, or includes a notable amount of unrelated content.
- 4: Loosely related — touches the general topic but does not actually answer what was asked.
- 2: Barely related — connects only tangentially to the topic of the question.
- 0: Off-topic, non-responsive, or refuses to answer without justification.

Your output MUST be in JSON format with only two keys:

- "score": a number between 0 and 10 (use intermediate values when the answer falls between two rubric anchors)
- "explanation": a single paragraph written in English, indicating which parts of the answer support the score."""

# Evaluator
def relevance(inputs: dict, outputs: dict) -> dict:
    """Score whether the answer is relevant to the question."""
    answers = (
        f"QUESTION: {inputs['question']}\n"
        f"ANSWER: {outputs['answer']}"
    )
    return _run_structured_eval(grader_llm, relevance_instructions, answers)

############################################################################################
######################################## Groundedness ######################################
############################################################################################

# Grade prompt
grounded_instructions = """You are an expert evaluator checking whether an ANSWER is supported by a given set of FACTS (retrieved documents), independent of whether the ANSWER is correct or relevant to the original question. You will be given FACTS and an ANSWER.

For each claim in the ANSWER, check whether it is stated or directly implied by the FACTS. Do not judge whether the FACTS are sufficient to fully answer the question, or whether the ANSWER is a good answer — only whether every claim in it traces back to the FACTS.

Use this rubric strictly. Pick the anchor whose description best matches the ANSWER; use an intermediate value when it falls between two anchors:

- 10: Every claim in the ANSWER is directly supported by the FACTS. No invented or extrapolated information.
- 8: Nearly all claims are supported; at most one minor, low-impact detail is not traceable to the FACTS.
- 6: Mostly grounded, but contains one claim of moderate importance that is not supported by the FACTS.
- 4: Mixed — a significant portion of the ANSWER is not supported by the FACTS, alongside some grounded content.
- 2: Mostly unsupported — only marginal parts of the ANSWER trace back to the FACTS.
- 0: Not grounded at all, or actively contradicts the FACTS.

Your output MUST be in JSON format with only two keys:

- "score": a number between 0 and 10 (use intermediate values when the answer falls between two rubric anchors)
- "explanation": a single paragraph written in English, identifying the specific claim(s) that are or are not supported by the FACTS."""

# Evaluator
def groundedness(_inputs: dict, outputs: dict) -> dict:
    """Score whether the answer is grounded in the provided documents."""
    documents = outputs.get("documents") or []
    if not documents:
        return {"score": 0.0, "explanation": "No documents were provided to check groundedness against."}

    doc_string = "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in documents)
    answers = (
        f"FACTS: {doc_string}\n"
        f"ANSWER: {outputs['answer']}"
    )
    return _run_structured_eval(grader_llm, grounded_instructions, answers)

############################################################################################
#################################### Retrieval Relevance ###################################
############################################################################################

# Grade prompt
retrieval_relevance_instructions = """You are an expert evaluator judging whether a set of retrieved DOCUMENTS is useful for answering a QUESTION. You will be given a QUESTION and the DOCUMENTS retrieved for it.

For each document, judge whether it contains information that would actually help answer the QUESTION — sharing a keyword or general topic with the QUESTION is NOT enough on its own if the document does not provide information useful to answering it. Then judge the set as a whole based on the proportion of documents that are genuinely useful.

Use this rubric strictly. Pick the anchor whose description best matches the DOCUMENTS as a whole; use an intermediate value when it falls between two anchors:

- 10: All (or nearly all) documents directly help answer the question.
- 8: A clear majority of documents are useful; one or two are tangential or unhelpful.
- 6: About half the documents are useful; the rest are off-topic or only superficially related.
- 4: A minority of documents are useful; most only share a topic or keyword with the question without providing information that helps answer it.
- 2: At most one document is marginally useful; the rest are unrelated.
- 0: None of the documents help answer the question.

Your output MUST be in JSON format with only two keys:

- "score": a number between 0 and 10 (use intermediate values when the set falls between two rubric anchors)
- "explanation": a single paragraph written in English, indicating which documents (by title, if available) were or were not useful."""

# Evaluator
def retrieval_relevance(inputs: dict, outputs: dict) -> dict:
    """Score whether provided documents are relevant to the question."""
    documents = outputs.get("documents") or []
    if not documents:
        return {"score": 0.0, "explanation": "No documents were retrieved for this question."}

    doc_string = "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in documents)
    answers = (
        f"DOCUMENTS: {doc_string}\n"
        f"QUESTION: {inputs['question']}"
    )
    return _run_structured_eval(grader_llm, retrieval_relevance_instructions, answers)

############################################################################################
###################################### Run on dataset ######################################
############################################################################################

# Load the examples for the dataset from JSON file
dataset_file = Path(__file__).resolve().parent / "dataset.json"

def _load_examples():
    with open(dataset_file, "r", encoding="utf-8") as f:
        return json.load(f)

def _sanitize_eval_result(ev):
    """Extract score and explanation from evaluation result, always preserving LLM explanations."""
    if not isinstance(ev, dict):
        return {"score": float(ev) if isinstance(ev, (int, float)) else 0.0}
    
    # Extract score (should already be 0-10 from Mistral)
    score_val = ev.get("score", 0)
    score = float(score_val) if isinstance(score_val, (int, float)) else 0.0
    out = {"score": score}
    out["explanation"] = ev.get("explanation") or ""
    return out

EVAL_METRICS = [
    "correctness",
    "relevance",
    "groundedness",
    "retrieval_relevance",
]

def _print_example_result(display_index: int, result: dict) -> None:
    print(f"\n{'=' * 80}")
    print(f"Example {display_index}:")
    print(f"{'=' * 80}")
    print(f"\nQuestion: {result['question']}\n")
    print(f"Expected Answer: {result['expected_answer']}\n")
    rag_answer = result['rag_answer'] or f"[no answer — error: {result.get('error') or 'unknown'}]"
    rag_answer_preview = rag_answer[:100] + "..." if len(rag_answer) > 100 else rag_answer
    print(f"RAG Answer (preview): {rag_answer_preview}\n")
    print("-" * 80)
    print("EVALUATION SCORES:")
    print("-" * 80)
    for metric, eval_result in result['evaluations'].items():
        print(f"  {metric.replace('_', ' ').title()}: {eval_result.get('score', 0):.1f}")
    print("-" * 80)

def main(num_workers: int = 1, limit_questions: int = None, timeout_seconds: int = None):
    _validate_config()
    _init_inprocess_chatbot()

    print("Testing Chatbot...")
    print("=" * 80)

    examples = _load_examples()
    if limit_questions is not None:
        examples = examples[:limit_questions]
    total = len(examples)
    results = []

    # print(..., flush=True) here so progress is visible live even when stdout is
    # piped through something that fully buffers non-tty output (e.g. `| tee log`),
    # instead of only appearing once the whole run finishes.
    def fetch_chatbot(example_index: int, example: dict) -> tuple[int, dict, dict]:
        q = example['inputs']['question']
        print(f"[{example_index}/{total}] Calling chatbot: {q[:80]}{'...' if len(q) > 80 else ''}", flush=True)
        start = time.perf_counter()
        output = _ask_chatbot_inprocess(q, timeout_seconds)
        elapsed = time.perf_counter() - start
        # Grading starts immediately only in parallel mode (grade futures are submitted
        # as each answer future completes). In sequential mode (num_workers=1, the
        # default) fetch_chatbot runs for ALL questions before evaluate_example runs for
        # any — so "— grading..." there would be a lie about what happens next.
        suffix = " — grading..." if num_workers > 1 else ""
        print(f"[{example_index}/{total}] Chatbot answered in {elapsed:.1f}s{suffix}", flush=True)
        return example_index, example, output

    def evaluate_example(example_index: int, example: dict, output: dict) -> dict:
        q = example['inputs']['question']
        expected = example['outputs']['answer']
        grade_start = time.perf_counter()
        evaluations = run_evaluation(q, output, expected, metrics=EVAL_METRICS_FOR_MODE)
        grade_elapsed = time.perf_counter() - grade_start
        print(f"[{example_index}/{total}] Graded in {grade_elapsed:.1f}s", flush=True)
        request_time = output.get("request_time", 0.0) if isinstance(output, dict) else 0.0
        rag_answer = output.get('answer') if isinstance(output, dict) else str(output)
        error = output.get("error") if isinstance(output, dict) else None
        documents = output.get("documents", [])
        filters = output.get("filters", {}) if isinstance(output, dict) else {}
        # Serialize documents for JSON
        serialized_documents = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]
        return {
            "question": q,
            "expected_answer": expected,
            "rag_answer": rag_answer,
            "error": error,
            "request_time": request_time,
            "evaluations": evaluations,
            "documents": serialized_documents,
            "index": example_index,
            # None for modes/endpoints that don't report them (e.g. an LLM backend
            # that never returns usage_metadata) rather than omitted, so downstream
            # aggregation can tell "no data" from "zero".
            "steps_used": filters.get("steps_used"),
            "token_usage": filters.get("token_usage"),
        }

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            chatbot_futures = {
                executor.submit(fetch_chatbot, i, example): (i, example)
                for i, example in enumerate(examples, 1)
            }
            eval_futures = {}

            for future in as_completed(chatbot_futures):
                _, _, output = future.result()
                idx, example = chatbot_futures[future]
                eval_future = executor.submit(evaluate_example, idx, example, output)
                eval_futures[eval_future] = idx

            for future in as_completed(eval_futures):
                result = future.result()
                results.append(result)
                _print_example_result(len(results), result)
    else:
        print(f"Fetching all {total} answers first, then grading — expect grading progress "
              f"only after the [{total}/{total}] answer lands.", flush=True)
        chatbot_results = [fetch_chatbot(i, example) for i, example in enumerate(examples, 1)]
        for idx, example, output in chatbot_results:
            result = evaluate_example(idx, example, output)
            results.append(result)
            _print_example_result(idx, result)

    # In parallel mode, results arrive in completion order rather than dataset
    # order; restore dataset order so the saved file and the printed summary
    # are stable and comparable across runs.
    results.sort(key=lambda r: r["index"])

    # Calculate average scores, only over metrics that were actually graded
    # this run (EVAL_METRICS_FOR_MODE — e.g. no-rag skips groundedness/
    # retrieval_relevance since there are no retrieved documents to grade).
    summary = None
    if results:
        graded_metrics = EVAL_METRICS_FOR_MODE if EVAL_METRICS_FOR_MODE is not None else EVAL_METRICS
        avg_by_metric = {
            metric: sum(r["evaluations"][metric].get("score", 0.0) for r in results) / len(results)
            for metric in graded_metrics
        }
        avg_request_time = sum(r.get("request_time", 0.0) for r in results) / len(results)
        avg_overall = sum(avg_by_metric.values()) / len(avg_by_metric)

        summary = {
            "total_examples": len(results),
            **avg_by_metric,
            "overall_average": avg_overall,
            "average_request_time_seconds": avg_request_time,
        }

        # Only agentic mode reports these (rag mode's filters is currently
        # empty) — skip the averages entirely rather than average in a bunch
        # of Nones as zeros.
        steps_values = [r["steps_used"] for r in results if r.get("steps_used") is not None]
        if steps_values:
            summary["average_steps_used"] = sum(steps_values) / len(steps_values)

        token_totals = [r["token_usage"]["total_tokens"] for r in results if r.get("token_usage")]
        if token_totals:
            summary["average_total_tokens"] = sum(token_totals) / len(token_totals)

    # Save results, with the summary first so it's readable at the top of the file
    results_file = "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as rf:
        json.dump({"summary": summary, "results": results}, rf, ensure_ascii=False, indent=2)

    print(f"\nEvaluation complete. Results saved to {results_file}")

    # Display average scores
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY - AVERAGE SCORES")
    print("=" * 80)

    if summary:
        print(f"Total examples evaluated: {summary['total_examples']}\n")
        print(f"Correctness:         {summary['correctness']:.1f}/10")
        print(f"Relevance:           {summary['relevance']:.1f}/10")
        if "groundedness" in summary:
            print(f"Groundedness:        {summary['groundedness']:.1f}/10")
        if "retrieval_relevance" in summary:
            print(f"Retrieval Relevance: {summary['retrieval_relevance']:.1f}/10")
        print(f"Average Chatbot Request Time: {summary['average_request_time_seconds']:.3f} seconds")
        if "average_steps_used" in summary:
            print(f"Average Steps Used:  {summary['average_steps_used']:.1f}")
        if "average_total_tokens" in summary:
            print(f"Average Total Tokens: {summary['average_total_tokens']:.0f}")
        print(f"\nOverall Average:     {summary['overall_average']:.1f}/10")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate chatbot responses")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions to test")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker threads for parallel evaluation (use -1 for all CPUs)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout in seconds for chatbot requests (default: no timeout)")
    args = parser.parse_args()
    workers = args.workers
    if workers == -1:
        workers = os.cpu_count() or 1
    elif workers < -1:
        parser.error("--workers must be -1 or a positive integer")
    main(workers, args.limit, args.timeout)
    print("=" * 80)
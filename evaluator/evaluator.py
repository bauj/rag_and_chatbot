import subprocess
import sys
import json
import os
from pathlib import Path

from typing_extensions import Annotated, TypedDict
import requests
from typing import Any

# Load configuration from evaluator/config.json with fallback on environment variables
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

evaluator_options = evaluator_config.get("evaluator_options", {})

AGENTIC_MODE = evaluator_options.get("agentic_mode", evaluator_options.get("agentic_mode"))

def traceable():
    """No-op decorator placeholder (LangSmith tracing removed)."""
    def decorator(func):
        return func
    return decorator

# Custom Mistral LLM client
class MistralLLM:
    def __init__(self, api_url: str, model: str, api_key: str, temperature: float = 0.7):
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.structured_output_enabled = False
    
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
        
        # Add structured output if enabled
        if self.structured_output_enabled:
            payload["response_format"] = {"type": "json_object"}
        
        response = requests.post(
            f"{self.api_url}/chat/completions",
            headers=headers,
            json=payload,
            verify=SSL_CERTIF if SSL_CERTIF else True
        )
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Parse JSON if structured output was requested
        if self.structured_output_enabled:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"content": content}
        
        # Return a simple object with content attribute for compatibility
        class Response:
            def __init__(self, content):
                self.content = content
        
        return Response(content)
    
    def with_structured_output(self, schema, method="json_schema", strict=True):
        """Enable structured output and return self for method chaining"""
        self.structured_output_enabled = True
        return self


def _extract_bool_from_grade(grade: dict, keys: list) -> bool:
    """Try multiple possible keys in the judge output to determine boolean score."""
    if not isinstance(grade, dict):
        return False
    for k in keys:
        if k in grade:
            v = grade[k]
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                lv = v.strip().lower()
                if lv in ("true", "yes", "1"):
                    return True
                if lv in ("false", "no", "0"):
                    return False
    # try nested common wrappers
    for wrapper in ("result", "data", "value"):
        if wrapper in grade and isinstance(grade[wrapper], dict):
            for k in keys:
                if k in grade[wrapper]:
                    v = grade[wrapper][k]
                    if isinstance(v, bool):
                        return v
                    if isinstance(v, (int, float)):
                        return bool(v)
                    if isinstance(v, str):
                        lv = v.strip().lower()
                        if lv in ("true", "yes", "1"):
                            return True
                        if lv in ("false", "no", "0"):
                            return False
    return False


def _run_structured_eval(llm: MistralLLM, instructions: str, content: str, score_keys: list):
    """Run a structured evaluator LLM and normalize the output.

    Returns: {score: bool, explanation: str, explanation_raw: Any}
    """
    try:
        grade = llm.invoke([
            {"role": "system", "content": instructions},
            {"role": "user", "content": content},
        ])
        explanation_text = ""
        explanation_raw = None
        if isinstance(grade, dict):
            explanation_raw = grade
            score = _extract_bool_from_grade(grade, score_keys)
            explanation_text = grade.get("explanation") or grade.get("content") or ""
            if not explanation_text and "reasoning" in grade:
                explanation_text = grade.get("reasoning")
        else:
            score = False
            explanation_text = getattr(grade, "content", "") or ""
            explanation_raw = {"content": explanation_text}
        return {"score": score, "explanation": explanation_text, "explanation_raw": explanation_raw}
    except Exception as e:
        return {"score": False, "explanation": str(e), "explanation_raw": None}


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Evaluator for RAG answer accuracy (normalized return)."""
    answers = f"QUESTION: {inputs['question']}\nGROUND TRUTH ANSWER: {reference_outputs['answer']}\nSTUDENT ANSWER: {outputs['answer']}"
    return _run_structured_eval(grader_llm, correctness_instructions, answers, ["correct", "correctness", "result"])

# Add decorator so this function is traced in LangSmith (no-op)
@traceable()
def rag_bot(question: str) -> dict:
    """Call the external chatbot to answer the question and retrieve documents"""
    try:
        # Call the chatbot.py script with JSON output mode
        # Redirect stderr to suppress debug prints from chatbot initialization
        
        if AGENTIC_MODE:
            result = subprocess.run(
                [sys.executable, "chatbot.py", "--mode", "agentic", "--question", question, "--json-output"],
                cwd=CHATBOT_DIR,
                capture_output=True,
                text=True,
                timeout=60
            )
        else:
            result = subprocess.run(
                [sys.executable, "chatbot.py", "--question", question, "--json-output"],
                cwd=CHATBOT_DIR,
                capture_output=True,
                text=True,
                timeout=60
            )
        
        if result.returncode != 0:
            return {
                "answer": f"Error calling chatbot: {result.stderr}",
                "documents": []
            }
        
        try:
            # Parse JSON response - try to extract JSON from stdout
            # which may contain debug prints before the JSON
            stdout = result.stdout.strip()
            # Find the start of JSON (starts with '{')
            json_start = stdout.find('{')
            if json_start >= 0:
                # Extract JSON from the line containing the start
                json_str = stdout[json_start:]
                # Find the end of the JSON object by finding the last '}'
                json_end = json_str.rfind('}')
                if json_end >= 0:
                    json_str = json_str[:json_end + 1]
                    chatbot_result = json.loads(json_str)
                else:
                    raise ValueError("Invalid JSON format")
            else:
                raise ValueError("No JSON found in chatbot output")
            
            answer = chatbot_result.get("answer", "No answer returned")
            sources = chatbot_result.get("sources", [])
            
            # Convert sources to document objects with page_content attribute
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
            
            return {
                "answer": answer,
                "documents": documents
            }
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback if JSON parsing fails
            return {
                "answer": f"Error parsing chatbot response: {str(e)}",
                "documents": []
            }
    except subprocess.TimeoutExpired:
        return {
            "answer": "Chatbot request timed out",
            "documents": []
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "documents": []
        }

# Load the examples for the dataset from JSON file
dataset_file = "dataset.json"
with open(dataset_file, "r", encoding="utf-8") as f:
    examples = json.load(f)

# Grade output schema
class CorrectnessGrade(TypedDict):
    # Note that the order in the fields are defined is the order in which the model will generate them.
    # It is useful to put explanations before responses because it forces the model to think through
    # its final response before generating it:
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

# Grade prompt
correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# Grader LLM
grader_llm = MistralLLM(api_url=LLM_API_URL, model=LLM_MODEL, api_key=LLM_API_KEY, temperature=0).with_structured_output(CorrectnessGrade, method="json_schema", strict=True)


# Grade output schema
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]

# Grade prompt
relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# Grader LLM
relevance_llm = MistralLLM(
    api_url=LLM_API_URL,
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    temperature=0
).with_structured_output(
    RelevanceGrade, method="json_schema", strict=True
)

# Evaluator
def relevance(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer helpfulness."""
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    return _run_structured_eval(relevance_llm, relevance_instructions, answer, ["relevant", "relevance", "result"])

# Grade output schema
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]

# Grade prompt
grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# Grader LLM
grounded_llm = MistralLLM(
    api_url=LLM_API_URL,
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    temperature=0
).with_structured_output(
    GroundedGrade, method="json_schema", strict=True
)

# Evaluator
def groundedness(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer groundedness."""
    docs = outputs.get("documents") or []
    doc_string = "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    return _run_structured_eval(grounded_llm, grounded_instructions, answer, ["grounded", "groundedness", "result"])

# Grade output schema
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]

# Grade prompt
retrieval_relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# Grader LLM
retrieval_relevance_llm = MistralLLM(
    api_url=LLM_API_URL,
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    temperature=0
).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)

def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """An evaluator for document relevance"""
    docs = outputs.get("documents") or []
    doc_string = "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)
    answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
    # Run evaluator
    return _run_structured_eval(retrieval_relevance_llm, retrieval_relevance_instructions, answer, ["relevant", "relevance", "result"])

def target(inputs: dict) -> dict:
    return rag_bot(inputs["question"])

# Test the RAG system with examples
print("Testing RAG system...")
print("=" * 80)

results = []
def _sanitize_eval_result(ev, include_raw_if_empty: bool = False):
    if not isinstance(ev, dict):
        return {"score": bool(ev)}
    out = {"score": bool(ev.get("score", False))}
    expl = ev.get("explanation") or ""
    if expl:
        out["explanation"] = expl
        return out

    if include_raw_if_empty:
        raw = ev.get("explanation_raw")
        if raw:
            # try to extract readable text from common keys
            if isinstance(raw, dict):
                for k in ("explanation", "reasoning", "REASONING", "reason", "content", "text"):
                    if k in raw and raw[k]:
                        val = raw[k]
                        if isinstance(val, list):
                            out["explanation"] = " \n".join(str(x) for x in val)
                        else:
                            out["explanation"] = str(val)
                        return out
                # try to find any list of strings in values
                for v in raw.values():
                    if isinstance(v, list) and all(isinstance(x, str) for x in v):
                        out["explanation"] = " \n".join(v)
                        return out
            # fallback to JSON dump
            try:
                out["explanation"] = json.dumps(raw, ensure_ascii=False)
                return out
            except Exception:
                pass

    return out

for i, example in enumerate(examples, 1):
    print(f"\n{'=' * 80}")
    print(f"Example {i}:")
    print(f"{'=' * 80}")
    q = example['inputs']['question']
    expected = example['outputs']['answer']
    print(f"\nQuestion: {q}\n")
    print(f"Expected Answer: {expected}\n")

    # Get RAG response
    output = target(example["inputs"])
    rag_answer = output.get('answer') if isinstance(output, dict) else str(output)
    # Display only the first 100 characters of RAG answer to keep terminal readable
    rag_answer_preview = rag_answer[:100] + "..." if len(rag_answer) > 100 else rag_answer
    print(f"RAG Answer (preview): {rag_answer_preview}\n")
    print("-" * 80)

    # Run evaluators and collect structured results
    print("EVALUATION SCORES:")
    print("-" * 80)
    try:
        correct = correctness(example["inputs"], output, example["outputs"])
        print(f"  Correctness:         {correct['score']}")
    except Exception as e:
        correct = {"score": False, "explanation": str(e)}
        print(f"  Correctness eval error: {e}")

    try:
        relevant = relevance(example["inputs"], output)
        print(f"  Relevance:           {relevant['score']}")
    except Exception as e:
        relevant = {"score": False, "explanation": str(e)}
        print(f"  Relevance eval error: {e}")

    try:
        grounded = groundedness(example["inputs"], output)
        print(f"  Groundedness:        {grounded['score']}")
    except Exception as e:
        grounded = {"score": False, "explanation": str(e)}
        print(f"  Groundedness eval error: {e}")

    try:
        retrieval_rel = retrieval_relevance(example["inputs"], output)
        print(f"  Retrieval Relevance: {retrieval_rel['score']}")
    except Exception as e:
        retrieval_rel = {"score": False, "explanation": str(e)}
        print(f"  Retrieval Relevance eval error: {e}")

    # Append structured result (sanitize to remove explanation_raw and empty explanations)
    results.append({
        "question": q,
        "expected_answer": expected,
        "rag_answer": rag_answer,
        "evaluations": {
            "correctness": _sanitize_eval_result(correct),
            "relevance": _sanitize_eval_result(relevant),
            "groundedness": _sanitize_eval_result(grounded),
            "retrieval_relevance": _sanitize_eval_result(retrieval_rel, include_raw_if_empty=True),
        }
    })

    print("-" * 80)

# Save results to JSON
results_file = "evaluation_results.json"
with open(results_file, "w", encoding="utf-8") as rf:
    json.dump(results, rf, ensure_ascii=False, indent=2)

print(f"Saved evaluation results to {results_file}")

# Calculate and display average scores
print("\n" + "=" * 80)
print("EVALUATION SUMMARY - AVERAGE SCORES")
print("=" * 80)

if results:
    avg_correctness = sum(1 for r in results if r["evaluations"]["correctness"].get("score", False)) / len(results)
    avg_relevance = sum(1 for r in results if r["evaluations"]["relevance"].get("score", False)) / len(results)
    avg_groundedness = sum(1 for r in results if r["evaluations"]["groundedness"].get("score", False)) / len(results)
    avg_retrieval_relevance = sum(1 for r in results if r["evaluations"]["retrieval_relevance"].get("score", False)) / len(results)
    
    print(f"Total examples evaluated: {len(results)}\n")
    print(f"Correctness:         {avg_correctness:.2%}")
    print(f"Relevance:           {avg_relevance:.2%}")
    print(f"Groundedness:        {avg_groundedness:.2%}")
    print(f"Retrieval Relevance: {avg_retrieval_relevance:.2%}")
    
    avg_overall = (avg_correctness + avg_relevance + avg_groundedness + avg_retrieval_relevance) / 4
    print(f"\nOverall Average:     {avg_overall:.2%}")
    print("=" * 80)
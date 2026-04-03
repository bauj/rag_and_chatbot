import subprocess
import sys
import json
import os
import re
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


def _normalize_score_value(value: Any) -> float:
    if isinstance(value, bool):
        return 10.0 if value else 0.0
    if isinstance(value, (int, float)):
        score = float(value)
        if 0.0 <= score <= 1.0:
            return score * 10.0
        if 0.0 <= score <= 10.0:
            return score
        if 0.0 <= score <= 100.0:
            return score / 10.0
        return max(0.0, min(score, 10.0))
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            text = text[:-1].strip()
        if text.lower() in ("true", "yes"):
            return 10.0
        if text.lower() in ("false", "no"):
            return 0.0
        try:
            return _normalize_score_value(float(text))
        except ValueError:
            return 0.0
    return 0.0


def _extract_score_from_grade(grade: dict, keys: list) -> float:
    """Try multiple possible keys in the judge output to determine a normalized score on 0-10."""
    if not isinstance(grade, dict):
        return 0.0
    if "score" in grade:
        return _normalize_score_value(grade["score"])
    for k in keys:
        if k in grade:
            return _normalize_score_value(grade[k])
    # try nested common wrappers
    for wrapper in ("result", "data", "value"):
        if wrapper in grade and isinstance(grade[wrapper], dict):
            if "score" in grade[wrapper]:
                return _normalize_score_value(grade[wrapper]["score"])
            for k in keys:
                if k in grade[wrapper]:
                    return _normalize_score_value(grade[wrapper][k])
    return 0.0


def _run_structured_eval(llm: MistralLLM, instructions: str, content: str, score_keys: list):
    """Run a structured evaluator LLM and normalize the output.

    Returns: {score: float, explanation: str, explanation_raw: Any}
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
            score = _extract_score_from_grade(grade, score_keys)
            explanation_text = grade.get("explanation") or grade.get("content") or ""
            if not explanation_text and "reasoning" in grade:
                explanation_text = grade.get("reasoning")
        else:
            explanation_text = getattr(grade, "content", "") or ""
            score = _normalize_score_value(explanation_text)
            explanation_raw = {"content": explanation_text}
        return {"score": score, "explanation": explanation_text, "explanation_raw": explanation_raw}
    except Exception as e:
        return {"score": 0.0, "explanation": str(e), "explanation_raw": None}


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

def _load_examples():
    with open(dataset_file, "r", encoding="utf-8") as f:
        return json.load(f)

# Grade output schema
class CorrectnessGrade(TypedDict):
    # Note that the order in the fields are defined is the order in which the model will generate them.
    # It is useful to put explanations before responses because it forces the model to think through
    # its final response before generating it:
    explanation: Annotated[str, ..., "Expliquez votre raisonnement pour la note"]
    score: Annotated[float, ..., "Note entre 0 et 10"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

# Grade prompt
correctness_instructions = """Vous êtes un professeur qui note un quiz. Vous recevrez une QUESTION, la RÉPONSE DE RÉFÉRENCE et la RÉPONSE DE L'ÉTUDIANT. Utilisez une note entre 0 et 10 pour évaluer la précision factuelle de la réponse.

Critères de correction :
(1) Évaluez uniquement la précision factuelle par rapport à la réponse de référence.
(2) Vérifiez que la réponse de l'étudiant ne contient pas de contradictions internes.
(3) Il est acceptable que la réponse contienne plus d'informations que la réponse de référence, tant qu'elles sont factuellement exactes.

Répondez en français uniquement. Donnez la note dans la clé `score` comme un nombre entre 0 et 10. Dans `explanation`, donnez uniquement votre commentaire de notation, sans répéter la question, la réponse de référence ou la réponse de l'étudiant."""

# Grader LLM
grader_llm = MistralLLM(api_url=LLM_API_URL, model=LLM_MODEL, api_key=LLM_API_KEY, temperature=0).with_structured_output(CorrectnessGrade, method="json_schema", strict=True)


# Grade output schema
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Expliquez votre raisonnement pour la note"]
    score: Annotated[float, ..., "Note entre 0 et 10"]
    relevant: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]

# Grade prompt
relevance_instructions = """Vous êtes un professeur qui note un quiz. Vous recevrez une QUESTION et une RÉPONSE DE L'ÉTUDIANT. Utilisez une note entre 0 et 10 pour évaluer la pertinence de la réponse.

Critères de pertinence :
(1) La réponse doit être concise et directement liée à la question.
(2) La réponse doit aider à répondre à la question.

Répondez en français uniquement. Donnez la note dans la clé `score` comme un nombre entre 0 et 10. Dans `explanation`, donnez uniquement votre commentaire de notation, sans répéter la question ou la réponse de l'étudiant."""

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
    explanation: Annotated[str, ..., "Expliquez votre raisonnement pour la note"]
    score: Annotated[float, ..., "Note entre 0 et 10"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]

# Grade prompt
grounded_instructions = """Vous êtes un professeur qui note un quiz. Vous recevrez des FAITS et une RÉPONSE DE L'ÉTUDIANT. Utilisez une note entre 0 et 10 pour évaluer si la réponse est fondée sur les faits.

Critères de fondement :
(1) La réponse doit être ancrée dans les FAITS fournis.
(2) La réponse ne doit pas contenir d'informations « hallucinées » hors du champ des FAITS.

Répondez en français uniquement. Donnez la note dans la clé `score` comme un nombre entre 0 et 10. Dans `explanation`, donnez uniquement votre commentaire de notation, sans répéter les faits ou la réponse de l'étudiant."""

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
    explanation: Annotated[str, ..., "Expliquez votre raisonnement pour la note"]
    score: Annotated[float, ..., "Note entre 0 et 10"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]

# Grade prompt
retrieval_relevance_instructions = """Vous êtes un professeur qui note un quiz. Vous recevrez une QUESTION et un ensemble de DOCUMENTS fournis par l'étudiant. Utilisez une note entre 0 et 10 pour évaluer la pertinence de ces documents.

Critères de pertinence :
(1) Identifiez les DOCUMENTS qui sont complètement sans rapport avec la QUESTION.
(2) Si les documents contiennent des mots-clés ou un sens sémantique lié à la question, considérez-les comme pertinents.
(3) Il est acceptable que les documents contiennent des informations partiellement sans rapport tant que le critère (2) est respecté.

Répondez en français uniquement. Donnez la note dans la clé `score` comme un nombre entre 0 et 10. Dans `explanation`, donnez uniquement votre commentaire de notation, sans répéter la question ou les documents fournis."""

# Grader LLM
retrieval_relevance_llm = MistralLLM(
    api_url=LLM_API_URL,
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    temperature=0
).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)

def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """An evaluator for document relevance using agentic mode sources"""
    # Always use agentic mode sources for evaluation
    try:
        agentic_result = subprocess.run(
            [sys.executable, "chatbot.py", "--mode", "agentic", "--question", inputs['question'], "--json-output"],
            cwd=CHATBOT_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )
        if agentic_result.returncode == 0:
            stdout = agentic_result.stdout.strip()
            json_start = stdout.find('{')
            if json_start >= 0:
                json_str = stdout[json_start:]
                json_end = json_str.rfind('}')
                if json_end >= 0:
                    json_str = json_str[:json_end + 1]
                    chatbot_result = json.loads(json_str)
                    sources = chatbot_result.get("sources", [])
                else:
                    sources = []
            else:
                sources = []
        else:
            sources = []
    except:
        sources = []
    
    # Convert sources to document objects
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
    
    docs = documents
    doc_string = "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)
    answer = f"DOCUMENTS: {doc_string}\nQUESTION: {inputs['question']}"
    # Run evaluator
    return _run_structured_eval(retrieval_relevance_llm, retrieval_relevance_instructions, answer, ["relevant", "relevance", "result"])

def target(inputs: dict) -> dict:
    return rag_bot(inputs["question"])

# Test the RAG system with examples
print("Testing RAG system...")
print("=" * 80)

def _extract_explanation_from_raw(raw):
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts = [_extract_explanation_from_raw(item) for item in raw]
        return " \n".join(part for part in parts if part)
    if isinstance(raw, dict):
        for key in ("explanation", "reasoning", "REASONING", "reason", "message", "content", "text", "output"):
            if key in raw and raw[key]:
                return _extract_explanation_from_raw(raw[key])

        if "error" in raw and raw["error"]:
            error = raw["error"]
            if isinstance(error, str):
                return error.strip()
            if isinstance(error, dict):
                for key in ("message", "reason", "description", "details"):
                    if key in error and error[key]:
                        return _extract_explanation_from_raw(error[key])

        if "evaluation" in raw and isinstance(raw["evaluation"], dict):
            return _extract_explanation_from_raw(raw["evaluation"])

        parts = []
        for value in raw.values():
            candidate = _extract_explanation_from_raw(value)
            if candidate and candidate not in parts:
                parts.append(candidate)
        return " \n".join(parts)

    return ""


def _clean_explanation_text(text) -> str:
    if not text:
        return ""
    
    # Handle dict inputs by extracting string content
    if isinstance(text, dict):
        text = _extract_explanation_from_raw(text)
    
    # Ensure we have a string now
    if not isinstance(text, str):
        text = str(text)
    
    cleaned = text.replace("\r\n", "\n").strip()
    patterns = [
        r"(?ims)^question\s*:\s*.*?(?=\n(?:ground truth answer|student answer|facts|evaluation|$))",
        r"(?ims)^ground truth answer\s*:\s*.*?(?=\n(?:question|student answer|facts|evaluation|$))",
        r"(?ims)^student answer\s*:\s*.*?(?=\n(?:question|ground truth answer|facts|evaluation|$))",
        r"(?ims)^facts\s*:\s*.*?(?=\n(?:question|ground truth answer|student answer|evaluation|$))",
        r"(?ims)^question\s*[-]\s*.*?(?=\n(?:ground truth answer|student answer|facts|evaluation|$))",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)

    marker_patterns = [
        r"(?i)la réponse de l['’]étudiant",
        r"(?i)la réponse de l['’]étudiant·e",
        r"(?i)l['’]étudiant",
        r"(?i)l['’]étudiante",
        r"(?i)le modèle",
        r"(?i)la réponse est",
        r"(?i)en accord avec la vérité terre?rain",
    ]
    for marker in marker_patterns:
        m = re.search(marker, cleaned)
        if m:
            cleaned = cleaned[m.start():].strip()
            break

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _sanitize_eval_result(ev):
    """Extract score and explanation from evaluation result, always preserving LLM explanations."""
    if not isinstance(ev, dict):
        return {"score": bool(ev)}
    
    out = {"score": _normalize_score_value(ev.get("score", False))}
    expl = ev.get("explanation") or ""
    if expl:
        out["explanation"] = _clean_explanation_text(expl)
        return out
    
    expl = _extract_explanation_from_raw(ev.get("explanation_raw"))
    out["explanation"] = _clean_explanation_text(expl)
    return out

def main():
    examples = _load_examples()
    results = []
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
                "retrieval_relevance": _sanitize_eval_result(retrieval_rel),
            }
        })

        # Save results incrementally after each question
        results_file = "evaluation_results.json"
        with open(results_file, "w", encoding="utf-8") as rf:
            json.dump(results, rf, ensure_ascii=False, indent=2)

        print("-" * 80)

    print(f"\nEvaluation complete. Results saved to {results_file}")

    # Calculate and display average scores
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY - AVERAGE SCORES")
    print("=" * 80)

    if results:
        avg_correctness = sum(r["evaluations"]["correctness"].get("score", 0.0) for r in results) / len(results)
        avg_relevance = sum(r["evaluations"]["relevance"].get("score", 0.0) for r in results) / len(results)
        avg_groundedness = sum(r["evaluations"]["groundedness"].get("score", 0.0) for r in results) / len(results)
        avg_retrieval_relevance = sum(r["evaluations"]["retrieval_relevance"].get("score", 0.0) for r in results) / len(results)
        
        print(f"Total examples evaluated: {len(results)}\n")
        print(f"Correctness:         {avg_correctness:.1f}/10")
        print(f"Relevance:           {avg_relevance:.1f}/10")
        print(f"Groundedness:        {avg_groundedness:.1f}/10")
        print(f"Retrieval Relevance: {avg_retrieval_relevance:.1f}/10")
        
        avg_overall = (avg_correctness + avg_relevance + avg_groundedness + avg_retrieval_relevance) / 4
        print(f"\nOverall Average:     {avg_overall:.1f}/10")


if __name__ == "__main__":
    main()
    print("=" * 80)
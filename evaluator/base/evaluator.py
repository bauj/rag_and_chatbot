import subprocess
import sys
import json
import os
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing_extensions import Annotated, TypedDict
import requests
from typing import Any, Optional, Dict

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

evaluator_options = evaluator_config.get("evaluator_options", {})

AGENTIC_MODE = bool(evaluator_options.get("agentic_mode", False))

############################################################################################
################################ Custom Mistral LLM client #################################
############################################################################################

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

def _run_structured_eval(llm: MistralLLM, instructions: str, content: str):
    """Run a structured evaluator LLM and extract the output.

    Returns: {score: float, explanation: str}
    """
    try:
        grade = llm.invoke([
            {"role": "system", "content": instructions},
            {"role": "user", "content": content},
        ])
        if isinstance(grade, dict):
            score = float(grade.get("score", 0.0))
            explanation_text = grade.get("explanation", "")
        else:
            # Fallback for non-structured responses
            explanation_text = "Non-structured response ..."
            score = 0.0
        return {"score": score, "explanation": explanation_text}
    except Exception as e:
        return {"score": 0.0, "explanation": str(e)}

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

def _call_chatbot(question: str, mode: str = None, timeout_seconds: int = None,
                   config: Optional[Dict[str, Any]] = None) -> dict:
    """Call chatbot.py and return parsed answer/documents.

    `config` carries hyperparameter overrides (k, temperature, reranker_enabled,
    top_n, deep_dive for rag mode; temperature for agentic mode) and is translated
    into the matching chatbot.py CLI flags.
    """
    start_time = time.perf_counter()
    cmd = [sys.executable, "chatbot.py", "--question", question, "--json-output"]
    if mode:
        cmd.extend(["--mode", mode])

    if config:
        if config.get("k") is not None:
            cmd.extend(["--k", str(config["k"])])
        if config.get("temperature") is not None:
            cmd.extend(["--temperature", str(config["temperature"])])
        if config.get("top_n") is not None:
            cmd.extend(["--top-n", str(config["top_n"])])
        if config.get("reranker_enabled") is False:
            cmd.append("--no-rerank")
        if config.get("deep_dive"):
            cmd.append("--deep-dive")

    try:
        result = subprocess.run(
            cmd,
            cwd=CHATBOT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed = time.perf_counter() - start_time
        if result.returncode != 0:
            return {
                "answer": f"Error calling chatbot: {result.stderr}",
                "documents": [],
                "request_time": elapsed,
            }

        stdout = result.stdout.strip()
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

def run_evaluation(question: str, answer_dict: Dict, reference_answer: str, executor: Optional[ThreadPoolExecutor] = None) -> Dict[str, Dict]:
    """Run all evaluators on a single answer."""
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
            for metric in EVAL_METRICS
        }
        raw_results = {}
        for future in as_completed(futures):
            metric_name = futures[future]
            raw_results[metric_name] = future.result()
        for metric in EVAL_METRICS:
            evaluations[metric] = raw_results.get(metric, {"score": 0.0, "explanation": "No result returned."})
    else:
        for metric in EVAL_METRICS:
            evaluations[metric] = _eval_metric(metric)

    return evaluations

############################################################################################
####################################### Correctness ########################################
############################################################################################

# Grade output schema
class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "A step-by-step explanation in French justifying the score based on the criteria"]
    score: Annotated[float, ..., "A number between 0 and 10"]

# Grade prompt
correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER.

Your task is to evaluate the student’s answer based on the following criteria:

(1) Grade the student answer based ONLY on its factual accuracy relative to the ground truth answer.
(2) Ensure that the student answer does not contain any internal contradictions.
(3) It is acceptable for the student answer to include additional information, as long as it is factually accurate and consistent with the ground truth.

You must assign a score between 0 and 10 reflecting the factual accuracy of the student’s answer.

A score of 10 means the answer is completely factually correct.
A score of 0 means the answer is entirely incorrect.

Your output MUST be in JSON format with only two keys:

- "score": a number between 0 and 10
- "explanation": a single string paragraph written in English, explaining the score clearly and concisely, based on the criteria above."""

# Grader LLM
grader_llm = MistralLLM(
    api_url=LLM_API_URL,
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    temperature=0
).with_structured_output(CorrectnessGrade, method="json_schema", strict=True)

# Evaluator
def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Score the answer against the reference output."""
    answers = (
        f"QUESTION: {inputs['question']}\n"
        f"GROUND TRUTH ANSWER: {reference_outputs['answer']}\n"
        f"STUDENT ANSWER: {outputs['answer']}"
    )
    return _run_structured_eval(grader_llm, correctness_instructions, answers)

############################################################################################
######################################## Relevance #########################################
############################################################################################

# Grade output schema
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "A step-by-step explanation in French justifying the score based on the criteria"]
    score: Annotated[float, ..., "A number between 0 and 10"]

# Grade prompt
relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER.

Your task is to evaluate the relevance of the student’s answer based on the following criteria:

(1) Ensure the STUDENT ANSWER is concise and directly relevant to the QUESTION.
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION.

You must assign a score between 0 and 10 reflecting the relevance of the student’s answer:

A score of 10 means the answer is fully relevant, concise, and directly answers the question.
A score of 0 means the answer is irrelevant, off-topic, or does not help answer the question.

Your output MUST be in JSON format with only two keys:

- "score": a number between 0 and 10
- "explanation": a single string paragraph written in English, explaining the score clearly and concisely, based on the criteria above."""

# Grader LLM
relevance_llm = MistralLLM(
    api_url=LLM_API_URL,
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    temperature=0
).with_structured_output(RelevanceGrade, method="json_schema", strict=True)

# Evaluator
def relevance(inputs: dict, outputs: dict) -> dict:
    """Score whether the answer is relevant to the question."""
    answers = (
        f"QUESTION: {inputs['question']}\n"
        f"STUDENT ANSWER: {outputs['answer']}"
    )
    return _run_structured_eval(relevance_llm, relevance_instructions, answers)

############################################################################################
######################################## Groundedness ######################################
############################################################################################

# Grade output schema
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "A step-by-step explanation in French justifying the score based on the criteria"]
    score: Annotated[float, ..., "A number between 0 and 10"]

# Grade prompt
grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER.

Your task is to evaluate whether the student’s answer is grounded in the provided facts based on the following criteria:

(1) Ensure the STUDENT ANSWER is fully grounded in the FACTS.
(2) Ensure the STUDENT ANSWER does not contain any “hallucinated” information outside the scope of the FACTS.

You must assign a score between 0 and 10 reflecting how well the student’s answer is grounded in the facts:

A score of 10 means the answer is entirely based on the provided facts with no hallucinated information.
A score of 0 means the answer is not grounded in the facts at all or contains significant hallucinated information.

Your output MUST be in JSON format with only two keys:

- "score": a number between 0 and 10
- "explanation": a single string paragraph written in English, explaining the score clearly and concisely, based on the criteria above."""

# Grader LLM
grounded_llm = MistralLLM(
    api_url=LLM_API_URL,
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    temperature=0
).with_structured_output(GroundedGrade, method="json_schema", strict=True)

# Evaluator
def groundedness(_inputs: dict, outputs: dict) -> dict:
    """Score whether the answer is grounded in the provided documents."""
    documents = outputs.get("documents") or []
    if not documents:
        return {"score": 0.0, "explanation": "No documents were provided by the student."}
    
    doc_string = "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in documents)
    answers = (
        f"FACTS: {doc_string}\n"
        f"STUDENT ANSWER: {outputs['answer']}"
    )
    return _run_structured_eval(grounded_llm, grounded_instructions, answers)

############################################################################################
#################################### Retrieval Relevance ###################################
############################################################################################

# Grade output schema
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "A step-by-step explanation in French justifying the score based on the criteria"]
    score: Annotated[float, ..., "A number between 0 and 10"]

# Grade prompt
retrieval_relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS (documents) provided by the student.

Your task is to evaluate the relevance of these facts with respect to the question based on the following criteria:

(1) Identify any FACTS that are completely unrelated to the QUESTION.
(2) If the FACTS contain ANY keywords or semantic meaning related to the QUESTION, consider them relevant.
(3) It is acceptable for the FACTS to include some unrelated information as long as criterion (2) is met.

You must assign a score between 0 and 10 reflecting the overall relevance of the provided facts:

A score of 10 means the facts clearly contain relevant keywords or semantic meaning related to the question.
A score of 0 means the facts are completely unrelated to the question.

Your output MUST be in JSON format with only two keys:

- "score": a number between 0 and 10
- "explanation": a single string paragraph written in English, explaining the score clearly and concisely, based on the criteria above."""

# Grader LLM
retrieval_relevance_llm = MistralLLM(
    api_url=LLM_API_URL,
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    temperature=0
).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)

# Evaluator
def retrieval_relevance(inputs: dict, outputs: dict) -> dict:
    """Score whether provided documents are relevant to the question."""
    documents = outputs.get("documents") or []
    if not documents:
        return {"score": 0.0, "explanation": "No documents were provided by the student."}

    doc_string = "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in documents)
    answers = (
        f"DOCUMENTS: {doc_string}\n"
        f"QUESTION: {inputs['question']}"
    )
    return _run_structured_eval(retrieval_relevance_llm, retrieval_relevance_instructions, answers)

############################################################################################
###################################### Run on dataset ######################################
############################################################################################

# Load the examples for the dataset from JSON file
dataset_file = Path(__file__).resolve().parent / "dataset.json"

def _load_examples():
    with open(dataset_file, "r", encoding="utf-8") as f:
        return json.load(f)
    
# Test the Chatbot with dataset
print("Testing Chatbot...")
print("=" * 80)

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

def main(num_workers: int = 1, limit_questions: int = None, timeout_seconds: int = None):
    examples = _load_examples()
    if limit_questions:
        examples = examples[:limit_questions]
    results = []

    def fetch_chatbot(example_index: int, example: dict) -> tuple[int, dict, dict]:
        q = example['inputs']['question']
        output = _call_chatbot(q, "agentic" if AGENTIC_MODE else None, timeout_seconds)
        return example_index, example, output

    def evaluate_example(example_index: int, example: dict, output: dict) -> dict:
        q = example['inputs']['question']
        expected = example['outputs']['answer']
        evaluations = run_evaluation(q, output, expected)
        request_time = output.get("request_time", 0.0) if isinstance(output, dict) else 0.0
        rag_answer = output.get('answer') if isinstance(output, dict) else str(output)
        documents = output.get("documents", [])
        # Serialize documents for JSON
        serialized_documents = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]
        return {
            "question": q,
            "expected_answer": expected,
            "rag_answer": rag_answer,
            "request_time": request_time,
            "evaluations": evaluations,
            "documents": serialized_documents,
            "index": example_index,
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
                i = len(results)
                print(f"\n{'=' * 80}")
                print(f"Example {i}:")
                print(f"{'=' * 80}")
                print(f"\nQuestion: {result['question']}\n")
                print(f"Expected Answer: {result['expected_answer']}\n")
                rag_answer_preview = result['rag_answer'][:100] + "..." if len(result['rag_answer']) > 100 else result['rag_answer']
                print(f"RAG Answer (preview): {rag_answer_preview}\n")
                print("-" * 80)
                print("EVALUATION SCORES:")
                print("-" * 80)
                for metric, eval_result in result['evaluations'].items():
                    print(f"  {metric.replace('_', ' ').title()}: {eval_result.get('score', 0):.1f}")
                print("-" * 80)
    else:
        chatbot_results = [fetch_chatbot(i, example) for i, example in enumerate(examples, 1)]
        for idx, example, output in chatbot_results:
            result = evaluate_example(idx, example, output)
            results.append(result)
            print(f"\n{'=' * 80}")
            print(f"Example {idx}:")
            print(f"{'=' * 80}")
            print(f"\nQuestion: {result['question']}\n")
            print(f"Expected Answer: {result['expected_answer']}\n")
            rag_answer_preview = result['rag_answer'][:100] + "..." if len(result['rag_answer']) > 100 else result['rag_answer']
            print(f"RAG Answer (preview): {rag_answer_preview}\n")
            print("-" * 80)
            print("EVALUATION SCORES:")
            print("-" * 80)
            for metric, eval_result in result['evaluations'].items():
                print(f"  {metric.replace('_', ' ').title()}: {eval_result.get('score', 0):.1f}")
            print("-" * 80)

    # Save results
    results_file = "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as rf:
        json.dump(results, rf, ensure_ascii=False, indent=2)

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
        avg_request_time = sum(r.get("request_time", 0.0) for r in results) / len(results)

        print(f"Total examples evaluated: {len(results)}\n")
        print(f"Correctness:         {avg_correctness:.1f}/10")
        print(f"Relevance:           {avg_relevance:.1f}/10")
        print(f"Groundedness:        {avg_groundedness:.1f}/10")
        print(f"Retrieval Relevance: {avg_retrieval_relevance:.1f}/10")
        print(f"Average Chatbot Request Time: {avg_request_time:.3f} seconds")

        avg_overall = (avg_correctness + avg_relevance + avg_groundedness + avg_retrieval_relevance) / 4
        print(f"\nOverall Average:     {avg_overall:.1f}/10")

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
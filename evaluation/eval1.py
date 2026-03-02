import sys
import math
import os
import json
import requests
from langchain.chat_models import ChatOpenAI

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from evaluation.test import TestQuestion, load_tests
from implementation.answer import answer_question, fetch_context

load_dotenv(override=True)

# ----------------------------------
# Configuration
# ----------------------------------
llm = ChatOpenAI(
    model_name="groq/mixtral-8x7b-32768",
    temperature=0,
    openai_api_key="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",
    openai_api_base="https://api.groq.com/openai/v1"
)

# ----------------------------------
# Data Models
# ----------------------------------
class RetrievalEval(BaseModel):
    mrr: float
    ndcg: float
    keywords_found: int
    total_keywords: int
    keyword_coverage: float


class AnswerEval(BaseModel):
    feedback: str
    accuracy: float
    completeness: float
    relevance: float

# ----------------------------------
# Retrieval Metrics
# ----------------------------------
def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    keyword_lower = keyword.lower()
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0
        for doc in retrieved_docs[:k]
    ]
    dcg = calculate_dcg(relevances, k)
    idcg = calculate_dcg(sorted(relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval:
    retrieved_docs = fetch_context(test.question)

    # MRR
    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    # nDCG
    ndcg_scores = [
        calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords
    ]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # Keyword coverage
    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords else 0.0

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


# ----------------------------------
# Answer Evaluation Using Groq
# ----------------------------------
def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]:
    generated_answer, retrieved_docs = answer_question(test.question)

    # Build prompt for Groq API
    messages = [
        {"role": "system", "content": "You are an expert evaluator. Compare the generated answer with the reference answer."},
        {"role": "user", "content": f"""
Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Provide detailed feedback and scores (scale 1-5) for:
- accuracy
- completeness
- relevance

Output only valid JSON with keys:
feedback, accuracy, completeness, relevance
"""}
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(GROQ_BASE_URL, headers=headers, json=payload)
    data = response.json()

    # Extract text
    text = data["choices"][0]["message"]["content"]

    # Parse JSON safely
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {
            "feedback": text,
            "accuracy": 1.0,
            "completeness": 1.0,
            "relevance": 1.0,
        }

    answer_eval = AnswerEval(**parsed)
    return answer_eval, generated_answer, retrieved_docs


# ----------------------------------
# CLI Evaluation
# ----------------------------------
def run_cli_evaluation(test_number: int):
    tests = load_tests("tests.jsonl")
    if test_number < 0 or test_number >= len(tests):
        print("Error: test_row_number must be valid")
        sys.exit(1)

    test = tests[test_number]

    print(f"\n{'='*60}\nTest #{test_number}\n{'='*60}")
    print(f"Question: {test.question}")
    print(f"Reference Answer: {test.reference_answer}\n")

    # Retrieval metrics
    r = evaluate_retrieval(test)
    print(f"MRR: {r.mrr:.4f}, nDCG: {r.ndcg:.4f}, Coverage: {r.keyword_coverage:.1f}%\n")

    # Answer evaluation
    a, gen_ans, _ = evaluate_answer(test)
    print(f"Generated Answer:\n{gen_ans}\n")
    print("Evaluation Feedback:")
    print(f" Feedback: {a.feedback}")
    print(f" Accuracy: {a.accuracy}/5")
    print(f" Completeness: {a.completeness}/5")
    print(f" Relevance: {a.relevance}/5\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: python eval_groq.py <test_row_number>")
        return
    try:
        idx = int(sys.argv[1])
    except ValueError:
        print("Test number must be an integer")
        return

    run_cli_evaluation(idx)


if __name__ == "__main__":
    main()
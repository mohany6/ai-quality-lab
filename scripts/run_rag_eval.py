"""RAG regression gate: retrieval quality, grounding, citations, and numbers.

Usage (from the project root):

    python scripts/run_rag_eval.py
    python scripts/run_rag_eval.py --min-hit-rate 0.9 --min-groundedness 0.85
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_quality_lab.metrics import (  # noqa: E402
    citation_accuracy,
    contains_terms,
    groundedness,
    unsupported_numbers,
)
from ai_quality_lab.rag import (  # noqa: E402
    BM25Retriever,
    extractive_answer,
    load_corpus,
    load_retrieval_cases,
    retrieval_metrics,
)

CORPUS = ROOT / "datasets" / "rag_corpus.jsonl"
CASES = ROOT / "datasets" / "rag_cases.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-hit-rate", type=float, default=1.0)
    parser.add_argument("--min-mrr", type=float, default=0.80)
    parser.add_argument("--min-groundedness", type=float, default=0.85)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args(argv)

    retriever = BM25Retriever(load_corpus(CORPUS))
    cases = load_retrieval_cases(CASES)
    metrics = retrieval_metrics(cases, retriever)

    failures: list[str] = []
    if metrics["hit_rate"] < args.min_hit_rate:
        failures.append(f"hit_rate {metrics['hit_rate']:.3f} < {args.min_hit_rate:.3f}")
    if metrics["mrr"] < args.min_mrr:
        failures.append(f"mrr {metrics['mrr']:.3f} < {args.min_mrr:.3f}")

    case_results: list[dict[str, object]] = []
    for case in cases:
        retrieved = retriever.search(case.question, k=case.k)
        documents = [retriever.documents[item.doc_id] for item in retrieved]
        answer, citations = extractive_answer(case.question, documents)
        context = [document.text for document in documents]

        score = groundedness(answer, context)
        missing = contains_terms(answer, case.expected_terms)
        invented_numbers = unsupported_numbers(answer, context)
        citation_score = citation_accuracy(citations, list(case.relevant_ids))

        reasons: list[str] = []
        if not answer:
            reasons.append("empty answer")
        if missing:
            reasons.append(f"missing expected terms: {missing}")
        if invented_numbers:
            reasons.append(f"unsupported numbers: {invented_numbers}")
        if score < args.min_groundedness:
            reasons.append(f"groundedness {score:.3f} < {args.min_groundedness:.3f}")
        if not citations:
            reasons.append("no citations")

        for reason in reasons:
            failures.append(f"{case.id}: {reason}")

        case_results.append(
            {
                "id": case.id,
                "retrieved": [item.doc_id for item in retrieved],
                "citations": citations,
                "answer": answer,
                "groundedness": score,
                "citation_accuracy": citation_score,
                "missing_terms": missing,
                "unsupported_numbers": invented_numbers,
                "passed": not reasons,
                "reasons": reasons,
            }
        )

    artifacts = ROOT / args.artifacts
    artifacts.mkdir(parents=True, exist_ok=True)
    output = artifacts / "rag_report.json"
    output.write_text(
        json.dumps(
            {
                "summary": {
                    "cases": metrics["cases"],
                    "hit_rate": metrics["hit_rate"],
                    "mrr": metrics["mrr"],
                    "failed": len(failures),
                },
                "retrieval": metrics,
                "answers": case_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"rag cases={metrics['cases']} hit_rate={metrics['hit_rate']:.2%} "
        f"mrr={metrics['mrr']:.3f} failures={len(failures)}"
    )
    print(f"report: {output}")

    if failures:
        print("RAG quality gates failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

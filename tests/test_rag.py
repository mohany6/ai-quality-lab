from __future__ import annotations

import pytest

from ai_quality_lab.metrics import (
    citation_accuracy,
    contains_terms,
    groundedness,
    unsupported_numbers,
)
from ai_quality_lab.rag import (
    BM25Retriever,
    Document,
    extractive_answer,
    retrieval_metrics,
)


def test_corpus_loads_with_unique_ids(rag_documents) -> None:
    assert len(rag_documents) == 10
    assert len({document.doc_id for document in rag_documents}) == 10


def test_search_respects_k_and_orders_by_score(retriever) -> None:
    results = retriever.search("rate limit for the payments api", k=3)

    assert len(results) == 3
    assert results[0].doc_id == "helios-rate-limits"
    assert [item.score for item in results] == sorted(
        (item.score for item in results), reverse=True
    )


def test_search_rejects_k_below_one(retriever) -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        retriever.search("anything", k=0)


def test_retriever_rejects_duplicate_documents() -> None:
    documents = [
        Document(doc_id="a", title="A", text="first"),
        Document(doc_id="a", title="A", text="second"),
    ]

    with pytest.raises(ValueError, match="duplicate document id"):
        BM25Retriever(documents)


def test_retriever_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError, match="at least one document"):
        BM25Retriever([])


def test_labelled_set_reaches_full_hit_rate(rag_cases, retriever) -> None:
    metrics = retrieval_metrics(rag_cases, retriever)

    assert metrics["cases"] == len(rag_cases)
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] >= 0.8


def test_retrieval_metrics_expose_per_case_evidence(rag_cases, retriever) -> None:
    metrics = retrieval_metrics(rag_cases, retriever, k=3)

    details = metrics["details"]
    assert isinstance(details, list)
    assert len(details) == len(rag_cases)
    assert all(len(item["retrieved"]) == 3 for item in details)
    assert all(item["hit"] for item in details)


def test_extractive_answer_abstains_without_overlap(retriever) -> None:
    retrieved = retriever.search("quantum flux capacitor", k=3)
    documents = [retriever.documents[item.doc_id] for item in retrieved]

    answer, citations = extractive_answer("quantum flux capacitor", documents)

    assert answer == ""
    assert citations == []


def test_extractive_answers_are_grounded_and_cited(rag_cases, retriever) -> None:
    for case in rag_cases:
        retrieved = retriever.search(case.question, k=case.k)
        documents = [retriever.documents[item.doc_id] for item in retrieved]

        answer, citations = extractive_answer(case.question, documents)

        assert contains_terms(answer, case.expected_terms) == [], case.id
        assert citations, case.id
        assert groundedness(answer, [document.text for document in documents]) >= 0.85, case.id
        assert citation_accuracy(citations, list(case.relevant_ids)) >= 0.5, case.id


def test_hallucinated_number_is_detected() -> None:
    context = ["The Payments API rate limit is 600 requests per minute per client."]
    answer = "The rate limit is 5000 requests per minute for enterprise accounts."

    assert unsupported_numbers(answer, context) == ["5000"]
    assert groundedness(answer, context) < 0.7


def test_cited_wrong_source_reduces_accuracy() -> None:
    assert citation_accuracy(["helios-refunds"], ["helios-rate-limits"]) == 0.0
    assert citation_accuracy(["helios-refunds"], ["helios-rate-limits", "helios-refunds"]) == 1.0


def test_extractive_answer_rejects_invalid_sentence_budget(retriever) -> None:
    retrieved = retriever.search("rate limit", k=1)
    documents = [retriever.documents[item.doc_id] for item in retrieved]

    with pytest.raises(ValueError, match="max_sentences"):
        extractive_answer("rate limit", documents, max_sentences=0)

"""A small, dependency-free RAG evaluation target.

The point is not to be a production retriever. It is to give the test suite a
realistic retrieval pipeline whose failures are deterministic and explainable:
BM25 retrieval -> extractive answer -> citation and groundedness assertions.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .metrics import content_tokens

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievalCase:
    id: str
    question: str
    relevant_ids: tuple[str, ...]
    expected_terms: tuple[str, ...] = ()
    k: int = 3


@dataclass(frozen=True)
class Retrieved:
    doc_id: str
    score: float


class BM25Retriever:
    """Okapi BM25 over a fixed corpus, with deterministic tie-breaking."""

    def __init__(self, documents: list[Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("retriever needs at least one document")

        seen: set[str] = set()
        for document in documents:
            if document.doc_id in seen:
                raise ValueError(f"duplicate document id: {document.doc_id}")
            seen.add(document.doc_id)

        self.documents = {document.doc_id: document for document in documents}
        self._terms: dict[str, dict[str, int]] = {}
        self._lengths: dict[str, int] = {}
        self._document_frequency: dict[str, int] = {}

        for document in documents:
            tokens = content_tokens(f"{document.title} {document.text}")
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self._terms[document.doc_id] = counts
            self._lengths[document.doc_id] = len(tokens)
            for term in counts:
                self._document_frequency[term] = self._document_frequency.get(term, 0) + 1

        self._total_docs = len(documents)
        self._average_length = sum(self._lengths.values()) / self._total_docs
        self._k1 = k1
        self._b = b

    def _idf(self, term: str) -> float:
        frequency = self._document_frequency.get(term, 0)
        return math.log(1 + (self._total_docs - frequency + 0.5) / (frequency + 0.5))

    def search(self, query: str, *, k: int = 3) -> list[Retrieved]:
        if k < 1:
            raise ValueError("k must be at least 1")

        query_terms = content_tokens(query)
        scored: list[Retrieved] = []

        for doc_id, counts in self._terms.items():
            score = 0.0
            length = self._lengths[doc_id]
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * length / self._average_length
                )
                score += self._idf(term) * (frequency * (self._k1 + 1)) / denominator
            scored.append(Retrieved(doc_id=doc_id, score=round(score, 6)))

        scored.sort(key=lambda item: (-item.score, item.doc_id))
        return scored[:k]


def load_corpus(path: str | Path) -> list[Document]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"corpus not found: {source}")

    documents: list[Document] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        payload = json.loads(raw)
        required = {"doc_id", "title", "text"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"{source.name}:{line_number} missing keys: {sorted(missing)}")
        documents.append(
            Document(
                doc_id=str(payload["doc_id"]),
                title=str(payload["title"]),
                text=str(payload["text"]),
            )
        )

    if not documents:
        raise ValueError(f"{source.name} contains no documents")
    return documents


def load_retrieval_cases(path: str | Path) -> list[RetrievalCase]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"cases not found: {source}")

    cases: list[RetrievalCase] = []
    seen: set[str] = set()

    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        payload = json.loads(raw)
        required = {"id", "question", "relevant_ids"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"{source.name}:{line_number} missing keys: {sorted(missing)}")

        case_id = str(payload["id"])
        if case_id in seen:
            raise ValueError(f"{source.name}:{line_number} duplicate case id: {case_id}")
        seen.add(case_id)

        cases.append(
            RetrievalCase(
                id=case_id,
                question=str(payload["question"]),
                relevant_ids=tuple(payload["relevant_ids"]),
                expected_terms=tuple(payload.get("expected_terms", ())),
                k=int(payload.get("k", 3)),
            )
        )

    if not cases:
        raise ValueError(f"{source.name} contains no cases")
    return cases


def retrieval_metrics(
    cases: list[RetrievalCase], retriever: BM25Retriever, *, k: int | None = None
) -> dict[str, object]:
    """Compute hit-rate@k and MRR, plus per-case evidence for failure triage."""
    details: list[dict[str, object]] = []
    hits = 0
    reciprocal_ranks: list[float] = []

    for case in cases:
        top_k = k or case.k
        retrieved = [item.doc_id for item in retriever.search(case.question, k=top_k)]
        relevant = set(case.relevant_ids)
        hit = bool(relevant & set(retrieved))
        rank = next(
            (index for index, doc_id in enumerate(retrieved, start=1) if doc_id in relevant), None
        )

        hits += int(hit)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        details.append(
            {
                "id": case.id,
                "retrieved": retrieved,
                "relevant": sorted(relevant),
                "hit": hit,
                "rank": rank,
            }
        )

    total = len(cases)
    return {
        "cases": total,
        "k": k,
        "hits": hits,
        "hit_rate": round(hits / total, 4) if total else 0.0,
        "mrr": round(sum(reciprocal_ranks) / total, 4) if total else 0.0,
        "details": details,
    }


def extractive_answer(
    question: str,
    retrieved_docs: list[Document],
    *,
    max_sentences: int = 2,
) -> tuple[str, list[str]]:
    """Build a grounded answer by selecting the highest-overlap sentences.

    Returns ``(answer, cited_doc_ids)``. If nothing overlaps the question, it
    returns an empty answer with no citations so the suite can assert that the
    system abstains instead of hallucinating.
    """
    if max_sentences < 1:
        raise ValueError("max_sentences must be at least 1")

    query_terms = set(content_tokens(question))
    candidates: list[tuple[int, int, str, str]] = []

    for doc_rank, document in enumerate(retrieved_docs):
        for sentence_rank, sentence in enumerate(_SENTENCE_SPLIT.split(document.text)):
            sentence = sentence.strip()
            if not sentence:
                continue
            overlap = len(query_terms & set(content_tokens(sentence)))
            if overlap:
                position = doc_rank * 100 + sentence_rank
                candidates.append((-overlap, position, sentence, document.doc_id))

    if not candidates:
        return "", []

    candidates.sort()
    selected = candidates[:max_sentences]
    answer = " ".join(sentence for _, _, sentence, _ in selected)

    citations: list[str] = []
    for _, _, _, doc_id in selected:
        if doc_id not in citations:
            citations.append(doc_id)
    return answer, citations

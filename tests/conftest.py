from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_quality_lab.harness import load_cases  # noqa: E402
from ai_quality_lab.providers import ScriptedProvider  # noqa: E402
from ai_quality_lab.rag import (  # noqa: E402
    BM25Retriever,
    load_corpus,
    load_retrieval_cases,
)

DATA = ROOT / "datasets"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA


@pytest.fixture(scope="session")
def golden_cases():
    return load_cases(DATA / "golden_qa.jsonl")


@pytest.fixture(scope="session")
def mock_responses() -> dict[str, str]:
    responses: dict[str, str] = {}
    for raw in (DATA / "golden_qa.jsonl").read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        responses[payload["prompt"]] = payload["mock_response"]
    return responses


@pytest.fixture
def baseline_provider(mock_responses) -> ScriptedProvider:
    return ScriptedProvider(mock_responses, latency_ms=1.0, model="scripted-baseline")


@pytest.fixture(scope="session")
def rag_documents():
    return load_corpus(DATA / "rag_corpus.jsonl")


@pytest.fixture(scope="session")
def retriever(rag_documents) -> BM25Retriever:
    return BM25Retriever(rag_documents)


@pytest.fixture(scope="session")
def rag_cases():
    return load_retrieval_cases(DATA / "rag_cases.jsonl")

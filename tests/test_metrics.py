from __future__ import annotations

import pytest

from ai_quality_lab.metrics import (
    CostModel,
    citation_accuracy,
    citation_recall,
    contains_forbidden,
    contains_terms,
    exact_match,
    groundedness,
    normalise,
    percentile,
    similarity,
    token_f1,
    tokenize,
    unsupported_numbers,
)


def test_tokenize_keeps_digits_and_lowercases() -> None:
    assert tokenize("HTTP 429: Retry After 5s") == ["http", "429", "retry", "after", "5s"]


def test_normalise_ignores_case_and_punctuation() -> None:
    assert normalise("Retry-After, 30s!") == "retry after 30s"


def test_exact_match_uses_normalised_text() -> None:
    assert exact_match("HTTP 401", "http   401.")
    assert not exact_match("HTTP 401", "HTTP 403")


def test_token_f1_identical_text_is_one() -> None:
    assert token_f1("a relevant document in the top k", "a relevant document in the top k") == 1.0


def test_token_f1_disjoint_text_is_zero() -> None:
    assert token_f1("rate limit", "unrelated sentence") == 0.0


def test_similarity_prefers_paraphrase_over_unrelated() -> None:
    expected = "A smoke test verifies the most critical paths of a build."
    paraphrase = "A smoke test checks the most critical paths before deeper testing."
    unrelated = "The refund window is 180 days for card payments."

    assert similarity(expected, paraphrase) > similarity(expected, unrelated)
    assert 0.0 <= similarity(expected, unrelated) < 1.0


def test_contains_terms_lists_only_missing_terms() -> None:
    assert (
        contains_terms("run regression on every pull request", ("regression", "pull request")) == []
    )
    assert contains_terms("run it later", ("regression", "pull request")) == [
        "regression",
        "pull request",
    ]


def test_contains_forbidden_returns_leaks() -> None:
    assert contains_forbidden("this is a performance test", ("performance",)) == ["performance"]
    assert contains_forbidden("this is a smoke test", ("performance",)) == []


def test_citation_precision_and_recall() -> None:
    cited = ["doc-a", "doc-b"]
    relevant = ["doc-b", "doc-c"]

    assert citation_accuracy(cited, relevant) == 0.5
    assert citation_recall(cited, relevant) == 0.5
    assert citation_accuracy([], relevant) == 0.0
    assert citation_recall(cited, []) == 1.0


def test_groundedness_is_bounded_and_directional() -> None:
    context = ["The rate limit is 600 requests per minute per client."]

    grounded = groundedness("The rate limit is 600 requests per minute.", context)
    invented = groundedness(
        "The rate limit is 5000 requests per second for enterprise accounts.", context
    )

    assert grounded == 1.0
    assert 0.0 <= invented < 0.7


def test_unsupported_numbers_catches_numeric_hallucinations() -> None:
    context = ["The rate limit is 600 requests per minute. Refunds are allowed for 180 days."]

    assert unsupported_numbers("The rate limit is 600 requests per minute.", context) == []
    assert unsupported_numbers("The rate limit is 5000 requests per minute.", context) == ["5000"]


def test_unsupported_numbers_ignores_thousands_separators() -> None:
    assert unsupported_numbers("A limit of 1,000 records.", ["The limit is 1000 records."]) == []


def test_percentile_interpolates_between_values() -> None:
    assert percentile([10, 20, 30, 40], 50) == 25.0
    assert percentile([42], 95) == 42.0


def test_percentile_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1, 2], 101)


def test_cost_model_prices_input_and_output_separately() -> None:
    model = CostModel(input_per_million=2.5, output_per_million=10.0)
    assert model.cost(1_000_000, 1_000_000) == pytest.approx(12.5)
    assert model.cost(0, 0) == 0.0

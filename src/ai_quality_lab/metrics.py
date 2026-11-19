"""Deterministic, dependency-free quality metrics for text and RAG outputs.

The metrics in this module are intentionally explainable: every score can be
traced back to token overlap or citation sets. That makes failures debuggable
in CI instead of producing an opaque "the judge said no" result.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Small, explicit stopword list. Kept local so retrieval and scoring share the
# exact same notion of a "content token".
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "between",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "should",
        "than",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, preserving digits (API versions, status codes)."""
    return _WORD_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens that carry meaning for overlap scoring, with stopwords removed."""
    return [token for token in tokenize(text) if token not in STOPWORDS]


def normalise(text: str) -> str:
    """Collapse whitespace and punctuation so formatting never causes a miss."""
    return " ".join(tokenize(text))


def exact_match(expected: str, actual: str) -> bool:
    """Normalised string equality. Useful for classification and routing tasks."""
    return normalise(expected) == normalise(actual)


def token_f1(expected: str, actual: str) -> float:
    """Bag-of-words F1. Robust to word order, sensitive to missing content."""
    expected_counts = Counter(content_tokens(expected))
    actual_counts = Counter(content_tokens(actual))

    if not expected_counts and not actual_counts:
        return 1.0
    if not expected_counts or not actual_counts:
        return 0.0

    overlap = sum((expected_counts & actual_counts).values())
    precision = overlap / sum(actual_counts.values())
    recall = overlap / sum(expected_counts.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def similarity(expected: str, actual: str) -> float:
    """Blend token overlap with character-level similarity.

    Token F1 catches missing facts; character similarity catches morphological
    variation ("retries" vs "retry") and partial answers. The 65/35 weighting
    favours semantic content while remaining fully deterministic.
    """
    f1 = token_f1(expected, actual)
    char_ratio = SequenceMatcher(None, normalise(expected), normalise(actual)).ratio()
    return round(0.65 * f1 + 0.35 * char_ratio, 4)


def contains_terms(text: str, required: tuple[str, ...]) -> list[str]:
    """Return the required terms that are missing from ``text``."""
    haystack = normalise(text)
    return [term for term in required if normalise(term) not in haystack]


def contains_forbidden(text: str, forbidden: tuple[str, ...]) -> list[str]:
    """Return forbidden terms that leaked into the output."""
    haystack = normalise(text)
    return [term for term in forbidden if normalise(term) in haystack]


def citation_accuracy(cited: list[str], relevant: list[str]) -> float:
    """Share of cited documents that are actually relevant (precision)."""
    if not cited:
        return 0.0
    if not relevant:
        return 1.0
    return len(set(cited) & set(relevant)) / len(set(cited))


def citation_recall(cited: list[str], relevant: list[str]) -> float:
    """Share of relevant documents that were cited."""
    if not relevant:
        return 1.0
    return len(set(cited) & set(relevant)) / len(set(relevant))


def groundedness(answer: str, context: list[str]) -> float:
    """Fraction of answer content tokens supported by the retrieved context.

    This is a deliberately conservative lexical proxy. It is not a substitute
    for an LLM judge, but it is deterministic, cheap, and catches the common
    failure mode where a model invents a number or product name that was never
    retrieved.
    """
    answer_tokens = content_tokens(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = set(content_tokens(" ".join(context)))
    supported = sum(1 for token in answer_tokens if token in context_tokens)
    return round(supported / len(answer_tokens), 4)


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def unsupported_numbers(answer: str, context: list[str]) -> list[str]:
    """Numbers present in the answer but absent from the retrieved context.

    Numeric hallucinations are the highest-risk failure mode in RAG systems
    (amounts, rates, status codes, dates), so they get a dedicated check rather
    than relying on aggregate token overlap.
    """

    def canonical(value: str) -> str:
        return value.replace(",", "")

    context_numbers = {
        canonical(number) for text in context for number in _NUMBER_RE.findall(text)
    }
    return sorted(
        {
            number
            for number in _NUMBER_RE.findall(answer)
            if canonical(number) not in context_numbers
        }
    )


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile. ``p`` must be between 0 and 100."""
    if not values:
        raise ValueError("percentile() needs at least one value")
    if not 0 <= p <= 100:
        raise ValueError("percentile must be between 0 and 100")

    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    rank = (p / 100) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


@dataclass(frozen=True)
class CostModel:
    """Token pricing in USD per 1M tokens, matching public model price sheets."""

    input_per_million: float
    output_per_million: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens / 1_000_000 * self.input_per_million
            + completion_tokens / 1_000_000 * self.output_per_million
        )

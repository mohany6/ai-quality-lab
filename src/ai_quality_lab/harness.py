"""Case loading, per-case evaluation, suite gates, and baseline regression checks."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .metrics import (
    CostModel,
    contains_forbidden,
    contains_terms,
    percentile,
    similarity,
    tokenize,
)
from .providers import Provider


class QualityGateError(AssertionError):
    """Raised when a suite is valid but does not meet the release thresholds."""


@dataclass(frozen=True)
class EvalCase:
    id: str
    prompt: str
    expected: str
    threshold: float = 0.70
    required_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    max_latency_ms: float = 2_000.0
    max_completion_tokens: int = 200
    tags: tuple[str, ...] = ()


@dataclass
class EvalResult:
    case_id: str
    passed: bool
    score: float
    latency_ms: float
    completion_tokens: int
    output: str
    model: str
    failures: list[str] = field(default_factory=list)
    tags: tuple[str, ...] = ()


@dataclass
class SuiteReport:
    suite: str
    results: list[EvalResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return percentile([result.latency_ms for result in self.results], 95)

    @property
    def total_tokens(self) -> int:
        return sum(result.completion_tokens for result in self.results)

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.score for result in self.results) / self.total

    @property
    def failed_ids(self) -> list[str]:
        return [result.case_id for result in self.results if not result.passed]

    def estimated_cost(self, cost_model: CostModel, prompt_tokens: int = 0) -> float:
        return cost_model.cost(prompt_tokens, self.total_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": round(self.pass_rate, 4),
                "p95_latency_ms": round(self.p95_latency_ms, 2),
                "mean_score": round(self.mean_score, 4),
                "total_completion_tokens": self.total_tokens,
            },
            "failures": [
                {"case_id": result.case_id, "reasons": result.failures}
                for result in self.results
                if not result.passed
            ],
            "results": [asdict(result) for result in self.results],
        }


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load JSONL eval cases, validating required keys and duplicate ids."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"case file not found: {source}")

    cases: list[EvalCase] = []
    seen: set[str] = set()

    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source.name}:{line_number} is not valid JSON: {exc}") from exc

        missing = {"id", "prompt", "expected"} - payload.keys()
        if missing:
            raise ValueError(f"{source.name}:{line_number} missing keys: {sorted(missing)}")

        case_id = str(payload["id"])
        if case_id in seen:
            raise ValueError(f"{source.name}:{line_number} duplicate case id: {case_id}")
        seen.add(case_id)

        cases.append(
            EvalCase(
                id=case_id,
                prompt=str(payload["prompt"]),
                expected=str(payload["expected"]),
                threshold=float(payload.get("threshold", 0.70)),
                required_terms=tuple(payload.get("required_terms", ())),
                forbidden_terms=tuple(payload.get("forbidden_terms", ())),
                max_latency_ms=float(payload.get("max_latency_ms", 2_000)),
                max_completion_tokens=int(payload.get("max_completion_tokens", 200)),
                tags=tuple(payload.get("tags", ())),
            )
        )

    if not cases:
        raise ValueError(f"{source.name} contains no cases")
    return cases


def run_case(provider: Provider, case: EvalCase) -> EvalResult:
    """Execute one case and return a fully explained pass/fail result."""
    started = time.perf_counter()
    try:
        completion = provider.complete(case.prompt)
    except Exception as exc:  # noqa: BLE001 - provider failures must become test evidence
        return EvalResult(
            case_id=case.id,
            passed=False,
            score=0.0,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            completion_tokens=0,
            output="",
            model=getattr(provider, "name", "unknown"),
            failures=[f"provider_error: {type(exc).__name__}: {exc}"],
            tags=case.tags,
        )

    latency_ms = (time.perf_counter() - started) * 1000
    score = similarity(case.expected, completion.text)
    # Providers that do not report usage still get a token budget check.
    completion_tokens = completion.completion_tokens or len(tokenize(completion.text))

    failures: list[str] = []
    if score < case.threshold:
        failures.append(f"score {score:.3f} below threshold {case.threshold:.2f}")

    missing = contains_terms(completion.text, case.required_terms)
    if missing:
        failures.append(f"missing required terms: {missing}")

    leaked = contains_forbidden(completion.text, case.forbidden_terms)
    if leaked:
        failures.append(f"forbidden terms present: {leaked}")

    if latency_ms > case.max_latency_ms:
        failures.append(f"latency {latency_ms:.1f}ms exceeds budget {case.max_latency_ms:.0f}ms")

    if completion_tokens > case.max_completion_tokens:
        failures.append(
            f"completion tokens {completion_tokens} exceed budget {case.max_completion_tokens}"
        )

    return EvalResult(
        case_id=case.id,
        passed=not failures,
        score=score,
        latency_ms=round(latency_ms, 2),
        completion_tokens=completion_tokens,
        output=completion.text,
        model=completion.model,
        failures=failures,
        tags=case.tags,
    )


def run_suite(
    provider: Provider, cases: Iterable[EvalCase], *, name: str = "default"
) -> SuiteReport:
    return SuiteReport(suite=name, results=[run_case(provider, case) for case in cases])


def assert_quality_gates(
    report: SuiteReport,
    *,
    min_pass_rate: float = 0.95,
    max_p95_latency_ms: float = 1_500.0,
    min_mean_score: float | None = None,
    max_cost_usd: float | None = None,
    cost_model: CostModel | None = None,
    prompt_tokens: int = 0,
) -> None:
    """Fail the build with every violated gate listed in one actionable message."""
    violations: list[str] = []

    if report.pass_rate < min_pass_rate:
        violations.append(
            f"pass rate {report.pass_rate:.1%} < required {min_pass_rate:.0%} "
            f"(failed: {', '.join(report.failed_ids) or 'n/a'})"
        )
    if report.p95_latency_ms > max_p95_latency_ms:
        violations.append(
            f"p95 latency {report.p95_latency_ms:.1f}ms > budget {max_p95_latency_ms:.0f}ms"
        )
    if min_mean_score is not None and report.mean_score < min_mean_score:
        violations.append(f"mean score {report.mean_score:.3f} < required {min_mean_score:.3f}")
    if max_cost_usd is not None:
        if cost_model is None:
            raise ValueError("cost_model is required when max_cost_usd is set")
        cost = report.estimated_cost(cost_model, prompt_tokens=prompt_tokens)
        if cost > max_cost_usd:
            violations.append(f"estimated cost ${cost:.6f} > budget ${max_cost_usd:.6f}")

    if violations:
        raise QualityGateError("quality gates failed:\n- " + "\n- ".join(violations))


def to_baseline(report: SuiteReport) -> dict[str, dict[str, Any]]:
    """Serialise a report into the compact shape stored for regression checks."""
    return {
        result.case_id: {"score": result.score, "passed": result.passed}
        for result in report.results
    }


def compare_to_baseline(
    report: SuiteReport,
    baseline: dict[str, dict[str, Any]],
    *,
    score_tolerance: float = 0.05,
) -> list[str]:
    """Return human-readable regressions against a previous accepted run.

    A regression is either a case that used to pass and now fails, a score drop
    larger than ``score_tolerance``, or a baseline case that disappeared from
    the candidate suite (silent coverage loss).
    """
    if score_tolerance < 0:
        raise ValueError("score_tolerance must be non-negative")

    regressions: list[str] = []
    current = {result.case_id: result for result in report.results}

    for case_id, previous in baseline.items():
        result = current.get(case_id)
        if result is None:
            regressions.append(f"{case_id}: missing from candidate suite")
            continue
        if previous.get("passed") and not result.passed:
            reasons = "; ".join(result.failures)
            regressions.append(f"{case_id}: passed before, now fails ({reasons})")
        elif result.score < float(previous.get("score", 0.0)) - score_tolerance:
            regressions.append(
                f"{case_id}: score dropped {float(previous['score']):.3f} -> {result.score:.3f}"
            )

    return regressions

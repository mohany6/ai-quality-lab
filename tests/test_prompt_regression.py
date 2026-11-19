from __future__ import annotations

import pytest

from ai_quality_lab.harness import (
    EvalResult,
    SuiteReport,
    compare_to_baseline,
    run_suite,
    to_baseline,
)
from ai_quality_lab.providers import ScriptedProvider


def test_baseline_round_trip_has_no_regressions(golden_cases, baseline_provider) -> None:
    report = run_suite(baseline_provider, golden_cases)
    baseline = to_baseline(report)

    assert set(baseline) == {case.id for case in golden_cases}
    assert compare_to_baseline(report, baseline) == []


def test_detects_pass_to_fail(golden_cases, mock_responses) -> None:
    baseline = to_baseline(
        run_suite(ScriptedProvider(mock_responses, latency_ms=1.0), golden_cases)
    )

    broken = dict(mock_responses)
    broken[golden_cases[1].prompt] = "The answer is unavailable."
    candidate = run_suite(ScriptedProvider(broken, latency_ms=1.0), golden_cases)

    regressions = compare_to_baseline(candidate, baseline)

    assert any(
        golden_cases[1].id in item and "passed before, now fails" in item for item in regressions
    )


def test_detects_score_drop_beyond_tolerance() -> None:
    result = EvalResult(
        case_id="qa-001",
        passed=True,
        score=0.80,
        latency_ms=10.0,
        completion_tokens=12,
        output="answer",
        model="candidate",
    )
    report = SuiteReport(suite="candidate", results=[result])
    baseline = {"qa-001": {"score": 0.90, "passed": True}}

    regressions = compare_to_baseline(report, baseline, score_tolerance=0.05)

    assert len(regressions) == 1
    assert "score dropped" in regressions[0]


def test_ignores_drop_within_tolerance() -> None:
    result = EvalResult(
        case_id="qa-001",
        passed=True,
        score=0.86,
        latency_ms=10.0,
        completion_tokens=12,
        output="answer",
        model="candidate",
    )
    report = SuiteReport(suite="candidate", results=[result])
    baseline = {"qa-001": {"score": 0.90, "passed": True}}

    assert compare_to_baseline(report, baseline, score_tolerance=0.05) == []


def test_detects_silent_coverage_loss() -> None:
    report = SuiteReport(suite="candidate", results=[])
    baseline = {"qa-999": {"score": 1.0, "passed": True}}

    assert compare_to_baseline(report, baseline) == ["qa-999: missing from candidate suite"]


def test_compare_is_deterministic(golden_cases, baseline_provider) -> None:
    report = run_suite(baseline_provider, golden_cases)
    baseline = to_baseline(report)

    first = compare_to_baseline(report, baseline)
    second = compare_to_baseline(report, baseline)

    assert first == second == []


def test_negative_tolerance_is_rejected(golden_cases, baseline_provider) -> None:
    report = run_suite(baseline_provider, golden_cases)

    with pytest.raises(ValueError, match="score_tolerance"):
        compare_to_baseline(report, to_baseline(report), score_tolerance=-1.0)

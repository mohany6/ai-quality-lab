from __future__ import annotations

import json
from xml.etree import ElementTree

import pytest

from ai_quality_lab.harness import (
    EvalCase,
    QualityGateError,
    assert_quality_gates,
    load_cases,
    run_case,
    run_suite,
)
from ai_quality_lab.providers import Completion, ScriptedProvider
from ai_quality_lab.reporting import write_json_report, write_junit_report


class FailingProvider:
    name = "failing-provider"

    def complete(self, prompt: str, *, temperature: float = 0.0) -> Completion:
        raise RuntimeError("upstream returned 503")


def test_load_cases_reads_every_line(golden_cases) -> None:
    assert len(golden_cases) == 10
    assert golden_cases[0].id == "qa-001"
    assert golden_cases[0].tags == ("smoke", "fundamentals")


def test_load_cases_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "dupes.jsonl"
    row = {"id": "same", "prompt": "p", "expected": "e"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case id"):
        load_cases(path)


def test_load_cases_rejects_missing_keys(tmp_path) -> None:
    path = tmp_path / "missing.jsonl"
    path.write_text(json.dumps({"id": "x", "prompt": "p"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing keys"):
        load_cases(path)


def test_run_case_passes_for_grounded_answer(golden_cases, baseline_provider) -> None:
    result = run_case(baseline_provider, golden_cases[0])

    assert result.passed, result.failures
    assert result.score >= golden_cases[0].threshold
    assert result.failures == []


def test_run_case_flags_missing_required_term(baseline_provider) -> None:
    case = EvalCase(
        id="required-term",
        prompt="Irrelevant prompt",
        expected="Irrelevant expected answer",
        threshold=0.0,
        required_terms=("blockchain",),
        max_latency_ms=500,
        max_completion_tokens=50,
    )

    result = run_case(baseline_provider, case)

    assert not result.passed
    assert any("missing required terms" in failure for failure in result.failures)


def test_run_case_flags_forbidden_term() -> None:
    provider = ScriptedProvider({"prompt": "This performance benchmark looks good."})
    case = EvalCase(
        id="forbidden-term",
        prompt="prompt",
        expected="This performance benchmark looks good.",
        threshold=0.0,
        forbidden_terms=("performance",),
        max_latency_ms=500,
        max_completion_tokens=50,
    )

    result = run_case(provider, case)

    assert not result.passed
    assert any("forbidden terms" in failure for failure in result.failures)


def test_run_case_flags_latency_budget() -> None:
    provider = ScriptedProvider({"prompt": "fast enough"}, latency_ms=20)
    case = EvalCase(
        id="slow-case",
        prompt="prompt",
        expected="fast enough",
        threshold=0.0,
        max_latency_ms=5,
        max_completion_tokens=50,
    )

    result = run_case(provider, case)

    assert not result.passed
    assert any("latency" in failure and "budget" in failure for failure in result.failures)


def test_run_case_flags_token_budget() -> None:
    provider = ScriptedProvider({"prompt": "one two three four five"})
    case = EvalCase(
        id="verbose",
        prompt="prompt",
        expected="one two three four five",
        threshold=0.0,
        max_latency_ms=500,
        max_completion_tokens=1,
    )

    result = run_case(provider, case)

    assert not result.passed
    assert any("completion tokens" in failure for failure in result.failures)


def test_provider_error_becomes_test_evidence() -> None:
    case = EvalCase(id="boom", prompt="prompt", expected="answer")

    result = run_case(FailingProvider(), case)

    assert not result.passed
    assert result.score == 0.0
    assert result.failures == ["provider_error: RuntimeError: upstream returned 503"]


def test_suite_gates_pass_for_baseline(golden_cases, baseline_provider) -> None:
    report = run_suite(baseline_provider, golden_cases, name="golden-qa")

    assert_quality_gates(
        report,
        min_pass_rate=1.0,
        max_p95_latency_ms=150,
        min_mean_score=0.75,
    )
    assert report.total == 10
    assert report.failed == 0


def test_suite_gates_fail_with_actionable_message(golden_cases, mock_responses) -> None:
    broken = dict(mock_responses)
    broken[golden_cases[0].prompt] = "Something unrelated entirely."
    provider = ScriptedProvider(
        broken, latency_ms=1.0, model="scripted-regressed", name="regressed"
    )

    report = run_suite(provider, golden_cases, name="golden-qa")

    with pytest.raises(QualityGateError) as captured:
        assert_quality_gates(report, min_pass_rate=0.95)

    message = str(captured.value)
    assert "pass rate" in message
    assert "qa-001" in message


def test_gate_requires_cost_model_when_budget_set(golden_cases, baseline_provider) -> None:
    report = run_suite(baseline_provider, golden_cases)

    with pytest.raises(ValueError, match="cost_model"):
        assert_quality_gates(report, max_cost_usd=0.01)


def test_reports_are_written_and_parseable(golden_cases, baseline_provider, tmp_path) -> None:
    report = run_suite(baseline_provider, golden_cases, name="golden-qa")

    json_path = write_json_report(report, tmp_path / "artifacts" / "report.json")
    junit_path = write_junit_report(report, tmp_path / "artifacts" / "junit.xml")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["suite"] == "golden-qa"
    assert payload["summary"]["total"] == 10
    assert payload["summary"]["failed"] == 0

    root = ElementTree.parse(junit_path).getroot()
    assert root.tag == "testsuite"
    assert root.attrib["tests"] == "10"
    assert root.attrib["failures"] == "0"
    assert len(root.findall("testcase")) == 10


def test_junit_records_failure_reasons(golden_cases, mock_responses, tmp_path) -> None:
    broken = dict(mock_responses)
    broken[golden_cases[0].prompt] = "Something unrelated entirely."
    report = run_suite(ScriptedProvider(broken), golden_cases, name="golden-qa")
    junit_path = write_junit_report(report, tmp_path / "junit.xml")

    root = ElementTree.parse(junit_path).getroot()
    failures = root.findall(".//failure")
    assert len(failures) == 1
    assert "qa-001" in failures[0].text
    assert "score" in failures[0].text

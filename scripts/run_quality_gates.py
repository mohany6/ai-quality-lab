"""Release gate for the golden QA suite.

Usage (from the project root):

    python scripts/run_quality_gates.py                  # compare + enforce gates
    python scripts/run_quality_gates.py --update-baseline
    python scripts/run_quality_gates.py --live           # real OpenAI-compatible endpoint

Exits non-zero when a quality gate or a baseline regression fails, which is what
makes this usable as a CI step rather than a reporting-only script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_quality_lab.harness import (  # noqa: E402
    QualityGateError,
    assert_quality_gates,
    compare_to_baseline,
    load_cases,
    run_suite,
    to_baseline,
)
from ai_quality_lab.providers import (  # noqa: E402
    OpenAICompatibleProvider,
    ScriptedProvider,
)
from ai_quality_lab.reporting import write_json_report, write_junit_report  # noqa: E402

DATASET = ROOT / "datasets" / "golden_qa.jsonl"


def _mock_responses() -> dict[str, str]:
    responses: dict[str, str] = {}
    for raw in DATASET.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        responses[payload["prompt"]] = payload["mock_response"]
    return responses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="use a real OpenAI-compatible endpoint")
    parser.add_argument("--update-baseline", action="store_true", help="accept the current run")
    parser.add_argument("--baseline", default="baselines/golden_qa.json")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--max-p95-latency-ms", type=float, default=1_500.0)
    parser.add_argument("--min-mean-score", type=float, default=0.80)
    args = parser.parse_args(argv)

    cases = load_cases(DATASET)
    provider = (
        OpenAICompatibleProvider()
        if args.live
        else ScriptedProvider(_mock_responses(), latency_ms=1.0, model="scripted-baseline")
    )
    report = run_suite(provider, cases, name="golden-qa")

    artifacts = ROOT / args.artifacts
    json_report = write_json_report(report, artifacts / "golden_qa_report.json")
    junit_report = write_junit_report(report, artifacts / "golden_qa_junit.xml")

    baseline_path = ROOT / args.baseline
    status = 0

    if args.update_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(to_baseline(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"baseline updated: {baseline_path}")
    elif baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        regressions = compare_to_baseline(report, baseline)
        if regressions:
            status = 1
            print("baseline regressions detected:")
            for regression in regressions:
                print(f"  - {regression}")

    try:
        assert_quality_gates(
            report,
            min_pass_rate=args.min_pass_rate,
            max_p95_latency_ms=args.max_p95_latency_ms,
            min_mean_score=args.min_mean_score,
        )
    except QualityGateError as exc:
        status = 1
        print(exc)

    print(
        f"suite={report.suite} cases={report.total} passed={report.passed} "
        f"failed={report.failed} pass_rate={report.pass_rate:.1%} "
        f"p95={report.p95_latency_ms:.1f}ms mean_score={report.mean_score:.3f}"
    )
    print(f"reports: {json_report} | {junit_report}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

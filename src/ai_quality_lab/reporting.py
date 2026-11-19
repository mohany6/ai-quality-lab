"""Report writers: machine-readable JSON and CI-native JUnit XML."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

from .harness import SuiteReport


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json_report(report: SuiteReport, path: str | Path) -> Path:
    destination = Path(path)
    _ensure_parent(destination)
    destination.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return destination


def write_junit_report(report: SuiteReport, path: str | Path) -> Path:
    """Write a JUnit XML file that any CI runner can display natively."""
    destination = Path(path)
    _ensure_parent(destination)

    suite = ElementTree.Element(
        "testsuite",
        {
            "name": report.suite,
            "tests": str(report.total),
            "failures": str(report.failed),
            "errors": "0",
            "time": f"{sum(r.latency_ms for r in report.results) / 1000:.3f}",
        },
    )

    for result in report.results:
        case = ElementTree.SubElement(
            suite,
            "testcase",
            {
                "classname": f"ai_quality_lab.{report.suite}",
                "name": result.case_id,
                "time": f"{result.latency_ms / 1000:.3f}",
            },
        )
        if not result.passed:
            failure = ElementTree.SubElement(
                case,
                "failure",
                {"message": "; ".join(result.failures) or "quality gate failed"},
            )
            failure.text = (
                f"case_id={result.case_id}\n"
                f"score={result.score:.4f}\n"
                f"model={result.model}\n"
                f"output={result.output}\n"
                f"reasons={result.failures}"
            )

    ElementTree.ElementTree(suite).write(destination, encoding="utf-8", xml_declaration=True)
    return destination

from __future__ import annotations

import json
from pathlib import Path

from ai_quality_lab.harness import compare_to_baseline, run_suite

BASELINE = Path(__file__).resolve().parents[1] / "baselines" / "golden_qa.json"


def test_committed_baseline_matches_current_suite(golden_cases, baseline_provider) -> None:
    """The accepted baseline must cover every case and show no regressions.

    This keeps the committed file honest: adding a case without refreshing the
    baseline, or silently changing a prompt, fails here rather than in CI later.
    """
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    report = run_suite(baseline_provider, golden_cases, name="golden-qa")

    assert set(baseline) == {case.id for case in golden_cases}
    assert compare_to_baseline(report, baseline) == []

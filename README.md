# AI Quality Lab

A provider-agnostic evaluation harness for LLM and RAG systems, built as a **pytest quality gate** rather than a dashboard. It answers one question in CI: *did this prompt, model, or retriever change make the product worse?*

[![CI](https://github.com/mohany6/ai-quality-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/mohany6/ai-quality-lab/actions/workflows/ci.yml)

---

## Why this exists

Traditional assertions break on generative output because the same prompt can produce different wording on every run. This harness treats model output as a **system under test** and evaluates it with layered, explainable checks:

- **Quality** — semantic similarity, required facts, forbidden claims
- **Reliability** — pass rate against a golden dataset, baseline regression detection
- **Performance** — p95 latency and completion-token budgets per case
- **Cost** — token pricing converted into a per-suite budget
- **Grounding** — citation accuracy, groundedness, unsupported-number detection

Everything runs deterministically with zero API keys. A live OpenAI-compatible adapter is available behind an opt-in flag.

---

## What is covered

| Suite | Cases | What it proves |
|---|---:|---|
| Golden Q&A (`datasets/golden_qa.jsonl`) | 10 | Answer quality, required facts, forbidden claims, latency and token budgets |
| Prompt regression (`baselines/golden_qa.json`) | 10 | A prompt/model change cannot silently drop below the accepted baseline |
| RAG retrieval (`datasets/rag_cases.jsonl`) | 8 | hit-rate@k and MRR over a labelled corpus |
| RAG grounding | 8 | Citation accuracy, groundedness, numeric hallucination detection, abstention |
| Unit tests | 55 total | Metrics, harness behaviour, gates, reporting, providers, retriever |

Current mock-provider result: **10/10 cases pass, mean score 0.857, p95 latency 1.6 ms**, RAG **hit-rate@3 100%, MRR 1.000**. The full suite runs in about **1.5 seconds**.

---

## Architecture

```
datasets/*.jsonl
      │
      ▼
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Case loader │────▶│  Provider under  │────▶│  Metric layer   │
│  (validated) │     │  test (Protocol) │     │  similarity/F1  │
└──────────────┘     └──────────────────┘     │  grounding      │
                                               │  citations      │
                                               │  numbers        │
                                               └────────┬────────┘
                                                        │
                        ┌───────────────────────────────┼────────────────────┐
                        ▼                               ▼                    ▼
                 ┌─────────────┐               ┌───────────────┐    ┌──────────────┐
                 │ Quality     │               │ Baseline      │    │ JSON + JUnit │
                 │ gates       │               │ regression    │    │ artifacts    │
                 │ (fail CI)   │               │ (fail CI)     │    │ (any runner) │
                 └─────────────┘               └───────────────┘    └──────────────┘
```

Two interchangeable providers ship in the box:

- `ScriptedProvider` — deterministic replay of recorded outputs, used by CI
- `OpenAICompatibleProvider` — any `/chat/completions` endpoint, used for live runs

Both satisfy a two-method `Protocol`, so adding Bedrock, Azure OpenAI, or a local model is a small adapter, not a rewrite.

---

## Quickstart

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"

pytest
python scripts/run_quality_gates.py
python scripts/run_rag_eval.py
```

Outputs land in `artifacts/`:

- `golden_qa_report.json` — per-case scores, latency, tokens and failure reasons
- `golden_qa_junit.xml` — native CI test results
- `rag_report.json` — retrieval evidence, answers, citations and grounding scores

### Accepting a new baseline

Only after reviewing the diff and the generated report:

```bash
python scripts/run_quality_gates.py --update-baseline
```

### Running against a real model

```bash
pip install -e ".[live]"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="..."
export LLM_MODEL="gpt-4o-mini"

python scripts/run_quality_gates.py --live --min-pass-rate 0.8
```

Live runs are never part of CI: model output is non-deterministic and the suite must stay free and repeatable.

---

## Quality gates

`assert_quality_gates()` fails with **every** violated gate in one message, including the failing case IDs:

```python
assert_quality_gates(
    report,
    min_pass_rate=1.0,
    max_p95_latency_ms=150,
    min_mean_score=0.80,
    max_cost_usd=0.01,
    cost_model=CostModel(input_per_million=2.50, output_per_million=10.00),
)
```

RAG gates are enforced by `scripts/run_rag_eval.py`:

- `hit_rate >= 1.0` on the labelled set
- `MRR >= 0.80`
- groundedness `>= 0.85` per answer
- every expected fact present
- zero unsupported numbers
- every answer cited

---

## Design decisions

**1. Deterministic by default.** CI must not depend on a model vendor, a network, or a credit card. The scripted provider replays a fixed set of outputs, so a red build always means the *harness or the expected contract* changed — never "the model was in a mood".

**2. Explainable metrics over a black-box judge.** Every score decomposes into token overlap, character similarity, citation sets, or numeric sets. When a case fails, the report says which required term was missing and which number was invented. An LLM judge can be added later, but it should not be the only line of defence.

**3. Baselines are committed artifacts.** `baselines/golden_qa.json` is reviewed like code. `compare_to_baseline()` catches pass→fail transitions, score drops beyond a tolerance, and **silent coverage loss** when a case disappears.

**4. Numeric hallucinations get their own check.** Rates, amounts, status codes and dates are the highest-risk RAG failure mode, so `unsupported_numbers()` verifies every number in an answer against the retrieved context instead of trusting aggregate overlap.

**5. Abstention is a pass condition.** When nothing in the corpus overlaps the question, `extractive_answer()` returns an empty answer with no citations. The suite asserts that behaviour rather than rewarding a confident guess.

**6. Test evidence is a product.** JUnit XML, JSON reports, per-case failure reasons and token usage are emitted on every run. A failing pipeline should tell you exactly what to fix.

---

## Mapping to the role

| Requirement | Evidence in this repo |
|---|---|
| Design, develop, execute, maintain automated tests | 55 pytest tests across 6 suites, fixtures, parameterised cases, strict markers |
| Automated tests for AI-powered applications | Golden-dataset LLM evaluation, prompt regression, RAG quality gates |
| API testing | Provider contract tests, error-path tests, adapter boundary |
| Frameworks and reusable assets | `Provider` Protocol, reusable harness, metrics library, report writers |
| CI/CD integration | GitHub Actions on Python 3.10–3.12, JUnit artifact, non-zero exit on gate failure |
| Defect identification and reporting | Per-case failure reasons, JSON/JUnit evidence, baseline diffing |
| Python | Package layout, typed dataclasses, Protocol, argparse CLIs |
| GenAI / ML exposure | LLM output evaluation, RAG retrieval metrics, grounding and hallucination checks |

---

## Project layout

```
ai-quality-lab/
├── datasets/
│   ├── golden_qa.jsonl          # 10 reference answers + per-case budgets
│   ├── rag_corpus.jsonl         # 10-document Helios Payments API corpus
│   └── rag_cases.jsonl          # 8 labelled retrieval questions
├── baselines/
│   └── golden_qa.json           # accepted per-case scores
├── src/ai_quality_lab/
│   ├── harness.py               # case loading, execution, gates, baseline diff
│   ├── metrics.py               # similarity, F1, grounding, citations, cost
│   ├── providers.py             # ScriptedProvider + OpenAI-compatible adapter
│   ├── rag.py                   # BM25 retriever + extractive answerer
│   └── reporting.py             # JSON and JUnit writers
├── scripts/
│   ├── run_quality_gates.py     # golden Q&A release gate
│   └── run_rag_eval.py          # RAG retrieval + grounding gate
├── tests/                       # 55 tests
└── .github/workflows/ci.yml
```

---

## Limitations and next steps

- Similarity is lexical. Embedding cosine similarity and an optional LLM judge would improve paraphrase handling.
- The RAG corpus is intentionally small; the next step is a versioned corpus with per-document provenance and retrieval drift tracking.
- Agentic workflows need trajectory assertions (tool choice, argument validity, step budget), which would be a separate `AgentCase` type.
- Mutation testing of prompts (systematically weakening instructions) would harden the suite further.
- The live adapter currently targets OpenAI-compatible APIs only.

---

## Author

**Mohamed Hany Fathy** — Software Engineer, Helwan University (Computers & AI)
[GitHub](https://github.com/mohany6) · [Portfolio](https://mohany-portfolio.vercel.app) · mahamedhany8@gmail.com

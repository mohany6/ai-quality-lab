"""Model providers used by the evaluation harness.

Design rule: the harness depends on a tiny Protocol, never on a vendor SDK.
CI uses ``ScriptedProvider`` so every assertion is deterministic and free.
Live runs opt in through ``OpenAICompatibleProvider``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .metrics import tokenize


@dataclass(frozen=True)
class Completion:
    """A single model response plus the telemetry needed for cost budgets."""

    text: str
    model: str = "mock"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


@runtime_checkable
class Provider(Protocol):
    """Minimal surface the harness needs from any model under test."""

    name: str

    def complete(self, prompt: str, *, temperature: float = 0.0) -> Completion:
        """Return a completion for ``prompt``."""


class ScriptedProvider:
    """Deterministic provider backed by a prompt-to-answer mapping.

    Used to test the harness itself and to replay recorded model output in CI.
    ``default`` models an unhelpful or abstaining system, which lets the suite
    assert that quality gates actually fail when they should.
    """

    def __init__(
        self,
        responses: Mapping[str, str],
        *,
        default: str = "I cannot answer that from the provided information.",
        latency_ms: float = 0.0,
        model: str = "scripted-baseline",
        name: str = "scripted",
    ) -> None:
        self._responses = dict(responses)
        self._default = default
        self._latency_ms = latency_ms
        self.model = model
        self.name = name

    def complete(self, prompt: str, *, temperature: float = 0.0) -> Completion:
        if temperature != 0.0:
            raise ValueError("scripted provider only supports temperature=0")
        if self._latency_ms:
            time.sleep(self._latency_ms / 1000)

        text = self._responses.get(prompt, self._default)
        return Completion(
            text=text,
            model=self.model,
            prompt_tokens=len(tokenize(prompt)),
            completion_tokens=len(tokenize(text)),
            latency_ms=self._latency_ms,
        )


class OpenAICompatibleProvider:
    """Thin adapter for any OpenAI-compatible ``/chat/completions`` endpoint.

    Kept optional so the default install has zero third-party dependencies.
    Credentials are read from the environment and never logged.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Live provider requires the optional dependency: pip install 'ai-quality-lab[live]'"
            ) from exc

        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.timeout_s = timeout_s
        self.name = f"openai-compatible:{self.model}"

        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is required for live evaluation runs")

    def complete(self, prompt: str, *, temperature: float = 0.0) -> Completion:
        import httpx

        started = time.perf_counter()
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        elapsed_ms = (time.perf_counter() - started) * 1000

        usage = body.get("usage", {})
        return Completion(
            text=body["choices"][0]["message"]["content"],
            model=body.get("model", self.model),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=round(elapsed_ms, 2),
        )

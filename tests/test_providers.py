from __future__ import annotations

import pytest

from ai_quality_lab.providers import Completion, Provider, ScriptedProvider


def test_scripted_provider_is_deterministic() -> None:
    provider = ScriptedProvider({"prompt": "the answer"})

    first = provider.complete("prompt")
    second = provider.complete("prompt")

    assert first == second
    assert first.text == "the answer"


def test_scripted_provider_uses_default_for_unknown_prompts() -> None:
    provider = ScriptedProvider({}, default="I cannot answer that.")

    assert provider.complete("anything").text == "I cannot answer that."


def test_scripted_provider_rejects_non_zero_temperature() -> None:
    provider = ScriptedProvider({"prompt": "answer"})

    with pytest.raises(ValueError, match="temperature=0"):
        provider.complete("prompt", temperature=0.7)


def test_scripted_provider_reports_token_usage() -> None:
    provider = ScriptedProvider({"a b c": "one two three four"})

    completion = provider.complete("a b c")

    assert completion.prompt_tokens == 3
    assert completion.completion_tokens == 4
    assert completion.model == "scripted-baseline"


def test_scripted_provider_satisfies_protocol() -> None:
    assert isinstance(ScriptedProvider({}), Provider)


def test_completion_defaults_are_safe() -> None:
    completion = Completion(text="hello")

    assert completion.model == "mock"
    assert completion.prompt_tokens == 0
    assert completion.completion_tokens == 0

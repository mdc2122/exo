"""Tests for the `<model>:nothink` / `<model>:think` request aliases."""

from __future__ import annotations

from exo.api.adapters.chat_completions import normalize_thinking_model_suffix
from exo.api.types.api import ChatCompletionMessage, ChatCompletionRequest
from exo.shared.types.common import ModelId

_BASE_MODEL = "kernelpool/MiMo-V2.5-Pro-6bit"


def _request(model: str, enable_thinking: bool | None = None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=ModelId(model),
        messages=[ChatCompletionMessage(role="user", content="Hello")],
        enable_thinking=enable_thinking,
    )


def test_nothink_suffix_strips_model_and_disables_thinking() -> None:
    normalized = normalize_thinking_model_suffix(_request(f"{_BASE_MODEL}:nothink"))
    assert normalized.model == _BASE_MODEL
    assert normalized.enable_thinking is False


def test_think_suffix_strips_model_and_enables_thinking() -> None:
    normalized = normalize_thinking_model_suffix(_request(f"{_BASE_MODEL}:think"))
    assert normalized.model == _BASE_MODEL
    assert normalized.enable_thinking is True


def test_suffix_overrides_body_enable_thinking() -> None:
    # Upstream proxies have been observed injecting enable_thinking=true;
    # the explicit model suffix must win.
    normalized = normalize_thinking_model_suffix(
        _request(f"{_BASE_MODEL}:nothink", enable_thinking=True)
    )
    assert normalized.enable_thinking is False


def test_plain_model_passes_through_unchanged() -> None:
    request = _request(_BASE_MODEL, enable_thinking=None)
    normalized = normalize_thinking_model_suffix(request)
    assert normalized is request
    assert normalized.enable_thinking is None

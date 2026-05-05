"""Provider-aware token counting.

Each provider has its own canonical way to count tokens. We delegate to the
provider's API where it exists (vLLM `/tokenize`, Anthropic `messages.count_tokens`,
Google `models.count_tokens`) and fall back to tiktoken otherwise.

Counters are async because some are network calls. Sync callers can use
`count_tokens_sync` for quick tiktoken-only paths.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx
import tiktoken
from openai import AsyncOpenAI

from managers import telemetry as tel
from models.common import ProviderKind

# Tiktoken-based safety overhead for *unknown* models / fallback paths.
# Provider-native counts are authoritative and don't apply the multiplier.
_FALLBACK_OVERHEAD = 1.20


class TokenCounter(ABC):
    @abstractmethod
    async def count(self, text: str, *, model: str) -> int: ...


class TiktokenCounter(TokenCounter):
    """Tiktoken with the right encoding per model. Cheap, in-process."""

    def __init__(self, default_encoding: str = "o200k_base"):
        self._default = default_encoding
        self._cache: dict[str, tiktoken.Encoding] = {}

    def _encoding_for(self, model: str) -> tiktoken.Encoding:
        if model in self._cache:
            return self._cache[model]
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding(self._default)
        self._cache[model] = enc
        return enc

    async def count(self, text: str, *, model: str) -> int:
        return len(self._encoding_for(model).encode(text))


class FallbackCounter(TiktokenCounter):
    """Like TiktokenCounter but applies a safety multiplier for non-OpenAI
    models where the encoding doesn't perfectly match the upstream tokenizer."""

    async def count(self, text: str, *, model: str) -> int:
        raw = await super().count(text, model=model)
        return int(raw * _FALLBACK_OVERHEAD)


class AnthropicCounter(TokenCounter):
    """Anthropic exposes server-side token counting."""

    def __init__(self, client: Any):  # AsyncAnthropic
        self.client = client

    async def count(self, text: str, *, model: str) -> int:
        try:
            resp = await self.client.messages.count_tokens(
                model=model,
                messages=[{"role": "user", "content": text}],
            )
            return int(resp.input_tokens)
        except Exception as e:
            tel.warn("Anthropic count_tokens failed (%s) — falling back to tiktoken", e)
            return await FallbackCounter().count(text, model=model)


class GoogleCounter(TokenCounter):
    """Google Gemini server-side token counting."""

    def __init__(self, client: Any):  # genai.Client
        self.client = client

    async def count(self, text: str, *, model: str) -> int:
        try:
            resp = await self.client.aio.models.count_tokens(
                model=model, contents=text
            )
            return int(resp.total_tokens)
        except Exception as e:
            tel.warn("Google count_tokens failed (%s) — falling back to tiktoken", e)
            return await FallbackCounter().count(text, model=model)


class VLLMCounter(TokenCounter):
    """vLLM exposes a `/tokenize` HTTP endpoint that returns the exact token
    count for the model loaded on the server."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._http = httpx.AsyncClient(timeout=10.0)

    async def count(self, text: str, *, model: str) -> int:
        url = f"{self.base_url}/tokenize"
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = await self._http.post(
                url,
                headers=headers,
                json={"model": model, "prompt": text},
            )
            resp.raise_for_status()
            data = resp.json()
            return int(data.get("count") or len(data.get("tokens", [])))
        except Exception as e:
            tel.warn("vLLM /tokenize failed (%s) — falling back to tiktoken", e)
            return await FallbackCounter().count(text, model=model)

    async def aclose(self) -> None:
        await self._http.aclose()


def make_counter(
    provider_kind: ProviderKind, client: Any, base_url: Optional[str], api_key: Optional[str]
) -> TokenCounter:
    """Factory: pick the right counter for a provider config."""
    if provider_kind in (ProviderKind.OPENAI, ProviderKind.OPENAI_CHAT):
        return TiktokenCounter()
    if provider_kind == ProviderKind.GROQ:
        # Groq serves Llama / Mixtral; tiktoken cl100k_base is a reasonable
        # approximation but inexact, so apply the overhead multiplier.
        return FallbackCounter("cl100k_base")
    if provider_kind == ProviderKind.ANTHROPIC:
        return AnthropicCounter(client)
    if provider_kind == ProviderKind.GOOGLE:
        return GoogleCounter(client)
    if provider_kind == ProviderKind.VLLM:
        if not base_url:
            raise ValueError("VLLM provider requires base_url for /tokenize")
        return VLLMCounter(base_url=base_url, api_key=api_key)
    return FallbackCounter()


def make_default_openai_counter() -> TiktokenCounter:
    """Used by ContextOverflowManager for pre-flight checks when we don't yet
    know the target model. Cheap and fast — no network."""
    return TiktokenCounter()


__all__ = [
    "TokenCounter",
    "TiktokenCounter",
    "FallbackCounter",
    "AnthropicCounter",
    "GoogleCounter",
    "VLLMCounter",
    "make_counter",
    "make_default_openai_counter",
]

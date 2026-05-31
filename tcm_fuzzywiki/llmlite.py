"""Small llmlite-style LLM abstraction with an Azure ChatGPT REST adapter.

The extractor depends on this tiny interface instead of a concrete SDK.  It keeps
TCM-FuzzyWiki auditable: an LLM may extract observations, but downstream fuzzy
membership, rules, inference, and aggregation remain deterministic and replayable.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class ChatModel(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> object:
        """Return a JSON-compatible object produced by the model."""


@dataclass(slots=True)
class AzureChatGPTConfig:
    endpoint: str
    deployment: str
    api_key: str
    api_version: str = "2024-02-15-preview"
    temperature: float = 0.0
    max_tokens: int = 1200

    @classmethod
    def from_env(cls) -> "AzureChatGPTConfig":
        return cls(
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
            deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            temperature=float(os.environ.get("AZURE_OPENAI_TEMPERATURE", "0")),
            max_tokens=int(os.environ.get("AZURE_OPENAI_MAX_TOKENS", "1200")),
        )


class AzureChatGPTLLM:
    """Minimal Azure Chat Completions client using the llmlite interface."""

    def __init__(self, config: AzureChatGPTConfig):
        self.config = config

    def complete_json(self, system_prompt: str, user_prompt: str) -> object:
        url = (
            f"{self.config.endpoint}/openai/deployments/{self.config.deployment}"
            f"/chat/completions?api-version={self.config.api_version}"
        )
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"api-key": self.config.api_key, "content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Azure ChatGPT API request failed: {exc}") from exc
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)


class NullLLM:
    """A deterministic empty model used when no LLM provider is configured."""

    def complete_json(self, system_prompt: str, user_prompt: str) -> object:
        return {"observations": []}

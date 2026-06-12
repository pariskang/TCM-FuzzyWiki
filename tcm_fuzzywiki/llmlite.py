"""Small llmlite-style LLM abstraction with an Azure ChatGPT REST adapter.

The extractor depends on this tiny interface instead of a concrete SDK.  It keeps
TCM-FuzzyWiki auditable: an LLM may extract observations, but downstream fuzzy
membership, rules, inference, and aggregation remain deterministic and replayable.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

try:  # Optional dependency: JSON repair improves robustness but is never required.
    from json_repair import repair_json as _repair_json
except ImportError:  # pragma: no cover - depends on optional environment package.
    _repair_json = None


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


# ---------------------------------------------------------------------------
# Robust JSON payload parsing for OpenAI-compatible chat models
# ---------------------------------------------------------------------------
# Reasoning-style models (e.g. MiniMax-M3) may wrap JSON in <think> blocks or
# Markdown fences, truncate output, or emit trailing commas.  These helpers
# recover the observation payload deterministically without trusting the model
# for anything beyond observation rows.

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_PAYLOAD_LIST_ALIASES = ("observations", "observation", "items", "data", "results")


def strip_think_and_fences(text: str) -> str:
    text = _THINK_RE.sub("", text or "").strip()
    text = _FENCE_OPEN_RE.sub("", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _balanced_json_objects(text: str) -> list[str]:
    """Extract balanced ``{...}`` candidates while respecting quoted strings."""

    candidates: list[str] = []
    depth = 0
    start: int | None = None
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : index + 1])
                start = None
    return candidates


def _json_candidates(text: str) -> list[str]:
    cleaned = strip_think_and_fences(text)
    candidates = [cleaned] if cleaned else []
    candidates.extend(block.strip() for block in _FENCE_BLOCK_RE.findall(text or "") if block.strip())
    candidates.extend(_balanced_json_objects(cleaned))
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def _loads_candidate(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    if _repair_json is not None:
        try:
            return json.loads(_repair_json(candidate, ensure_ascii=False))
        except Exception:  # pragma: no cover - repair library internals vary.
            pass
    return json.loads(_TRAILING_COMMA_RE.sub(r"\1", candidate))


def parse_json_payload(text: str) -> dict[str, Any]:
    """Parse a chat completion into ``{"observations": [...]}`` or raise ValueError."""

    errors: list[str] = []
    for candidate in _json_candidates(text):
        try:
            payload = _loads_candidate(candidate)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        if isinstance(payload, list):
            return {"observations": payload}
        if isinstance(payload, dict):
            if "observations" not in payload:
                for alias in _PAYLOAD_LIST_ALIASES:
                    if isinstance(payload.get(alias), list):
                        return {"observations": payload[alias]}
            return payload
    preview = strip_think_and_fences(text)[:500]
    raise ValueError(f"No JSON payload could be parsed ({'; '.join(errors)[:400] or 'empty response'}): {preview!r}")


@dataclass(slots=True)
class OpenAICompatibleConfig:
    """Configuration for any OpenAI-compatible Chat Completions endpoint.

    Works with MiniMax (``https://api.minimaxi.com/v1`` or ``api.minimax.io``),
    OpenAI, vLLM, or other compatible servers.  ``extra_body`` is merged into the
    request payload (e.g. MiniMax ``{"thinking": {"type": "disabled"}}``).
    """

    base_url: str
    model: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 3000
    timeout: float = 180.0
    max_retries: int = 4
    retry_sleep: float = 4.0
    use_response_format: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        base_url: str | None = None,
        **overrides: Any,
    ) -> "OpenAICompatibleConfig":
        api_key = os.environ.get("MINIMAX_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        if not api_key:
            raise RuntimeError("Set MINIMAX_API_KEY or OPENAI_API_KEY for the OpenAI-compatible LLM adapter.")
        return cls(
            base_url=(base_url or os.environ.get("OPENAI_BASE_URL", "https://api.minimaxi.com/v1")).rstrip("/"),
            model=model or os.environ.get("OPENAI_MODEL", "MiniMax-M3"),
            api_key=api_key,
            **overrides,
        )


class OpenAICompatibleLLM:
    """Minimal OpenAI-compatible Chat Completions client with retry and JSON repair.

    Like :class:`AzureChatGPTLLM`, this stays SDK-free (urllib only) so the core
    pipeline gains no heavy dependency.  The model is only ever trusted for
    observation rows; all downstream fuzzy computation remains deterministic.
    """

    def __init__(self, config: OpenAICompatibleConfig):
        self.config = config

    def complete_json(self, system_prompt: str, user_prompt: str) -> object:
        payload, _meta = self.complete_json_with_meta(system_prompt, user_prompt)
        return payload

    def complete_json_with_meta(self, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                try:
                    return self._call_once(system_prompt, user_prompt, self.config.use_response_format, attempt)
                except urllib.error.HTTPError as exc:
                    body = self._http_error_body(exc)
                    if self.config.use_response_format and exc.code in (400, 422) and "response_format" in body:
                        return self._call_once(system_prompt, user_prompt, False, attempt)
                    raise RuntimeError(f"HTTP {exc.code}: {body[:300]}") from exc
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_sleep * attempt)
        raise RuntimeError(
            f"OpenAI-compatible completion failed after {self.config.max_retries} attempts: {last_error}"
        ) from last_error

    def _call_once(
        self, system_prompt: str, user_prompt: str, with_response_format: bool, attempt: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        body.update(self.config.extra_body)
        if with_response_format:
            body["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.config.api_key}", "content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        content = str(message.get("content") or "")
        reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
        payload = parse_json_payload(content if content.strip() else reasoning)
        meta = {
            "response_id": data.get("id", ""),
            "model": data.get("model", self.config.model),
            "finish_reason": choice.get("finish_reason", ""),
            "usage": data.get("usage", {}) or {},
            "attempt": attempt,
        }
        return payload, meta

    @staticmethod
    def _http_error_body(exc: urllib.error.HTTPError) -> str:
        try:
            return exc.read().decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - depends on server stream state.
            return str(exc)

from pathlib import Path

import pytest

from tcm_fuzzywiki.config import load_yaml
from tcm_fuzzywiki.llmlite import (
    AnthropicCompatibleConfig,
    AnthropicCompatibleLLM,
    text_from_anthropic_blocks,
)
from tcm_fuzzywiki.models import SourceUnit
from tcm_fuzzywiki.resume import extract_resumable


class _FakeMessage:
    def __init__(self, data: dict):
        self._data = data

    def model_dump(self) -> dict:
        return self._data


class _FakeMessages:
    def __init__(self, outer: "FakeAnthropic"):
        self.outer = outer

    def create(self, **kwargs):
        self.outer.calls.append(kwargs)
        if self.outer.fail_first > 0:
            self.outer.fail_first -= 1
            raise RuntimeError("simulated transient 529 overloaded")
        # Reasoning model emits a thinking block plus dirty JSON with trailing commas.
        return _FakeMessage(
            {
                "id": "msg_1",
                "model": "MiniMax-M3",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 12, "output_tokens": 8},
                "content": [
                    {"type": "thinking", "thinking": "让我分析原文……"},
                    {
                        "type": "text",
                        "text": '{"observations":[{"feature":"pain_quality","feature_value":"冷痛","evidence_text":"腰痛而冷",},],}',
                    },
                ],
            }
        )


class FakeAnthropic:
    def __init__(self, fail_first: int = 0):
        self.calls: list[dict] = []
        self.fail_first = fail_first
        self.messages = _FakeMessages(self)


def _llm(fail_first: int = 0) -> AnthropicCompatibleLLM:
    config = AnthropicCompatibleConfig(
        base_url="https://api.minimaxi.com/anthropic",
        model="MiniMax-M3",
        api_key="dummy",
        max_retries=3,
        retry_sleep=0.0,
    )
    llm = AnthropicCompatibleLLM(config, prefer_sdk=False)
    llm._client = FakeAnthropic(fail_first=fail_first)
    return llm


def test_text_from_anthropic_blocks_handles_dicts_and_objects():
    class Block:
        def __init__(self, type, **kw):
            self.type = type
            for k, v in kw.items():
                setattr(self, k, v)

    blocks = [Block("thinking", thinking="hmm"), Block("text", text="答案")]
    text, thinking = text_from_anthropic_blocks(blocks)
    assert text == "答案" and thinking == "hmm"

    text2, thinking2 = text_from_anthropic_blocks(
        [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}, {"type": "thinking", "thinking": "t"}]
    )
    assert text2 == "AB" and thinking2 == "t"
    assert text_from_anthropic_blocks(None) == ("", "")


def test_anthropic_adapter_parses_dirty_json_and_reports_meta():
    llm = _llm()
    payload, meta = llm.complete_json_with_meta("system", "user")
    assert payload["observations"][0]["feature_value"] == "冷痛"
    assert meta["usage"]["input_tokens"] == 12
    assert meta["finish_reason"] == "end_turn"
    assert meta["attempt"] == 1
    # system prompt routed to Anthropic top-level `system`, not a message role.
    call = llm._client.calls[0]
    assert call["system"] == "system"
    assert call["messages"] == [{"role": "user", "content": "user"}]


def test_anthropic_adapter_retries_then_succeeds():
    llm = _llm(fail_first=2)
    payload, meta = llm.complete_json_with_meta("system", "user")
    assert payload["observations"][0]["feature_value"] == "冷痛"
    assert meta["attempt"] == 3
    assert len(llm._client.calls) == 3


def test_anthropic_thinking_passed_through_when_configured():
    config = AnthropicCompatibleConfig(
        base_url="https://api.minimaxi.com/anthropic",
        model="MiniMax-M3",
        api_key="dummy",
        thinking={"type": "enabled", "budget_tokens": 1024},
    )
    llm = AnthropicCompatibleLLM(config, prefer_sdk=False)
    llm._client = FakeAnthropic()
    llm.complete_json_with_meta("s", "u")
    assert llm._client.calls[0]["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_anthropic_adapter_drives_resumable_extraction(tmp_path: Path):
    config = load_yaml("configs/tcm_fuzzywiki.yaml")
    sources = [SourceUnit(source_id="SRC_A", book_name="书", original_text="腰痛而冷，得温则缓。")]
    llm = _llm()
    observations, report = extract_resumable(
        sources, config, tmp_path, llm, chunk_chars=2000, chunk_overlap=0, input_sha256="H"
    )
    assert report["chunks_failed"] == 0
    assert any(o.feature_value == "冷痛" for o in observations)

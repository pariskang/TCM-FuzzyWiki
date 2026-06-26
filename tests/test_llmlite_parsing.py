import pytest

from tcm_fuzzywiki.llmlite import (
    OpenAICompatibleConfig,
    parse_json_payload,
    strip_think_and_fences,
)


def test_parse_plain_json_object():
    payload = parse_json_payload('{"observations": [{"feature": "symptom", "feature_value": "腰痛"}]}')
    assert payload["observations"][0]["feature_value"] == "腰痛"


def test_parse_strips_think_blocks_and_fences():
    text = '<think>让我想想……</think>\n```json\n{"observations": [{"feature_value": "冷痛"}]}\n```'
    payload = parse_json_payload(text)
    assert payload["observations"][0]["feature_value"] == "冷痛"
    assert strip_think_and_fences(text).startswith('{"observations"')


def test_parse_repairs_trailing_commas():
    payload = parse_json_payload('{"observations": [{"feature_value": "脉弦",},],}')
    assert payload["observations"][0]["feature_value"] == "脉弦"


def test_parse_accepts_bare_list_and_alias_keys():
    assert parse_json_payload('[{"feature_value": "苔黄腻"}]')["observations"][0]["feature_value"] == "苔黄腻"
    assert parse_json_payload('{"items": [{"feature_value": "久病"}]}')["observations"][0]["feature_value"] == "久病"


def test_parse_extracts_embedded_object_from_prose():
    text = '好的，结果如下：{"observations": [{"feature_value": "刺痛"}]} 以上。'
    assert parse_json_payload(text)["observations"][0]["feature_value"] == "刺痛"


def test_parse_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_json_payload("完全不是 JSON 的内容")


def test_azure_config_from_explicit_args():
    config = OpenAICompatibleConfig.from_azure(
        model="Kimi-K2.5",
        endpoint="https://fosterpearson-ft-5186-resource.openai.azure.com",
        api_key="secret-key",
    )
    assert config.model == "Kimi-K2.5"
    assert config.base_url == "https://fosterpearson-ft-5186-resource.openai.azure.com/openai/v1"
    assert config.api_key == "secret-key"


def test_azure_config_accepts_endpoint_with_v1_suffix():
    config = OpenAICompatibleConfig.from_azure(
        model="Kimi-K2.5",
        endpoint="https://res.openai.azure.com/openai/v1/",
        api_key="k",
    )
    assert config.base_url == "https://res.openai.azure.com/openai/v1"


def test_azure_config_reads_environment(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "Kimi-K2.5")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    config = OpenAICompatibleConfig.from_azure()
    assert config.base_url == "https://res.openai.azure.com/openai/v1"
    assert config.model == "Kimi-K2.5"
    assert config.api_key == "env-key"


def test_azure_config_requires_endpoint(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    with pytest.raises(RuntimeError):
        OpenAICompatibleConfig.from_azure()


def test_poe_config_defaults_to_gpt54_and_poe_base_url():
    config = OpenAICompatibleConfig.from_poe(api_key="poe-key")
    assert config.base_url == "https://api.poe.com/v1"
    assert config.model == "gpt-5.4"
    assert config.api_key == "poe-key"


def test_poe_config_reads_environment(monkeypatch):
    monkeypatch.setenv("POE_API_KEY", "env-poe")
    monkeypatch.setenv("POE_MODEL", "gpt-5.4")
    config = OpenAICompatibleConfig.from_poe()
    assert config.api_key == "env-poe"
    assert config.model == "gpt-5.4"
    assert config.base_url + "/chat/completions" == "https://api.poe.com/v1/chat/completions"


def test_poe_config_requires_api_key(monkeypatch):
    monkeypatch.delenv("POE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenAICompatibleConfig.from_poe()

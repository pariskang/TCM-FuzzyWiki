import pytest

from tcm_fuzzywiki.llmlite import parse_json_payload, strip_think_and_fences


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

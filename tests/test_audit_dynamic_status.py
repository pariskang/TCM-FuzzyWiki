import io
import json
import urllib.request

from tcm_fuzzywiki.audit import capability_rows
from tcm_fuzzywiki.extraction import LLMObservationExtractor
from tcm_fuzzywiki.llmlite import AzureChatGPTConfig, AzureChatGPTLLM
from tcm_fuzzywiki.models import SourceUnit


def _low_icc_status(config) -> str:
    rows = capability_rows({}, config)
    return next(row["status"] for row in rows if row["capability"] == "Low-ICC uncertainty propagation")


def test_low_icc_capability_stays_mvp_for_bootstrap_priors():
    config = {
        "linguistic_values": {
            "冷痛": {"maps_to": {"cold_property": {"fuzzy_set": "high", "review_status": "pending"}}},
        }
    }
    assert _low_icc_status(config) == "implemented_mvp"
    assert _low_icc_status(None) == "implemented_mvp"


def test_low_icc_capability_lifts_to_implemented_when_fully_expert_reviewed():
    config = {
        "linguistic_values": {
            "冷痛": {"maps_to": {"cold_property": {"fuzzy_set": "high", "review_status": "expert_reviewed"}}},
            "刺痛": {"maps_to": {"blood_stasis_tendency": {"fuzzy_set": "moderate", "review_status": "expert_reviewed"}}},
        }
    }
    assert _low_icc_status(config) == "implemented"


class _FakeResponse:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_azure_adapter_parses_fenced_json(monkeypatch):
    content = "<think>推理...</think>\n```json\n{\"observations\": [{\"feature\": \"pulse\", \"feature_value\": \"脉弦\", \"evidence_text\": \"脉弦\", \"extraction_confidence\": 0.9},]}\n```"
    body = {"choices": [{"message": {"content": content}}]}
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=60: _FakeResponse(body))
    llm = AzureChatGPTLLM(AzureChatGPTConfig(endpoint="https://x", deployment="d", api_key="k"))
    payload = llm.complete_json("system", "user")
    assert payload["observations"][0]["feature_value"] == "脉弦"


def test_legacy_llm_extractor_validates_rows(monkeypatch):
    class StubLLM:
        def complete_json(self, system_prompt, user_prompt):
            return {
                "observations": [
                    {"feature": "syndrome_conclusion", "feature_value": "肾虚证", "evidence_text": "x", "extraction_confidence": 2.0},
                    {"feature": "pulse", "feature_value": "", "evidence_text": "empty value dropped"},
                    {"feature": "pulse", "feature_value": "脉弦", "evidence_text": "脉弦", "extraction_confidence": 0.9},
                ]
            }

    observations = LLMObservationExtractor(StubLLM()).extract([SourceUnit("SRC_1", "Book")])
    assert len(observations) == 2
    # Non-whitelisted feature names are coerced to "other" instead of leaking through.
    assert observations[0].feature == "other"
    assert observations[0].extraction_confidence == 1.0
    assert observations[1].feature_value == "脉弦"

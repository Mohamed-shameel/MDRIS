"""
LLM second opinion (D-36): reply validation and the fusion boundary.
No network: parse_reply is tested directly.
"""

import inspect

import pytest

from ai.fusion import fusion_engine
from ai.llm_vision.service import LlmUnavailable, parse_reply


def test_valid_reply_is_counted():
    out = parse_reply('{"flood_visible": true, "objects": ['
                      '{"class_name": "person_in_water", "description": "a"},'
                      '{"class_name": "vehicle", "description": "b"}], "summary": "s"}')
    assert out["counts"] == {"person_in_water": 1, "person_in_vehicle": 0, "person": 0, "vehicle": 1}


def test_unknown_class_is_rejected_not_guessed():
    with pytest.raises(LlmUnavailable):
        parse_reply('{"objects": [{"class_name": "boat", "description": "x"}]}')


def test_non_json_reply_is_rejected():
    with pytest.raises(LlmUnavailable):
        parse_reply("I see four people.")


def test_fusion_never_imports_the_llm():
    """The LLM second opinion must never feed the priority score."""
    assert "llm" not in inspect.getsource(fusion_engine).lower()

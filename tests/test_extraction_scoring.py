"""
Unit tests for the JSON parsing and scoring logic — the piece of custom
code most likely to silently distort metrics if buggy (e.g. failing to
strip markdown fences and therefore marking every base-model output as
"invalid JSON" even when the content was actually correct).

Run with: pytest tests/test_extraction_scoring.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluate import normalize, parse_model_json


def test_parses_clean_json():
    raw = '{"adverse_events": [{"drug": "atorvastatin", "effect": "rhabdomyolysis"}]}'
    pairs = parse_model_json(raw)
    assert pairs == {("atorvastatin", "rhabdomyolysis")}


def test_parses_json_wrapped_in_markdown_fence():
    raw = '```json\n{"adverse_events": [{"drug": "aspirin", "effect": "GI bleeding"}]}\n```'
    pairs = parse_model_json(raw)
    assert pairs == {("aspirin", "gi bleeding")}


def test_parses_empty_adverse_events():
    raw = '{"adverse_events": []}'
    pairs = parse_model_json(raw)
    assert pairs == set()


def test_returns_none_on_garbage_output():
    raw = "I'm not sure, this sentence doesn't clearly mention a drug."
    assert parse_model_json(raw) is None


def test_returns_none_on_wrong_schema():
    raw = '{"drug": "aspirin", "effect": "bleeding"}'  # missing the adverse_events wrapper
    assert parse_model_json(raw) is None


def test_ignores_malformed_individual_events():
    raw = '{"adverse_events": [{"drug": "aspirin", "effect": "bleeding"}, {"drug": "ibuprofen"}]}'
    pairs = parse_model_json(raw)
    assert pairs == {("aspirin", "bleeding")}  # second entry missing "effect" is dropped, not crashed on


def test_normalize_case_and_whitespace_insensitive():
    assert normalize("  Rhabdomyolysis  ") == normalize("rhabdomyolysis")

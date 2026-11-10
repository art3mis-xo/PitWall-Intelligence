"""The agent eval's rule-based scoring (no LLM calls): number matching and grounding."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import agent_eval as ev  # noqa: E402


def test_number_matches_allow_rounding_and_fraction_of_a_percentage():
    assert ev._matches(62.4, 62.42)      # rounded
    assert ev._matches(0.624, 62.4)      # 62.4 % written as a fraction
    assert ev._matches(0.62, 0.6242)
    assert not ev._matches(10.54, 0.105)  # 0.105 % written as 10.5 %: the bug this must catch


def test_numbers_keep_signs_positions_and_ranges():
    assert ev.numbers("SHAP −0.336 and P4, 62.4%") == [-0.336, 4.0, 62.4]
    assert ev.numbers("seasons 2018-2024") == [2018.0, 2024.0]
    assert ev.numbers("never raced in Formula 1, the F1 series") == []   # series name, not a statistic
    assert ev.numbers("(‑0.228)") == [-0.228]   # non-breaking hyphen as minus, seen in real replies


def test_ungrounded_flags_only_invented_numbers():
    tools = "NOR — Teammate H2H win % (to date): 62.4% (Based on 166 races, latest: 2026 R14)"
    assert ev.ungrounded("Norris wins 62.4% of head-to-heads over 166 races.", tools) == []
    assert ev.ungrounded("Norris wins 71% of head-to-heads from P7.", tools) == [71.0, 7.0]


def test_cases_are_built_from_data_without_network():
    cases = ev.build_cases()
    assert len(cases) == 24
    assert {c["kind"] for c in cases} == {"stats", "predict", "explain", "ambiguous", "out_of_scope"}
    assert all("expect_number" in c for c in cases if c["kind"] == "stats")

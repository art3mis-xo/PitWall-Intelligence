"""LangGraph agent: routing and tools, without calling the LLM."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from pitwall.backend.services import agent


def test_routes_to_tools_when_model_requests_one():
    call = {"name": "race_predictor_tool", "args": {"season": 2026, "round_num": 14}, "id": "1"}
    state = {"messages": [HumanMessage("Who wins?"), AIMessage(content="", tool_calls=[call])]}
    assert agent.should_continue(state) == "tools"


def test_ends_when_model_answers_directly():
    state = {"messages": [HumanMessage("Hi"), AIMessage(content="Hello")]}
    assert agent.should_continue(state) == "end"


def test_ends_on_empty_state():
    assert agent.should_continue({"messages": []}) == "end"


def test_system_prompt_lists_exactly_the_bound_tools():
    for tool in agent.TOOLS:
        assert tool.name in agent.SYSTEM_PROMPT
    assert "search_regulations_tool" not in agent.SYSTEM_PROMPT


def test_race_predictor_tool_returns_top_ten():
    out = agent.race_predictor_tool.invoke({"season": 2026, "round_num": 14})
    assert out.count("\nP") == 10


def test_explain_tool_unknown_driver():
    out = agent.explain_prediction_tool.invoke({"season": 2026, "round_num": 14, "driver_id": "ZZZ"})
    assert out.startswith("No driver called 'ZZZ'")


@pytest.mark.parametrize("query, code", [
    ("NOR", "NOR"), ("nor", "NOR"), ("Lando Norris", "NOR"), ("Norris", "NOR"),
    ("Raikkonen", "RAI"),   # accent-insensitive: Räikkönen
])
def test_resolve_driver_by_code_or_name(query, code):
    assert agent.resolve_driver(query) == (code, "")


def test_resolve_driver_asks_when_ambiguous():
    code, message = agent.resolve_driver("Kimi")
    assert code is None
    assert "RAI" in message and "ANT" in message


def test_stats_tool_accepts_a_driver_name():
    # Regression: "Lando Norris" used to return "No data found"
    by_name = agent.features_for_stats_tool.invoke({"driver_id": "Lando Norris", "metric": "teammate_win_pct"})
    by_code = agent.features_for_stats_tool.invoke({"driver_id": "NOR", "metric": "teammate_win_pct"})
    assert by_name == by_code and by_name.startswith("NOR")


def test_stats_tool_rejects_unknown_metric():
    out = agent.features_for_stats_tool.invoke({"driver_id": "VER", "metric": "lap_record"})
    assert out.startswith("Unknown metric")


def test_stats_tool_returns_value_for_known_driver():
    out = agent.features_for_stats_tool.invoke({"driver_id": "VER", "metric": "teammate_win_pct"})
    assert out.startswith("VER")

"""Agent loop safety, with a fake model that never stops calling tools (no Groq)."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from pitwall.backend.services import agent


class AlwaysCallsTools:
    """Stands in for the tool-bound LLM: every turn asks for another prediction."""
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        call = {"name": "race_predictor_tool", "args": {"season": 2026, "round_num": 14}, "id": f"c{self.calls}"}
        return AIMessage(content="", tool_calls=[call])


class AnswersFromResults:
    """Stands in for the tool-free LLM used once the budget is spent."""
    def invoke(self, messages):
        return AIMessage(content="Final answer from the tool results.")


@pytest.fixture
def looping_model(monkeypatch):
    fake = AlwaysCallsTools()
    monkeypatch.setattr(agent, "llm_with_tools", fake)
    monkeypatch.setattr(agent, "llm", AnswersFromResults())
    return fake


def test_tool_budget_ends_a_runaway_loop_with_an_answer(looping_model):
    out = agent.app.invoke({"messages": [HumanMessage("Who wins?")]})
    assert out["messages"][-1].content == "Final answer from the tool results."
    assert looping_model.calls == agent.MAX_TOOL_ROUNDS


def test_without_the_budget_the_loop_only_stops_with_an_error(looping_model, monkeypatch):
    # The failure mode the budget prevents: the step limit raises and the user gets no answer
    monkeypatch.setattr(agent, "MAX_TOOL_ROUNDS", 10_000)
    with pytest.raises(GraphRecursionError):
        agent.app.invoke({"messages": [HumanMessage("Who wins?")]}, {"recursion_limit": 12})


def test_unknown_driver_suggests_real_names():
    code, message = agent.resolve_driver("Verstapen")
    assert code is None and "Max Verstappen (VER)" in message

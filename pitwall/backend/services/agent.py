from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv, dotenv_values 
from .race_predictor import predict_race, load_features, FEATURE_LABELS
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import operator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv() 

# accessing and printing value
api_key=os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    temperature=0,
    max_tokens=None,
    api_key=api_key,
)

@tool
def race_predictor_tool(season: int, round_num: int) -> str:
    """Predicts the finishing order for a given F1 race."""

    results = predict_race(season, round_num)
    top10 = results[:10]
    summary = "\n".join([
            f"P{r['predicted_rank']}: {r['driver_full_name']} ({r['team_id']}) "
            f"[grid: P{r['grid_position']}]"
            for r in top10
        ])
    return f"Predicted top 10 for {season} Round {round_num}:\n{summary}"

@tool
def explain_prediction_tool(season: int, round_num: int, driver_id: str) -> str:
    """Explains a specific driver's predicted race result."""
    try: 
        results = predict_race(season, round_num)
        driver_result = next(
            (r for r in results if r["driver_id"].upper() == driver_id.upper()),
            None
        )
        if not driver_result:
            return f"No prediction found for driver {driver_id}"
 
        shap = driver_result["shap_values"]
        top_factors = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
 
        factors_text = "\n".join([
            f"  - {FEATURE_LABELS.get(k, k)}: {'+' if v < 0 else '-'}{abs(v):.3f} "
            f"({'favours better finish' if v < 0 else 'hurts finish'})"
            for k, v in top_factors
        ])
 
        return (
            f"SHAP explanation for {driver_result['driver_full_name']} "
            f"(predicted P{driver_result['predicted_rank']}):\n\n"
            f"Key factors driving this prediction:\n{factors_text}\n\n"
            f"Grid position: P{driver_result['grid_position']}\n"
            f"Predicted finish: P{driver_result['predicted_rank']}\n\n"
            f"Use these specific values to generate a grounded natural language explanation. "
            f"Do not invent statistics not listed above."
        )
    except Exception as e:
        return f"Explanation error: {e}"

@tool
def features_for_stats_tool(driver_id: str, metric: str) -> str:
    """Returns a requested driver stat feature from the feature store."""
    # try: 
    #     features = load_features()
    #     feature_list = "\n".join([f"- {FEATURE_LABELS.get(f, f)}" for f in features])
    #     return f"Available features for race stats:\n{feature_list}" 
    # except Exception as e:
    #     return f"Error loading features: {e}"

    try:
        df = load_features()
        driver_rows = df[df["driver_id"].str.upper() == driver_id.upper()]
        if driver_rows.empty:
            return f"No data found for driver {driver_id}"
 
        valid_metrics = [
            "avg_finish_at_circuit", "teammate_win_pct", "career_starts_at_circuit",
            "avg_finish_grid_delta", "rolling_avg_finish_position"
        ]
        if metric not in valid_metrics:
            return f"Unknown metric. Valid options: {', '.join(valid_metrics)}"
 
        latest = driver_rows.sort_values(["season", "round"]).iloc[-1]
        value  = latest.get(metric, "N/A")
        label  = FEATURE_LABELS.get(metric, metric)
 
        return (
            f"{driver_id.upper()} — {label}: {round(float(value), 3) if value != 'N/A' else 'N/A'}\n"
            f"(Based on {len(driver_rows)} races in dataset, latest: "
            f"{int(latest['season'])} R{int(latest['round'])})"
        )
    except Exception as e:
        return f"Feature store query error: {e}"

# phasee 5 development
# @Tool
# def search_regulations_tool(query):


# llm_with_tools = llm.bind_tools([race_predictor_tool, explain_prediction_tool, features_for_stats_tool])
# output = llm_with_tools.invoke("What is 2 times 3?")
# tool_calls = output.tool_calls

#––––––––––––––––– defining graph
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

TOOLS = [race_predictor_tool, explain_prediction_tool, features_for_stats_tool]
llm_with_tools = llm.bind_tools(TOOLS)
tool_node = ToolNode(TOOLS)

SYSTEM_PROMPT = """You are PitWall Intelligence, an expert F1 analyst AI.
 
You have access to four tools:
- race_predictor_tool: predict finishing order for any F1 race
- explain_prediction_tool: get SHAP-grounded explanation for a driver's predicted result
- features_for_stats_tool: look up historical driver statistics
- search_regulations_tool: search F1 sporting regulations
 
Always ground your answers in data from these tools and any explicit race context in the user message. Do not invent statistics, grid positions, pole positions, or race results.
When explaining predictions, cite the specific SHAP values returned by explain_prediction.
Distinguish predicted finishing position from starting grid position. Never describe P4 as pole or front row, and never say pole-to-win unless the supplied grid position is P1.
Be concise, analytical, and use F1 terminology naturally.
When you have enough information to answer, respond directly without calling more tools."""

def agent_node(state:AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)]+state["messages"]
    output = llm_with_tools.invoke(messages)
    return {"messages":[output]}

def should_continue(state: AgentState):
    messages = state.get("messages") or []
    last_message = messages[-1] if messages else None
    if last_message is not None and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent",should_continue,{"tools":"tools",
                                                     "end": END})
graph.add_edge("tools","agent")
app = graph.compile()

async def chat(message: str, history: list[dict] | None = None) -> str:
    messages = []
    if history:
        for h in history:
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            elif h["role"] == "assistant":
                messages.append(AIMessage(content=h["content"]))
    messages.append(HumanMessage(content=message))
    result = app.invoke({"messages": messages})
    last   = result["messages"][-1]
    return last.content if hasattr(last, "content") else str(last)


async def chat_stream(message: str, history: list[dict] | None = None):
    messages = []

    if history:
        for h in history:
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            elif h["role"] == "assistant":
                messages.append(AIMessage(content=h["content"]))

    messages.append(HumanMessage(content=message))

    async for event in app.astream_events({"messages": messages}, version="v1"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

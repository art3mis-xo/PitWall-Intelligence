from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import re
from ..services.agent     import chat, chat_stream
from ..services.race_predictor import predict_race, load_features
from ..services.news      import get_f1_news
from ..services.rate_limit import GroqLimitExceeded, client_ip, limiter

router = APIRouter(prefix="/api/intelligence", tags=["Intelligence"])


def ground_narrative(text: str, grid_position: int | None) -> str:
    """Remove impossible pole/front-row claims from generated explanations."""
    if grid_position == 1:
        return text
    return re.sub(
        r"\b(?:pole-to-win|pole position|front[- ]row)\b",
        "front of the field",
        text,
        flags=re.IGNORECASE,
    )


class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None
    stream:  bool = False


@router.post("/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    try:
        limiter.check(client_ip(request))
    except GroqLimitExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))
    try:
        if req.stream:
            async def generate():
                async for chunk in chat_stream(req.message, req.history):
                    yield chunk
            return StreamingResponse(generate(), media_type="text/plain")
        else:
            response = await chat(req.message, req.history)
            return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/explain/{season}/{round_num}/{driver_id}")
async def explain_endpoint(season: int, round_num: int, driver_id: str, request: Request):
    try:
        predictions   = predict_race(season, round_num)
        driver_result = next(
            (d for d in predictions if d["driver_id"].upper() == driver_id.upper()),
            None
        )
        if not driver_result:
            raise HTTPException(status_code=404, detail=f"Driver {driver_id} not found")

        try:
            limiter.check(client_ip(request))
            narrative = await chat(
                f"Explain the prediction for {driver_result['driver_full_name']} "
                f"predicted P{driver_result['predicted_rank']} in {season} Round {round_num}. "
                f"Their grid position is P{driver_result['grid_position']}. "
                f"Use only these factors: {driver_result['top_factors']}. "
                "In each factor, 'value' is the feature's actual value for this driver and "
                "'shap_value' is its contribution to the model's score (negative = better finish); "
                "never present a shap_value as a time, gap, position or count. "
                "Do not invent statistics or race events. Distinguish predicted finish "
                "from grid position: never call P4 pole or front row, and do not claim "
                "a pole-to-win result unless the grid position is P1. "
                "Return concise Markdown with a short summary, a table of the factors, "
                "and a final explanation grounded in the supplied values."
            )
            narrative = ground_narrative(narrative, driver_result["grid_position"])
        except GroqLimitExceeded as e:
            # The SHAP factors below don't need Groq, so the endpoint still answers
            narrative = f"{e} The factors below come straight from the model."
        except Exception:
            narrative = (
                "Explanation service is unavailable. "
                "Here are the top factors that contributed to this predicted result."
            )

        return {
            "driver_id":      driver_result["driver_id"],
            "driver_full_name": driver_result["driver_full_name"],
            "team_id":        driver_result["team_id"],
            "grid_position":  driver_result["grid_position"],
            "predicted_rank": driver_result["predicted_rank"],
            "shap_values":    driver_result["shap_values"],
            "top_factors":    driver_result["top_factors"],
            "narrative":      narrative,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/news")
async def news_endpoint(limit: int = 8):
    try:
        articles = get_f1_news(max_items=limit)
        return {"articles": articles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/championship/{season}")
async def championship_endpoint(season: int):
    try:
        df = load_features()
        season_df = df[df["season"] == season]
        if season_df.empty:
            raise HTTPException(status_code=404, detail=f"No data for season {season}")

        standings = (
            season_df.groupby(["driver_id", "driver_full_name", "team_id"])["points"]
            .sum()
            .reset_index()
            .sort_values("points", ascending=False)
            .reset_index(drop=True)
        )
        standings["position"] = standings.index + 1
        return {"season": season, "standings": standings.to_dict(orient="records")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
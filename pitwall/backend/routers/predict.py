from fastapi import FastAPI, APIRouter, HTTPException, Request
from pitwall.backend.services.agent import predict_race

router = APIRouter(prefix="/api/predict", tags=["predict"])

@router.get("/{season}/{round_num}")
async def predict(season: int, round_num: int):
    try:
        predictions = predict_race(season, round_num)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

from .routers.telemetry import router as telemetry_router
from .routers.selectors import router as selectors_router
from .routers.predict import router as predict_router
from .routers.intelligence import router as intelligence_router
from .routers.live import router as live_router

app = FastAPI(title="PitWall API", version="2.0.0")


def cors_origins() -> list[str]:
    # Comma-separated, e.g. CORS_ORIGINS=https://pitwall.vercel.app; defaults to the local dev servers
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:4173")
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(telemetry_router)
app.include_router(selectors_router)
app.include_router(predict_router)
app.include_router(intelligence_router)
app.include_router(live_router)

@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}
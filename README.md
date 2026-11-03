# PitWall Intelligence

PitWall Intelligence is an end-to-end Formula 1 analytics platform for race prediction, telemetry exploration, and grounded AI analysis. It combines an XGBoost race predictor and SHAP explanations with FastF1 telemetry, a FastAPI backend, and a React/TypeScript dashboard.

**Status:** active development · data pipeline currently reaches 2026 Round 14 · last reviewed 2026-09-16

## What It Does

- Predicts driver finishing order from engineered race, qualifying, weather, circuit, teammate, and grid-penalty features.
- Exposes SHAP-grounded explanations for individual predictions.
- Explores historical sessions with position changes, lap times, tyre strategy, gear shifts, and multi-driver speed traces.
- Overlays shared circuit corner annotations on speed traces when FastF1 provides circuit metadata.
- Uses a LangGraph agent with Groq for race prediction, explanations, and historical driver statistics.
- Retrieves current F1 news from RSS feeds and exposes live race and championship data.

## Architecture

```text
FastF1 data and telemetry
          |
          v
Python feature pipeline -> data/features.csv -> XGBoost + SHAP
          |                                      |
          +-------------- FastAPI --------------+
                         |
                         +-- telemetry and selector APIs
                         +-- prediction and explanation APIs
                         +-- LangGraph/Groq agent
                         +-- RSS news and live-data APIs
                         |
                         v
                 React + TypeScript + Plotly UI
```

### Implemented

- **ML:** XGBoost finishing-position predictor, SHAP explanations, persisted model artifacts.
- **AI:** LangGraph workflow with Groq and prediction/statistics tools.
- **Backend:** FastAPI, FastF1 session caching, RSS news, live race data.
- **Frontend:** React, TypeScript, Vite, Tailwind CSS, Plotly, responsive race dashboard.
- **Telemetry:** multi-driver position, tyre, gear-shift, and speed-trace visualizations.

### Roadmap

Qdrant hybrid retrieval, LoRA commentary fine-tuning, RAGAS evaluation, Docker packaging, GitHub Actions CI/CD, and Google A2A integration are planned work, not included as completed capabilities in the current source tree.

## Repository Layout

```text
1Build_dataset_base.py       Build the driver-race base dataset
add_grid_penalties.py        Add qualifying-to-grid penalty features
2Feature_engineering.py      Build model-ready features
3Model_training.ipynb        Train and export the XGBoost model
pitwall/backend/              FastAPI app, routers, and services
pitwall/frontend/             React/Vite dashboard
data/features.csv             Model feature store
models/                       Published predictor artifacts
docs/project-writeups.md      Resume and portfolio descriptions
```

Large FastF1 caches, the local virtual environment, frontend dependencies, build output, and secrets are excluded by `.gitignore`.

## Setup

Requirements: Python 3.12+, Node.js 20+, npm, and network access for FastF1/RSS data.

```bash
git clone <your-github-repository-url>
cd FastF1

python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd pitwall/frontend
npm ci
cd ../..
```

Create a local `.env` file. Never commit it:

```dotenv
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-20b
LANGSMITH_API_KEY=your_langsmith_key
```

Run the backend and frontend in separate terminals:

```bash
./venv/bin/python -m uvicorn pitwall.backend.main:app --reload
```

```bash
cd pitwall/frontend
npm run dev
```

The backend is available at `http://127.0.0.1:8000`; the frontend is usually available at `http://localhost:5173`.

## Retrain The Predictor

Run the pipeline from the repository root:

```bash
./venv/bin/python 1Build_dataset_base.py
./venv/bin/python add_grid_penalties.py
./venv/bin/python 2Feature_engineering.py
```

Then open `3Model_training.ipynb` with the project virtual environment and run its cells. The final training cell writes:

```text
models/xgb_race_predictor_final.pkl
models/feature_cols.pkl
```

Restart the backend after retraining because the predictor caches its model and feature store in memory.

## API Surface

- `GET /health`
- `GET /api/selectors/events/{year}`
- `GET /api/selectors/sessions/{year}/{round}`
- `GET /api/selectors/drivers/{year}/{event}/{session}`
- `GET /api/telemetry/position-changes/{year}/{gp}`
- `GET /api/telemetry/tyre-strategy/{year}/{gp}`
- `GET /api/telemetry/gear-shifts/{year}/{gp}`
- `GET /api/telemetry/speed-traces/{year}/{gp}`
- `GET /api/predict/{season}/{round_num}`
- `GET /api/intelligence/news`
- `GET /api/intelligence/explain/{season}/{round_num}/{driver_id}`
- `POST /api/intelligence/chat`
- Live race and standings endpoints under `/api/live`

## Publishing Checklist

Before the first push:

1. Revoke and regenerate any API keys that were ever stored in `.env`.
2. Confirm `.env`, `venv/`, `cache_f1/`, `f1_cache/`, `node_modules/`, and `dist/` are ignored.
3. Decide whether the predictor `.pkl` files should be public; keep them only if the repository is intended to run immediately after cloning.
4. Review `git status --short` and `git diff --cached` before committing.

The repository does not currently include automated tests, Docker packaging, or CI workflows. Those should be added before presenting deployment-readiness claims.

## License And Data

Add a repository license before public release. FastF1, RSS publishers, and any upstream datasets retain their own terms. Do not redistribute cached upstream data unless permitted by the relevant source.

## Project Writeups

See [docs/project-writeups.md](docs/project-writeups.md) for resume and portfolio versions.

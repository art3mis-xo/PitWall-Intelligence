# PitWall Intelligence

PitWall predicts the finishing order of a Formula 1 race from past results, qualifying and grid penalties, and shows which inputs pushed each driver's prediction up or down. A React dashboard puts those predictions next to race charts from FastF1 and a chat assistant that answers by calling the prediction code.

Feature data covers 2018 to 2026 Round 14: 3,766 driver-race rows across 187 races.

## What It Does

- **Race predictor.** An XGBoost regressor predicts each driver's finishing position from 20 features built from race results, qualifying times, grid penalties, team and driver history, and race weather. Drivers are ranked by predicted position.
- **Per-driver explanations.** SHAP values (`shap.TreeExplainer`) show how much each feature moved a driver's prediction; the API labels each factor `improves` or `worsens`.
- **Chat assistant.** A LangGraph tool-calling loop on Groq (default model `openai/gpt-oss-20b`, set by `GROQ_MODEL`) with three tools: race prediction, SHAP explanation, and driver statistics from the feature store. Responses stream to the UI.
- **Race exploration.** Position changes, lap times, tyre stints, gear shifts and speed traces with corner markers, built from FastF1 session data. These charts are for display; telemetry is not a model input.
- **News and standings.** Headlines from four RSS feeds (cached 15 minutes), plus championship standings and last-race results from the Jolpica API.

## How It's Built

```text
FastF1 + Jolpica APIs
        |
        v
1Build_dataset_base.py -> add_grid_penalties.py -> 2Feature_engineering.py
        |
        v
data/features.csv -> 3Model_training.ipynb -> models/*.pkl
                                                  |
FastAPI backend (pitwall/backend) ----------------+
  routers: selectors, telemetry, predict, intelligence (chat, explain, news), live
  services: race_predictor (XGBoost + SHAP), agent (LangGraph + Groq), FastF1 session cache, RSS
        |
        v
React + TypeScript + Vite + Plotly dashboard (pitwall/frontend)
```

- **Data pipeline.** `1Build_dataset_base.py` pulls race results, qualifying times and race weather from FastF1. `add_grid_penalties.py` derives grid penalties by comparing qualifying and grid positions from Jolpica, with retries and a resumable cache. `2Feature_engineering.py` builds the feature table.
- **Model.** `3Model_training.ipynb` backtests season by season (train on earlier seasons, test on the next) and exports the final model and its feature list.
- **Backend.** FastAPI with 17 routes. The model, feature store and SHAP explainer are loaded once and cached; FastF1 sessions are cached in memory, with a per-session async lock so concurrent requests load each session once.
- **Frontend.** React 19, TypeScript, Vite, Tailwind and Plotly. Charts arrive as Plotly JSON from the backend, and driver toggles change trace visibility in the browser without refetching.
- **Tests.** 34 pytest tests (see below).

## Repository Layout

```text
1Build_dataset_base.py     Driver-race base table from FastF1
add_grid_penalties.py      Grid penalties from Jolpica (qualifying vs grid position)
2Feature_engineering.py    Model features -> data/features.csv
3Model_training.ipynb      Season backtest, final training, model export
models/                    Trained model and its feature list
pitwall/backend/           FastAPI app, routers and services
pitwall/frontend/          React dashboard
tests/                     pytest suite (unit + FastF1 integration)
evals/                     LLM eval script (see "Evals" below)
legacy/                    Earlier experiments, not used by the app (see legacy/README.md)
```

## Setup

Developed with Python 3.12 and Node.js 22.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt          # Apple silicon, to run evals/eval_llm.py: requirements-mac.txt

cd pitwall/frontend && npm ci && cd ../..
```

Create a `.env` file in the repository root (see `.env.example`):

```dotenv
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-20b
```

Race telemetry loads from the FastF1 cache in `cache_f1/`; load a session once to populate it, for example:

```bash
./venv/bin/python -c "import fastf1; fastf1.Cache.enable_cache('cache_f1'); fastf1.get_session(2024, 'Monaco Grand Prix', 'R').load()"
```

Run the backend and frontend in separate terminals:

```bash
./venv/bin/python -m uvicorn pitwall.backend.main:app --reload    # http://127.0.0.1:8000
cd pitwall/frontend && npm run dev                                 # http://localhost:5173
```

## Tests

```bash
pytest                          # all 34 tests
pytest -m "not integration"     # 26 unit tests: no network, no Groq calls
```

The unit tests cover the predictor (ranking, SHAP values summing to the prediction, factor directions), the API endpoints with the LLM mocked, and the agent's routing and tools. The 8 integration tests load real sessions through FastF1 and read them from the FastF1 cache.

## Evals

`evals/eval_llm.py` scores a local Llama 3.1 8B (4-bit, via MLX) on two hand-written F1 questions, with keyword checks and an LLM judge. It does not evaluate the Groq agent used by the app.

## Retrain The Predictor

```bash
./venv/bin/python 1Build_dataset_base.py
./venv/bin/python add_grid_penalties.py
./venv/bin/python 2Feature_engineering.py
```

Then run `3Model_training.ipynb`. Its final cell writes `models/xgb_race_predictor_final.pkl` and `models/feature_cols.pkl`. Restart the backend afterwards; it caches the model and feature store in memory.

## API

| Route | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `GET /api/selectors/years` | Seasons offered in the explorer |
| `GET /api/selectors/events/{year}` | Race weekends for a season |
| `GET /api/selectors/sessions/{year}/{round}` | Sessions in a weekend |
| `GET /api/selectors/drivers/{year}/{event}/{session}` | Drivers in a session |
| `GET /api/telemetry/position-changes/{year}/{gp}` | Positions (race) or lap times (other sessions) |
| `GET /api/telemetry/tyre-strategy/{year}/{gp}` | Tyre stints |
| `GET /api/telemetry/gear-shifts/{year}/{gp}` | Fastest-lap gear map |
| `GET /api/telemetry/speed-traces/{year}/{gp}` | Fastest-lap speed with corner markers |
| `GET /api/predict/{season}/{round_num}` | Predicted order with SHAP values |
| `GET /api/intelligence/explain/{season}/{round_num}/{driver_id}` | One driver's factors plus an LLM-written summary |
| `POST /api/intelligence/chat` | Chat assistant (`stream: true` for streaming) |
| `GET /api/intelligence/news` | RSS headlines |
| `GET /api/intelligence/championship/{season}` | Points table from the feature store |
| `GET /api/live/next-race` | Next race on the calendar |
| `GET /api/live/standings/{season}` | Driver and constructor standings (Jolpica) |
| `GET /api/live/last-race/{season}` | Latest race results (Jolpica) |

Telemetry routes return 503 when a session's data can't be loaded.

## Roadmap: Planned, Not Built

- Retrieval over the FIA sporting regulations, with a retrieval evaluation.
- An evaluation of the agent: tool choice and whether answers stick to tool output.

## License And Data

No license has been chosen yet. FastF1, Jolpica and the RSS publishers keep their own terms; cached upstream data is not redistributed here.

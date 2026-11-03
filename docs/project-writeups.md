# Project Writeups

## Resume Version

**PitWall Intelligence | F1 Race Analytics and AI Platform**

- Built an end-to-end Formula 1 analytics platform combining an XGBoost finishing-position predictor, SHAP explanations, FastAPI APIs, and a React/TypeScript dashboard.
- Engineered race, qualifying, weather, tyre-strategy, grid-penalty, telemetry, and circuit features from FastF1 data; trained the predictor on historical race data through 2026 Round 14.
- Added interactive multi-driver telemetry visualizations for position changes, tyre strategy, gear shifts, and speed traces with shared corner annotations and client-side Plotly visibility controls.
- Integrated a LangGraph agent backed by Groq with tools for race prediction, grounded explanations, and historical driver statistics, plus live RSS news aggregation.

## Portfolio Version

PitWall Intelligence is an F1 analytics workspace that connects predictive modeling, telemetry exploration, and grounded AI analysis in one interface. The data pipeline builds driver-race and lap-level datasets from FastF1, adds grid-penalty and engineered historical-performance features, and trains an XGBoost model that predicts finishing order. SHAP values are exposed through the API so users can inspect why a prediction changes for a particular driver.

The `/explore` experience focuses on race understanding rather than a static report. Users can select a season, Grand Prix, session, and drivers, then inspect position changes, tyre stints, gear shifts, and speed traces. Telemetry for selected drivers is precomputed by the backend, while the Plotly frontend changes visibility locally so comparisons remain responsive. The speed-trace view also maps shared circuit corner annotations onto the distance axis when FastF1 provides circuit metadata.

The `/predict` experience adds an AI analysis layer. A LangGraph workflow uses Groq and typed tools to retrieve model predictions, SHAP-grounded explanations, and historical driver metrics without inventing race facts. The project also includes RSS-based F1 news, live standings and race data integrations, and a React dashboard designed for repeated analysis workflows.

## Current Scope

Implemented: FastAPI backend, React/Vite frontend, XGBoost predictor, SHAP explanations, LangGraph/Groq agent, RSS news, selectors, live race data, and interactive telemetry charts.



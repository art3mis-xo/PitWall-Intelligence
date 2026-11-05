# Deploying PitWall

The API runs on **Render** from the repository's `Dockerfile`; the dashboard runs on **Vercel**. Race telemetry charts are switched off on the public deployment (`TELEMETRY_ENABLED=false`): since October 2026, F1 timing data needs an F1TV login, so those charts only work locally with a populated FastF1 cache. Predictions, SHAP explanations, the chat assistant, news and standings work without it.

Deploy in this order, because each side needs the other's URL.

## 1. API on Render

1. In the Render dashboard: **New → Web Service**, connect the GitHub repository `art3mis-xo/PitWall-Intelligence`, branch `main`.
2. **Language / runtime:** Docker. **Dockerfile path:** `./Dockerfile`. **Root directory:** leave empty (repository root).
3. **Instance type:** the backend used about 360 MB of memory locally, so choose an instance with at least 512 MB. Check Render's current plans for price and idle behaviour; free instances may sleep when idle, which makes the first request afterwards slow.
4. **Health check path:** `/api/health`.
5. **Environment variables:**

   | Variable | Value | Required |
   |---|---|---|
   | `GROQ_API_KEY` | your Groq API key | yes |
   | `CORS_ORIGINS` | your Vercel URL, e.g. `https://pitwall-intelligence.vercel.app` (no trailing slash; comma-separate several). Set a placeholder now and update it in step 3. | yes |
   | `GROQ_MODEL` | `openai/gpt-oss-20b` (the default if unset) | no |
   | `GROQ_LIMIT_PER_IP` | AI requests per visitor per window (default `10`) | no |
   | `GROQ_LIMIT_WINDOW_S` | window length in seconds (default `600`) | no |
   | `GROQ_DAILY_CAP` | AI requests per day across all visitors (default `300`) | no |

   `PORT` is set by Render; the image listens on it. `TELEMETRY_ENABLED` is already `false` in the image.
6. Create the service. When it is live, open `https://<your-service>.onrender.com/api/health`; it should return `{"status":"ok","version":"2.0.0"}`. Note the service URL.

## 2. Dashboard on Vercel

1. In Vercel: **Add New → Project**, import the same repository.
2. **Root directory:** `pitwall/frontend`. Vercel detects Vite; the build command is `npm run build` and the output directory is `dist`.
3. **Environment variable:** `VITE_API_URL` = the Render URL from step 1, e.g. `https://<your-service>.onrender.com` (no trailing slash, no `/api`).
4. Deploy. `vercel.json` makes routes like `/predict` load the app instead of a 404.

`VITE_API_URL` is read at build time: if you change it, redeploy.

## 3. Connect them

1. Back on Render, set `CORS_ORIGINS` to the Vercel URL (e.g. `https://pitwall-intelligence.vercel.app`) and save; Render redeploys.
2. Open the Vercel URL. The landing page should show standings and news, and `/predict` should show predictions with SHAP factors.

## Groq quota protection

`/api/intelligence/chat` and `/api/intelligence/explain` call Groq. Both are limited per visitor IP and per day (variables above). Over the limit, chat returns HTTP 429 with a message and explanations fall back to the SHAP factors without the AI-written summary. Limits are held in memory, so they reset when the service restarts, and they assume a single instance.

## Run the deployed setup locally

```bash
docker compose up --build        # dashboard on http://localhost:8080, API on http://localhost:8000
API_PORT=8001 WEB_PORT=8081 docker compose up --build   # if those ports are taken
```

`GROQ_API_KEY` is read from `.env` in the repository root.

## Updating the model

Retrain with `python train.py`, commit `models/`, and push: Render rebuilds the image from `main`. The image installs `requirements-server.txt`, which includes LightGBM but not XGBoost, matching the served model in `models/model_config.json`.

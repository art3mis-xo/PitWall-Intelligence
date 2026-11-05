# PitWall API: FastAPI backend with the trained model and feature store baked in.
#   docker build -t pitwall-api .
#   docker run -p 8000:8000 -e GROQ_API_KEY=... pitwall-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8000 \
    TELEMETRY_ENABLED=false

# libgomp1: OpenMP runtime that LightGBM needs on slim images
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

COPY pitwall/backend pitwall/backend
COPY models/race_predictor.pkl models/feature_cols.pkl models/model_config.json models/
COPY data/features.csv data/driver_race_results.csv data/
# the feature pipeline builds features for races that haven't happened yet
COPY 2Feature_engineering.py ./

RUN useradd --create-home --uid 10001 app \
 && mkdir -p cache_f1 \
 && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '8000'), timeout=4)"

# Render (and most hosts) pass the port to listen on in $PORT
CMD ["sh", "-c", "exec uvicorn pitwall.backend.main:app --host 0.0.0.0 --port ${PORT}"]

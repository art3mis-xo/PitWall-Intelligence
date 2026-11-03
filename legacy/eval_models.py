# evals/eval_models.py
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, top_k_accuracy_score,
    mean_absolute_error, r2_score
)
import fastf1
import json

fastf1.Cache.enable_cache("cache_f1/")


# ── Tyre degradation model eval ───────────────────────────────────────────────

def eval_tyre_deg_model(model, test_races: list[tuple]) -> dict:
    """
    model: your fitted polynomial regression
    test_races: [(year, gp, compound), ...]
    Returns MAE in seconds and R²
    """
    all_true, all_pred = [], []

    for year, gp, compound in test_races:
        session = fastf1.get_session(year, gp, "R")
        session.load(laps=True, telemetry=False, weather=False)

        laps = session.laps[
            (session.laps["Compound"] == compound) &
            (session.laps["IsAccurate"] == True)
        ].copy()
        laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
        laps = laps.dropna(subset=["TyreLife", "LapTimeSeconds"])

        if len(laps) < 5:
            continue

        X = laps["TyreLife"].values.reshape(-1, 1)
        y_true = laps["LapTimeSeconds"].values
        y_pred = model.predict(X)

        all_true.extend(y_true)
        all_pred.extend(y_pred)

    mae = mean_absolute_error(all_true, all_pred)
    r2  = r2_score(all_true, all_pred)

    return {
        "mae_seconds": round(mae, 4),   # target: < 0.5s
        "r2":          round(r2, 4),    # target: > 0.85
        "n_samples":   len(all_true),
    }


# ── Race winner model eval ────────────────────────────────────────────────────

def eval_race_winner_model(model, test_races: list) -> dict:
    """
    Evaluates on held-out 2023-2024 races
    Metrics:
      - Top-1 accuracy: did model predict the actual winner?
      - Top-3 accuracy: was winner in top 3 predictions?
      - Podium accuracy: how many podium finishers in top-3 predictions?
    """
    top1_correct = 0
    top3_correct = 0
    podium_hits  = 0
    total        = 0

    for race_data in test_races:
        # race_data = {"features": [...], "actual_winner": "VER", "actual_podium": [...]}
        probs     = model.predict_proba([race_data["features"]])[0]
        top3_pred = np.argsort(probs)[::-1][:3]
        top3_names = [model.classes_[i] for i in top3_pred]

        if top3_names[0] == race_data["actual_winner"]:
            top1_correct += 1
        if race_data["actual_winner"] in top3_names:
            top3_correct += 1

        # Podium: how many of top 3 predictions were actually on podium
        hits = len(set(top3_names) & set(race_data["actual_podium"]))
        podium_hits += hits
        total += 1

    return {
        "top1_accuracy":    round(top1_correct / total, 3),  # target: > 0.35
        "top3_accuracy":    round(top3_correct / total, 3),  # target: > 0.70
        "avg_podium_hits":  round(podium_hits / (total * 3), 3),
        "n_races":          total,
    }


# ── Pit stop optimizer eval ───────────────────────────────────────────────────

def eval_pit_optimizer(optimizer, test_races: list) -> dict:
    """
    Compare optimizer's recommended strategy vs what teams actually did
    Metric: did the optimizer's strategy save time vs actual?
    """
    results = []

    for race in test_races:
        recommended = optimizer.optimize(race["circuit"], race["year"])
        actual_time = race["actual_race_time"]
        sim_time    = optimizer.simulate_time(
            recommended["stop_laps"],
            recommended["compounds"],
            race["circuit"]
        )
        delta = actual_time - sim_time   # positive = optimizer found better strategy

        results.append({
            "race":          race["name"],
            "recommended":   recommended,
            "actual_time_s": actual_time,
            "sim_time_s":    sim_time,
            "delta_s":       round(delta, 2),
        })

    avg_delta = np.mean([r["delta_s"] for r in results])
    return {
        "avg_time_saving_s": round(avg_delta, 2),
        "results":           results,
    }

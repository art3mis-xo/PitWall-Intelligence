"""Race predictor service: output shape, ranking and SHAP consistency (no network)."""
import pandas as pd
import pytest

from pitwall.backend.services import race_predictor as rp

# 2026 Round 14 is the latest completed race in data/features.csv;
# Round 15 is not in it, so it exercises the upcoming-race path.
COMPLETED = (2026, 14)
UPCOMING = (2026, 15)


@pytest.fixture(scope="module")
def completed():
    return rp.predict_race(*COMPLETED)


def test_one_prediction_per_driver(completed):
    ids = [r["driver_id"] for r in completed]
    assert len(ids) == len(set(ids)) > 0


def test_ranks_follow_predicted_score(completed):
    assert [r["predicted_rank"] for r in completed] == list(range(1, len(completed) + 1))
    scores = [r["predicted_score"] for r in completed]
    assert scores == sorted(scores)


def test_shap_covers_every_model_feature(completed):
    feat_cols = rp.load_feature_cols()
    for r in completed:
        assert list(r["shap_values"]) == feat_cols


def test_shap_values_add_up_to_prediction(completed):
    # TreeSHAP is exact: base value + sum of contributions = model output.
    # Tolerance covers rounding (score and each SHAP value to 4 dp).
    base = rp.shap_base_value()
    for r in completed:
        assert base + sum(r["shap_values"].values()) == pytest.approx(r["predicted_score"], abs=0.01)


def test_top_factors_are_largest_by_magnitude(completed):
    for r in completed:
        top = [f["feature"] for f in r["top_factors"]]
        by_size = sorted(r["shap_values"], key=lambda k: abs(r["shap_values"][k]), reverse=True)
        assert top == by_size[:5]


def test_factor_value_is_the_feature_not_the_shap_value(completed):
    # Regression: 'value' used to repeat the SHAP value, and the LLM narrated it as a real gap/time
    df = rp.load_features()
    race = df[(df["season"] == COMPLETED[0]) & (df["round"] == COMPLETED[1])].set_index("driver_id")
    for r in completed:
        for f in r["top_factors"]:
            actual = race.loc[r["driver_id"], f["feature"]]
            if pd.isna(actual):
                assert f["value"] is None
            else:
                assert f["value"] == pytest.approx(float(actual), abs=1e-4)
            assert f["shap_value"] == r["shap_values"][f["feature"]]


def test_direction_matches_shap_sign(completed):
    for r in completed:
        for f in r["top_factors"]:
            assert f["direction"] == ("improves" if f["shap_value"] < 0 else "worsens")


def test_upcoming_race_uses_active_grid_only():
    results = rp.predict_race(*UPCOMING)
    assert {r["driver_id"] for r in results} <= set(rp.ACTIVE_DRIVERS_2026)
    assert len(results) > 0


def test_upcoming_race_does_not_present_an_old_grid():
    # Regression: the previous race's grid used to be shown as this race's grid
    results = rp.predict_race(*UPCOMING)
    assert all(r["pre_qualifying"] and r["grid_position"] is None for r in results)
    for r in results:
        for f in r["top_factors"]:
            assert f["estimated"] == (f["feature"] in rp.ESTIMATED_INPUTS)


def test_upcoming_race_circuit_features_use_the_actual_circuit():
    # Regression: circuit history used to come from the previous race's circuit
    rows = rp._upcoming_race_rows(*UPCOMING).set_index("driver_id")
    base = pd.read_csv(rp.BASE_RESULTS_PATH)
    past = base[(base["driver_id"] == "VER") & (base["circuit_id"] == "Azerbaijan Grand Prix")]
    assert rows.loc["VER", "circuit_id"] == "Azerbaijan Grand Prix"
    assert rows.loc["VER", "career_starts_at_circuit"] == len(past)
    assert rows.loc["VER", "avg_finish_at_circuit"] == pytest.approx(past["finish_position"].mean())


def test_upcoming_race_quali_inputs_are_recent_averages():
    rows = rp._upcoming_race_rows(*UPCOMING).set_index("driver_id")
    history = rp.load_features()
    last3 = history[history["driver_id"] == "NOR"].sort_values(["season", "round"]).tail(3)
    assert rows.loc["NOR", "quali_position"] == pytest.approx(last3["quali_position"].mean())
    assert rows.loc["NOR", "grid_position"] == rows.loc["NOR", "quali_position"]

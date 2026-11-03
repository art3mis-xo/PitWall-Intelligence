"""Race predictor service: output shape, ranking and SHAP consistency (no network)."""
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


def test_ranks_follow_predicted_position(completed):
    assert [r["predicted_rank"] for r in completed] == list(range(1, len(completed) + 1))
    positions = [r["predicted_position"] for r in completed]
    assert positions == sorted(positions)


def test_shap_covers_every_model_feature(completed):
    feat_cols = rp.load_feature_cols()
    for r in completed:
        assert list(r["shap_values"]) == feat_cols


def test_shap_values_add_up_to_prediction(completed):
    # TreeSHAP is exact: base value + sum of contributions = model output.
    # Tolerance covers rounding (prediction to 2 dp, each SHAP value to 4 dp).
    base = float(rp.load_explainer().expected_value)
    for r in completed:
        assert base + sum(r["shap_values"].values()) == pytest.approx(r["predicted_position"], abs=0.01)


def test_top_factors_are_largest_by_magnitude(completed):
    for r in completed:
        top = [f["feature"] for f in r["top_factors"]]
        by_size = sorted(r["shap_values"], key=lambda k: abs(r["shap_values"][k]), reverse=True)
        assert top == by_size[:5]


def test_direction_matches_shap_sign(completed):
    for r in completed:
        for f in r["top_factors"]:
            assert f["direction"] == ("improves" if f["shap_value"] < 0 else "worsens")


def test_upcoming_race_uses_active_grid_only():
    results = rp.predict_race(*UPCOMING)
    assert {r["driver_id"] for r in results} <= set(rp.ACTIVE_DRIVERS_2026)
    assert len(results) > 0

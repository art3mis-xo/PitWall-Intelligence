"""Features must use only information available before the race starts.

Scrambling the results of race t (finishing order, points, status, rain, weather)
must leave every feature of race t and of all earlier races unchanged.
"""
import importlib

import numpy as np
import pandas as pd
import pytest

fe = importlib.import_module("2Feature_engineering")

RACE_T = (2023, 10)
# Raw columns that are race outcomes or race-time measurements: changed on purpose
OUTCOMES = ["finish_position", "points", "classified_status", "race_had_rain",
            "race_avg_air_temp", "race_avg_track_temp"]


@pytest.fixture(scope="module")
def original_and_scrambled():
    base = fe.load_base()
    scrambled = base.copy()
    in_t = (scrambled["season"] == RACE_T[0]) & (scrambled["round"] == RACE_T[1])
    n = in_t.sum()
    scrambled.loc[in_t, "finish_position"] = np.arange(n, 0, -1)   # reverse the order
    scrambled.loc[in_t, "points"] = np.arange(n)
    scrambled.loc[in_t, "classified_status"] = "Retired"
    scrambled.loc[in_t, "race_had_rain"] = ~scrambled.loc[in_t, "race_had_rain"].astype(bool)
    scrambled.loc[in_t, ["race_avg_air_temp", "race_avg_track_temp"]] += 15
    return fe.build_features(base), fe.build_features(scrambled)


def _up_to_t(df):
    return (df["season"] < RACE_T[0]) | ((df["season"] == RACE_T[0]) & (df["round"] <= RACE_T[1]))


def test_features_of_race_t_and_earlier_unchanged(original_and_scrambled):
    original, scrambled = original_and_scrambled
    feature_cols = [c for c in original.columns if c not in OUTCOMES]
    mask = _up_to_t(original)
    pd.testing.assert_frame_equal(
        original.loc[mask, feature_cols].reset_index(drop=True),
        scrambled.loc[mask, feature_cols].reset_index(drop=True),
    )


def test_later_races_do_see_the_change(original_and_scrambled):
    # Guards against a vacuous pass: history-based features after race t must move.
    original, scrambled = original_and_scrambled
    later = ~_up_to_t(original)
    for col in ["driver_avg_finish_last3", "team_avg_finish_last3", "teammate_wins", "circuit_wet_race_frequency"]:
        assert not original.loc[later, col].equals(scrambled.loc[later, col]), col

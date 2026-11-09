from copy import deepcopy

import pandas as pd
import pytest

from mpl_predictor.analysis.season_results import build_playoff_match_comparison
from mpl_predictor.config import get_project_paths
from mpl_predictor.dashboard import (
    PLAYOFF_COMPARISON_STYLES,
    _archived_playoff_match_comparison,
    build_playoff_bracket_figure,
    build_playoff_comparison_table,
    load_dashboard_data,
)


def example_matches():
    prediction = pd.DataFrame(
        [
            {
                "match_id": "play_in_1",
                "team_a_id": "AE",
                "team_b_id": "ONIC",
                "winner_team_id": "ONIC",
                "outcome_type": "prediction",
            }
        ]
    )
    actual = pd.DataFrame(
        [
            {
                "match_id": "play_in_1",
                "team_a_id": "AE",
                "team_b_id": "ONIC",
                "team_a_score": 3,
                "team_b_score": 1,
                "winner_team_id": "AE",
                "outcome_type": "actual",
            }
        ]
    )
    return prediction, actual


@pytest.mark.parametrize(
    "case, expected",
    [
        ("same_winner", "same_winner"),
        ("different_winner", "different_winner"),
        ("reversed_sides", "same_winner"),
        ("different_pair_same_winner", "different_pairing"),
        ("future_prediction", "pending_result"),
        ("incomplete_score", "pending_result"),
        ("known_result_not_prediction", "prediction_unavailable"),
    ],
)
def test_status_requires_same_pair_and_a_genuine_prediction(case, expected):
    predicted, actual = example_matches()
    if case in {"same_winner", "reversed_sides", "different_pair_same_winner"}:
        predicted.loc[0, "winner_team_id"] = "AE"
    if case == "reversed_sides":
        actual.loc[0, ["team_a_id", "team_b_id", "team_a_score", "team_b_score"]] = (
            "ONIC",
            "AE",
            1,
            3,
        )
    elif case == "different_pair_same_winner":
        actual.loc[0, "team_b_id"] = "BTR"
    elif case == "future_prediction":
        actual.loc[0, "outcome_type"] = "prediction"
    elif case == "incomplete_score":
        actual.loc[0, "team_b_score"] = None
    elif case == "known_result_not_prediction":
        predicted.loc[0, "outcome_type"] = "actual"
    comparison = build_playoff_match_comparison(predicted, actual)
    assert comparison.iloc[0]["comparison_status"] == expected
    if expected == "pending_result":
        assert comparison.filter(regex="^actual_").isna().all().all()
    if expected == "prediction_unavailable":
        assert comparison.filter(regex="^predicted_").isna().all().all()


def test_comparison_preserves_inputs_and_handles_missing_rows():
    predicted, actual = example_matches()
    old_prediction, old_actual = predicted.copy(deep=True), actual.copy(deep=True)
    build_playoff_match_comparison(predicted, actual)
    pd.testing.assert_frame_equal(predicted, old_prediction)
    pd.testing.assert_frame_equal(actual, old_actual)
    assert (
        build_playoff_match_comparison(predicted, pd.DataFrame()).iloc[0]["comparison_status"]
        == "pending_result"
    )
    assert (
        build_playoff_match_comparison(pd.DataFrame(), actual).iloc[0]["comparison_status"]
        == "prediction_unavailable"
    )
    assert build_playoff_match_comparison(pd.DataFrame(), pd.DataFrame()).empty


@pytest.mark.parametrize("side", ["prediction", "actual"])
def test_duplicate_bracket_slots_are_rejected(side):
    predicted, actual = example_matches()
    if side == "prediction":
        predicted = pd.concat([predicted, predicted])
    else:
        actual = pd.concat([actual, actual])
    with pytest.raises(ValueError, match="unique, non-null match IDs"):
        build_playoff_match_comparison(predicted, actual)


@pytest.fixture(scope="module")
def playoff_data():
    data, missing = load_dashboard_data(get_project_paths())
    assert not missing
    return data


def test_original_scenario_comparison_is_not_reported_as_pre_match_accuracy(playoff_data):
    comparison = _archived_playoff_match_comparison(playoff_data)
    assert comparison.comparison_status.value_counts().to_dict() == {
        "different_pairing": 5,
        "different_winner": 2,
        "same_winner": 1,
    }
    rows = comparison.set_index("match_id")
    assert rows.loc["play_in_1", "comparison_status"] == "different_winner"
    assert rows.loc["play_in_2", "comparison_status"] == "same_winner"
    assert rows.loc["lower_final", "comparison_status"] == "different_winner"
    assert rows.loc["grand_final", "comparison_status"] == "different_pairing"
    table = build_playoff_comparison_table(comparison).set_index("Babak")
    assert table.loc["Final bawah", "Prediksi awal"] == "BTR vs TLID → TLID"
    assert table.loc["Final bawah", "Hasil aktual"] == "TLID 3-4 BTR"


@pytest.mark.parametrize("view", ["prediction", "actual"])
def test_comparison_cards_show_both_outcomes_without_crossing_connector_lines(playoff_data, view):
    data = deepcopy(playoff_data)
    comparison = _archived_playoff_match_comparison(data)
    seeds = data["preplayoff_teams"][["seed", "team_id"]].rename(columns={"seed": "projected_seed"})
    seeds["projected_seed_probability"] = 1.0
    config = data["simulation_config"]
    scenario = data["playoff_matches"]
    if view == "prediction":
        config = data["historical_simulation_config"]
        scenario = pd.DataFrame(data["preplayoff_report"]["champion_paths"]["TLID"]["matches"])
    figure = build_playoff_bracket_figure(seeds, config, scenario, comparison=comparison)
    annotations = {item.name: item for item in figure.layout.annotations if item.name}
    for row in comparison.itertuples():
        label, color = PLAYOFF_COMPARISON_STYLES[row.comparison_status]
        status = annotations[f"comparison-status-{row.match_id}"]
        assert label in status.text
        assert status.font.color == color
        assert f"comparison-prediction-{row.match_id}" in annotations
        assert f"comparison-actual-{row.match_id}" in annotations
    assert annotations["comparison-actual-lower_final"].text == "Hasil: TLID 3-4 BTR"
    shapes = figure.layout.shapes
    cards = [shape for shape in shapes if shape.name and shape.name.startswith("match-")]
    tolerance = 1e-9
    for line in shapes:
        if not line.name or not line.name.startswith(("connector-", "transfer-")):
            continue
        for card in cards:
            if line.x0 == line.x1:
                crosses = card.x0 + tolerance < line.x0 < card.x1 - tolerance and max(
                    min(line.y0, line.y1), card.y0 + tolerance
                ) < min(max(line.y0, line.y1), card.y1 - tolerance)
            else:
                crosses = card.y0 + tolerance < line.y0 < card.y1 - tolerance and max(
                    min(line.x0, line.x1), card.x0 + tolerance
                ) < min(max(line.x0, line.x1), card.x1 - tolerance)
            assert not crosses, f"{line.name} crosses {card.name}"
    for card in cards:
        assert card.y0 > figure.layout.yaxis.range[0]

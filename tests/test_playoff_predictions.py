from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from mpl_predictor.analysis.common import load_canonical_tables
from mpl_predictor.data.playoffs import parse_playoff_schedule_html
from mpl_predictor.features.snapshots import load_feature_config
from mpl_predictor.models.final import train_final_match_model
from mpl_predictor.models.playoffs import (
    build_playoff_forecast,
    load_playoff_schedule,
    write_playoff_forecast,
)
from mpl_predictor.models.season18_predictions import load_live_season_tables
from mpl_predictor.models.simulation import load_simulation_config
from mpl_predictor.models.walk_forward import load_model_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inputs():
    tables = load_canonical_tables(ROOT / "data/processed/canonical")
    teams, _, regular = load_live_season_tables(ROOT / "data/season18")
    archive = ROOT / "data/season18/playoff_prediction_snapshots/2026-10-19"
    playoff = load_playoff_schedule(archive / "source_schedule.csv")
    features = load_feature_config(ROOT / "config/feature_config.json")
    artifact = train_final_match_model(
        pd.read_parquet(ROOT / "data/processed/features/match_features.parquet"),
        load_model_config(ROOT / "config/model_config.json"),
    )
    config = load_simulation_config(archive / "simulation_config.json")
    return tables, teams, regular, playoff, features, artifact, config


def forecast(inputs, schedule=None, cutoff=date(2026, 10, 19)):
    tables, teams, regular, playoff, features, artifact, config = inputs
    return build_playoff_forecast(
        tables,
        teams,
        regular,
        playoff if schedule is None else schedule,
        features,
        artifact,
        config,
        cutoff,
    )


@pytest.fixture(scope="module")
def initial_forecast(inputs):
    return forecast(inputs)


def mark_completed(schedule, match_id, a, b, winner):
    result = schedule.copy()
    mask = result.match_id.eq(match_id)
    wins = int(result.loc[mask, "best_of"].iloc[0]) // 2 + 1
    result.loc[mask, "team_a_id"] = a
    result.loc[mask, "team_b_id"] = b
    result.loc[mask, "team_a_score"] = wins if winner == a else 0
    result.loc[mask, "team_b_score"] = wins if winner == b else 0
    result.loc[mask, "winner_team_id"] = winner
    result.loc[mask, "status"] = "completed"
    result.loc[mask, "observed_at"] = str(result.loc[mask, "scheduled_at"].iloc[0].date())
    return result


def test_exact_bracket_covers_all_paths_and_normalizes_probabilities(initial_forecast):
    frames, report = initial_forecast
    assert report["terminal_scenario_count"] == 256
    assert report["completed_playoff_match_count"] == 0
    assert len(frames["teams"]) == 6
    assert len(frames["matches"]) == 8
    assert frames["teams"].champion_probability.sum() == pytest.approx(1)
    assert frames["teams"].grand_final_probability.sum() == pytest.approx(2)
    assert frames["matchups"].groupby("match_id").matchup_probability.sum().to_numpy() == (
        pytest.approx([1.0] * 8)
    )
    assert frames["matchups"].team_a_win_probability.between(0, 1).all()
    assert frames["matches"].outcome_type.eq("prediction").all()
    assert frames["matches"][["team_a_score", "team_b_score"]].isna().all().all()
    for team_id, path in report["champion_paths"].items():
        assert path["matches"][-1]["winner_team_id"] == team_id
        assert 0 < path["conditional_probability"] <= 1


def test_actual_elimination_removes_team_and_routes_winner(inputs):
    schedule = mark_completed(inputs[3], "play_in_1", "AE", "ONIC", "AE")
    frames, report = forecast(inputs, schedule, date(2026, 11, 4))
    assert report["completed_playoff_match_count"] == 1
    assert report["terminal_scenario_count"] == 128
    assert frames["teams"].set_index("team_id").loc["ONIC", "champion_probability"] == 0
    semifinal = frames["matchups"].loc[frames["matchups"].match_id.eq("upper_semifinal_1")]
    assert semifinal.team_b_id.eq("AE").all()
    assert frames["matches"].set_index("match_id").loc["play_in_1", "outcome_type"] == "actual"


def test_future_playoff_result_does_not_change_an_earlier_forecast(inputs, initial_forecast):
    schedule = mark_completed(inputs[3], "play_in_1", "AE", "ONIC", "AE")
    earlier, earlier_report = forecast(inputs, schedule)
    original, report = initial_forecast
    assert earlier_report["completed_playoff_match_count"] == 0
    assert earlier_report["terminal_scenario_count"] == report["terminal_scenario_count"]
    for key in original:
        pd.testing.assert_frame_equal(original[key], earlier[key])


def test_completed_grand_final_confirms_champion(inputs, initial_forecast):
    frames, _ = initial_forecast
    schedule = inputs[3]
    for row in frames["matches"].itertuples():
        schedule = mark_completed(
            schedule, row.match_id, row.team_a_id, row.team_b_id, row.winner_team_id
        )
    final, report = forecast(inputs, schedule, date(2026, 11, 8))
    champion = frames["matches"].iloc[-1].winner_team_id
    assert report["confirmed_champion"] == champion
    assert report["terminal_scenario_count"] == 1
    assert report["remaining_playoff_match_count"] == 0
    assert final["teams"].set_index("team_id").loc[champion, "champion_probability"] == 1
    assert final["matches"].outcome_type.eq("actual").all()
    assert final["matches"].team_a_win_probability.isna().all()


@pytest.mark.parametrize("issue", ["invalid_score", "wrong_participant", "missing_parent"])
def test_invalid_actual_results_are_rejected(inputs, issue):
    schedule = mark_completed(inputs[3], "play_in_1", "AE", "ONIC", "AE")
    if issue == "invalid_score":
        schedule.loc[schedule.match_id.eq("play_in_1"), "team_b_score"] = 3
    elif issue == "wrong_participant":
        schedule.loc[schedule.match_id.eq("play_in_1"), "team_b_id"] = "TLID"
    else:
        schedule = mark_completed(inputs[3], "upper_semifinal_1", "TLID", "AE", "TLID")
    with pytest.raises(ValueError):
        forecast(inputs, schedule, date(2026, 11, 5))


def test_official_parser_preserves_tbd_and_uses_configured_bo(inputs):
    config = inputs[-1]
    cards = []
    for position in range(8):
        cards.append(f"""<div class="match position-relative">
<div class="team1"><span class="name">TBD</span></div><div class="score"></div>
<div class="time">13:00</div><div class="score"></div>
<div class="team2"><span class="name">TBD</span></div>
<div id="modal2-{1108 + position}"></div></div>""")
    html = '<div id="playoffs"><div class="match date">4 November 2026</div>'
    html += "".join(cards) + "</div>"
    parsed = parse_playoff_schedule_html(html, date(2026, 10, 19), config)
    assert parsed[["team_a_id", "team_b_id", "team_a_score", "team_b_score"]].isna().all().all()
    assert parsed.status.eq("scheduled").all()
    assert parsed.best_of.tolist() == [5, 5, 5, 5, 5, 5, 7, 7]


def test_forecast_outputs_preserve_schedule_and_archive(initial_forecast, tmp_path):
    frames, report = initial_forecast
    write_playoff_forecast(
        frames, report, tmp_path / "predictions", tmp_path / "report.json", tmp_path / "archive"
    )
    for key, frame in frames.items():
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / "predictions" / f"season18_playoff_{key}.parquet"), frame
        )
        assert (tmp_path / "archive" / f"{key}.csv").exists()
    assert (tmp_path / "archive/report.json").exists()
    assert (tmp_path / "archive/simulation_config.json").exists()


def test_official_final_results_confirm_btr_and_correct_semifinal_routing(inputs):
    tables, teams, regular, _, features, artifact, _ = inputs
    schedule = load_playoff_schedule(ROOT / "data/season18/playoff_schedule_results.csv")
    config = load_simulation_config(ROOT / "config/simulation_config.json")
    frames, report = build_playoff_forecast(
        tables, teams, regular, schedule, features, artifact, config, date(2026, 11, 9)
    )
    assert report["confirmed_champion"] == "BTR"
    assert report["completed_playoff_match_count"] == 8
    assert report["remaining_playoff_match_count"] == 0
    assert report["terminal_scenario_count"] == 1
    assert report["format_confirmation"]["status"] == "official"
    assert frames["teams"].set_index("team_id").loc["BTR", "champion_probability"] == 1
    assert frames["matches"].outcome_type.eq("actual").all()
    assert frames["matches"].team_a_win_probability.isna().all()
    actual = frames["matches"].set_index("match_id")
    assert actual.loc["upper_semifinal_1", "team_a_id"] == "NAVI"
    assert actual.loc["upper_semifinal_2", "team_a_id"] == "TLID"
    assert actual.loc["grand_final", ["team_a_score", "team_b_score"]].tolist() == [3, 4]


def test_forecast_archive_keeps_source_schedule(initial_forecast, inputs, tmp_path):
    frames, report = initial_forecast
    write_playoff_forecast(
        frames,
        report,
        tmp_path / "predictions",
        tmp_path / "report.json",
        tmp_path / "archive",
        source_schedule=inputs[3],
    )
    pd.testing.assert_frame_equal(
        load_playoff_schedule(tmp_path / "archive/source_schedule.csv"), inputs[3]
    )

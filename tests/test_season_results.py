import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from mpl_predictor.analysis.season_results import (
    build_championship_comparison,
    load_preplayoff_archive,
)

ROOT = Path(__file__).resolve().parents[1]
SEASON_DIR = ROOT / "data/season18"


def final_matches():
    return pd.read_parquet(ROOT / "data/processed/predictions/season18_playoff_matches.parquet")


def test_archive_selection_never_uses_post_result_certainty():
    teams, report = load_preplayoff_archive(SEASON_DIR, final_matches(), season=18)
    assert report["as_of"] == "2026-10-19"
    assert teams["champion_probability"].max() < 0.3
    assert teams.set_index("team_id").loc["BTR", "champion_probability"] == pytest.approx(
        0.12413999407165649
    )


def test_comparison_preserves_original_forecast_and_labels_actual_finishes():
    teams, _ = load_preplayoff_archive(SEASON_DIR, final_matches(), season=18)
    comparison = build_championship_comparison(teams, final_matches()).set_index("team_id")
    original = teams.set_index("team_id")
    pd.testing.assert_series_equal(
        comparison.champion_probability.sort_index(), original.champion_probability.sort_index()
    )
    assert comparison["actual_result"].to_dict() == {
        "TLID": "Peringkat 3",
        "ONIC": "Peringkat 5-6",
        "NAVI": "Runner-up",
        "BTR": "Juara",
        "EVOS": "Peringkat 5-6",
        "AE": "Peringkat 4",
    }
    assert comparison.loc["BTR", "forecast_rank"] == 4


def test_predicted_scenarios_are_not_treated_as_observed_eliminations():
    teams, _ = load_preplayoff_archive(SEASON_DIR, final_matches(), season=18)
    predicted = final_matches().assign(outcome_type="prediction")
    comparison = build_championship_comparison(teams, predicted)
    assert comparison.actual_result.eq("Belum tereliminasi").all()


@pytest.mark.parametrize("issue", ["invalid_probability", "after_start", "known_result", "missing"])
def test_invalid_or_unavailable_archive_is_not_used_for_evaluation(tmp_path, issue):
    if issue == "missing":
        assert load_preplayoff_archive(tmp_path, final_matches(), season=18) is None
        return
    archive = tmp_path / "playoff_prediction_snapshots/2026-10-19"
    shutil.copytree(SEASON_DIR / "playoff_prediction_snapshots/2026-10-19", archive)
    if issue == "invalid_probability":
        teams = pd.read_csv(archive / "teams.csv")
        teams.loc[0, "champion_probability"] = 1.0
        teams.to_csv(archive / "teams.csv", index=False)
    else:
        report = json.loads((archive / "report.json").read_text())
        if issue == "after_start":
            report["as_of"] = "2026-11-09"
        else:
            report["completed_playoff_match_count"] = 1
        (archive / "report.json").write_text(json.dumps(report))
    assert load_preplayoff_archive(tmp_path, final_matches(), season=18) is None

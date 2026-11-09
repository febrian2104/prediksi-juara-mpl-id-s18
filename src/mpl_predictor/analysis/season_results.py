"""Compare archived, pre-playoff forecasts with observed tournament outcomes."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_preplayoff_archive(
    season_dir: Path, playoff_matches: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Choose the latest archive strictly before the first playoff match day.

    Post-result certainty must never be selected as a historical prediction. Invalid
    or incomplete archives are skipped, without fabricating replacement forecasts.
    """
    if playoff_matches.empty:
        return None
    first_day = (
        pd.to_datetime(playoff_matches["scheduled_at"], utc=True)
        .dt.tz_convert("Asia/Jakarta")
        .dt.date.min()
    )
    archive_root = season_dir / "playoff_prediction_snapshots"
    for report_path in sorted(archive_root.glob("*/report.json"), reverse=True):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            cutoff = date.fromisoformat(report["as_of"])
            if (
                int(report["season"]) != season
                or cutoff >= first_day
                or report["completed_playoff_match_count"] != 0
                or report.get("confirmed_champion") is not None
                or report_path.parent.name != cutoff.isoformat()
            ):
                continue
            teams = pd.read_csv(report_path.parent / "teams.csv")
            if (
                teams.empty
                or teams["team_id"].duplicated().any()
                or not teams["season"].eq(season).all()
                or not teams["as_of"].eq(cutoff.isoformat()).all()
                or not teams[["champion_probability", "grand_final_probability"]]
                .apply(lambda column: column.between(0, 1 + 1e-9))
                .all()
                .all()
                or not np.isclose(teams["champion_probability"].sum(), 1, atol=1e-9, rtol=0)
                or not np.isclose(teams["grand_final_probability"].sum(), 2, atol=1e-9, rtol=0)
            ):
                continue
        except (OSError, ValueError, TypeError, KeyError):
            continue
        return teams, report
    return None


def build_championship_comparison(
    archived_teams: pd.DataFrame, playoff_matches: pd.DataFrame
) -> pd.DataFrame:
    """Keep original probabilities untouched and label only completed eliminations."""
    comparison = (
        archived_teams.sort_values(["champion_probability", "team_id"], ascending=[False, True])
        .reset_index(drop=True)
        .copy()
    )
    comparison["forecast_rank"] = (
        comparison["champion_probability"].rank(method="min", ascending=False).astype(int)
    )
    outcomes: dict[str, str] = {}
    elimination_places = {
        "play_ins": "Peringkat 5-6",
        "lower_semifinal": "Peringkat 4",
        "lower_final": "Peringkat 3",
        "grand_final": "Runner-up",
    }
    for row in playoff_matches.loc[playoff_matches["outcome_type"].eq("actual")].itertuples():
        if row.round in elimination_places:
            outcomes[str(row.loser_team_id)] = elimination_places[row.round]
        if row.round == "grand_final":
            outcomes[str(row.winner_team_id)] = "Juara"
    comparison["actual_result"] = comparison["team_id"].map(outcomes).fillna("Belum tereliminasi")
    return comparison


def build_playoff_match_comparison(
    predictions: pd.DataFrame, actuals: pd.DataFrame
) -> pd.DataFrame:
    """Compare an archived scenario, not pre-match accuracy, against observed results.

    Match IDs identify bracket slots, not guaranteed pairings. Winner comparisons
    are meaningful only when both teams match, regardless of their displayed side.
    """
    lookups = []
    for frame in (predictions, actuals):
        if frame.empty:
            lookups.append({})
            continue
        if (
            "match_id" not in frame
            or frame["match_id"].isna().any()
            or frame["match_id"].duplicated().any()
        ):
            raise ValueError("Playoff comparison requires unique, non-null match IDs.")
        lookups.append(frame.set_index("match_id").to_dict(orient="index"))
    predicted_lookup, actual_lookup = lookups
    records = []
    for match_id in dict.fromkeys([*predicted_lookup, *actual_lookup]):
        predicted = predicted_lookup.get(match_id, {})
        actual = actual_lookup.get(match_id, {})
        prediction_available = (
            predicted.get("outcome_type") == "prediction"
            and all(
                pd.notna(predicted.get(key)) for key in ("team_a_id", "team_b_id", "winner_team_id")
            )
            and predicted.get("winner_team_id")
            in (predicted.get("team_a_id"), predicted.get("team_b_id"))
        )
        actual_available = actual.get("outcome_type") == "actual" and all(
            pd.notna(actual.get(key))
            for key in ("team_a_id", "team_b_id", "team_a_score", "team_b_score", "winner_team_id")
        )
        if not prediction_available:
            status = "prediction_unavailable"
        elif not actual_available:
            status = "pending_result"
        elif {predicted["team_a_id"], predicted["team_b_id"]} != {
            actual["team_a_id"],
            actual["team_b_id"],
        }:
            status = "different_pairing"
        elif predicted["winner_team_id"] == actual["winner_team_id"]:
            status = "same_winner"
        else:
            status = "different_winner"
        record = {"match_id": match_id, "comparison_status": status}
        for prefix, source, available in (
            ("predicted", predicted, prediction_available),
            ("actual", actual, actual_available),
        ):
            fields = ["team_a_id", "team_b_id", "winner_team_id"]
            if prefix == "actual":
                fields += ["team_a_score", "team_b_score"]
            for key in fields:
                record[f"{prefix}_{key}"] = source.get(key) if available else None
        records.append(record)
    return pd.DataFrame.from_records(
        records,
        columns=[
            "match_id",
            "comparison_status",
            "predicted_team_a_id",
            "predicted_team_b_id",
            "predicted_winner_team_id",
            "actual_team_a_id",
            "actual_team_b_id",
            "actual_winner_team_id",
            "actual_team_a_score",
            "actual_team_b_score",
        ],
    )

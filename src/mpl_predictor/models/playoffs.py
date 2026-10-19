from collections import defaultdict
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from mpl_predictor.analysis.common import dataframe_records, write_json
from mpl_predictor.features.snapshots import _append_match_history
from mpl_predictor.models.simulation import (
    _current_standings,
    _naive_date,
    _pair_probability_matrix,
    build_live_match_probabilities,
)
from mpl_predictor.models.tournament import adjusted_series_probability, validate_simulation_config


def load_playoff_schedule(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={
            column: "string"
            for column in ("match_id", "team_a_id", "team_b_id", "winner_team_id", "status")
        },
    )
    frame["scheduled_at"] = pd.to_datetime(frame["scheduled_at"], utc=True).dt.tz_convert(
        "Asia/Jakarta"
    )
    for column in ("team_a_score", "team_b_score"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("Int64")
    return frame


def build_playoff_forecast(
    tables: dict[str, pd.DataFrame],
    teams: pd.DataFrame,
    regular_schedule: pd.DataFrame,
    playoff_schedule: pd.DataFrame,
    feature_config: dict[str, Any],
    artifact: dict[str, Any],
    config: dict[str, Any],
    as_of: date,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Enumerate every remaining bracket outcome, conditional on past actual results."""
    validate_simulation_config(config)
    if (
        not regular_schedule["status"].eq("completed").all()
        or regular_schedule["scheduled_at"].dt.date.max() > as_of
    ):
        raise ValueError("Playoff forecasts require a completed regular season at the cutoff.")
    if set(regular_schedule["season"]) != {config["season"]} or set(teams["season"]) != {
        config["season"]
    }:
        raise ValueError("Playoff config and live data must refer to the same season.")
    bracket = config["playoffs"]["bracket"]
    match_ids = [match["match_id"] for match in bracket]
    if (
        playoff_schedule["match_id"].duplicated().any()
        or set(playoff_schedule["match_id"]) != set(match_ids)
        or set(playoff_schedule["season"]) != {config["season"]}
    ):
        raise ValueError("Playoff schedule must contain each configured match exactly once.")
    if not playoff_schedule["status"].isin(["scheduled", "completed"]).all():
        raise ValueError("Unsupported playoff match status.")
    scheduled = playoff_schedule.set_index("match_id").loc[match_ids]
    if not scheduled["scheduled_at"].is_monotonic_increasing:
        raise ValueError("Playoff dates must follow configured bracket execution order.")
    standings = _current_standings(
        regular_schedule, teams, config["regular_season"]["ranking_order"]
    )
    qualified = standings.head(int(config["regular_season"]["playoff_team_count"]))
    seeds = dict(enumerate(qualified["team_id"].astype(str), start=1))
    fixed: dict[str, tuple[str, str]] = {}
    actual_scores: dict[str, dict[str, int]] = {}

    def resolve(source: dict[str, Any], results: dict[str, tuple[str, str]]) -> str:
        if source["source"] == "seed":
            return seeds[int(source["seed"])]
        result = results[str(source["match_id"])]
        return result[0] if source["source"] == "winner" else result[1]

    for match in bracket:
        row = scheduled.loc[match["match_id"]]
        if int(row["best_of"]) != int(match["best_of"]):
            raise ValueError(f"BO mismatch for {match['match_id']}.")
        if row["status"] == "scheduled":
            if row[["team_a_score", "team_b_score", "winner_team_id"]].notna().any():
                raise ValueError("A scheduled match cannot contain actual outcomes.")
            continue
        observed_at = pd.to_datetime(row["observed_at"], errors="coerce")
        if pd.isna(observed_at) or row["scheduled_at"].date() > observed_at.date():
            raise ValueError("A playoff result cannot be observed before its match date.")
        wins = int(match["best_of"]) // 2 + 1
        scores = row[["team_a_score", "team_b_score"]]
        if (
            scores.isna().any()
            or max(scores) != wins
            or min(scores) < 0
            or min(scores) >= wins
            or row[["team_a_id", "team_b_id"]].isna().any()
        ):
            raise ValueError(f"Invalid completed playoff score for {match['match_id']}.")
        if row["scheduled_at"].date() > as_of:
            continue  # Future outcomes remain unavailable in a retrospective forecast.
        try:
            a, b = resolve(match["team_a"], fixed), resolve(match["team_b"], fixed)
        except KeyError as exc:
            raise ValueError("Completed playoff match has unresolved predecessor results.") from exc
        if {str(row["team_a_id"]), str(row["team_b_id"])} != {a, b}:
            raise ValueError(f"Actual participants do not match the bracket: {match['match_id']}.")
        score_map = {
            str(row["team_a_id"]): int(row["team_a_score"]),
            str(row["team_b_id"]): int(row["team_b_score"]),
        }
        winner, loser = (a, b) if score_map[a] > score_map[b] else (b, a)
        if str(row["winner_team_id"]) != winner:
            raise ValueError(f"Winner disagrees with score for {match['match_id']}.")
        fixed[match["match_id"]] = (winner, loser)
        actual_scores[match["match_id"]] = score_map

    probabilities, tracker, history = build_live_match_probabilities(
        tables, regular_schedule, teams, feature_config, artifact
    )
    scale = float(probabilities["online_learning_scale_after_update"].iloc[-1])
    slot_lookup = teams.set_index("team_id")["franchise_slot_id"].astype(str).to_dict()
    for match in bracket:
        match_id = match["match_id"]
        if match_id not in fixed:
            continue
        row = scheduled.loc[match_id]
        a, b = resolve(match["team_a"], fixed), resolve(match["team_b"], fixed)
        _append_match_history(
            SimpleNamespace(
                season=int(config["season"]),
                stage="playoffs",
                match_id=match_id,
                completion_date=_naive_date(row["scheduled_at"]),
                team_a_franchise_slot_id=slot_lookup[a],
                team_b_franchise_slot_id=slot_lookup[b],
                team_a_score=actual_scores[match_id][a],
                team_b_score=actual_scores[match_id][b],
                winner_side="team_a" if fixed[match_id][0] == a else "team_b",
            ),
            tracker,
            history,
        )
    all_ids = teams["team_id"].astype(str).tolist()
    index = {team_id: position for position, team_id in enumerate(all_ids)}
    matrix = _pair_probability_matrix(
        all_ids, slot_lookup, int(config["season"]), tracker, history, artifact, scale
    )
    reference_best_of = int(config["playoffs"]["probability_reference_best_of"])
    champion_id = config["playoffs"]["championship_match_id"]
    champion_mass: dict[str, float] = defaultdict(float)
    reach: dict[tuple[str, str], float] = defaultdict(float)
    win: dict[tuple[str, str], float] = defaultdict(float)
    pair_mass: dict[tuple[str, str, str], float] = defaultdict(float)
    pair_a_win: dict[tuple[str, str, str], float] = defaultdict(float)
    best_path: list[dict[str, Any]] = []
    best_weight = -1.0
    champion_paths: dict[str, tuple[float, list[dict[str, Any]]]] = {}
    total_mass = 0.0
    leaf_count = 0

    def enumerate_paths(
        position: int,
        results: dict[str, tuple[str, str]],
        weight: float,
        path: list[dict[str, Any]],
    ) -> None:
        nonlocal best_path, best_weight, total_mass, leaf_count
        if position == len(bracket):
            leaf_count += 1
            total_mass += weight
            champion_mass[results[champion_id][0]] += weight
            champion = results[champion_id][0]
            if champion not in champion_paths or weight > champion_paths[champion][0]:
                champion_paths[champion] = (weight, path)
            for entry in path:
                match_id, a, b = entry["match_id"], entry["team_a_id"], entry["team_b_id"]
                reach[match_id, a] += weight
                reach[match_id, b] += weight
                win[match_id, entry["winner_team_id"]] += weight
                pair_mass[match_id, a, b] += weight
                if entry["winner_team_id"] == a:
                    pair_a_win[match_id, a, b] += weight
            if weight > best_weight:
                best_weight, best_path = weight, path
            return
        match = bracket[position]
        match_id = match["match_id"]
        a, b = resolve(match["team_a"], results), resolve(match["team_b"], results)
        if a == b:
            raise ValueError("A playoff match cannot contain the same team twice.")
        probability = adjusted_series_probability(
            float(matrix[index[a], index[b]]), int(match["best_of"]), reference_best_of
        )
        choices = (
            [(fixed[match_id][0], 1.0)]
            if match_id in fixed
            else [(a, probability), (b, 1.0 - probability)]
        )
        for winner, branch_weight in choices:
            if branch_weight <= 0:
                continue
            loser = b if winner == a else a
            row = {
                "match_id": match_id,
                "round": match["round"],
                "best_of": match["best_of"],
                "team_a_id": a,
                "team_b_id": b,
                "winner_team_id": winner,
                "loser_team_id": loser,
                "team_a_win_probability": None if match_id in fixed else probability,
                "outcome_type": "actual" if match_id in fixed else "prediction",
                "team_a_score": actual_scores.get(match_id, {}).get(a),
                "team_b_score": actual_scores.get(match_id, {}).get(b),
                "scheduled_at": scheduled.loc[match_id, "scheduled_at"],
                "as_of": as_of.isoformat(),
            }
            enumerate_paths(
                position + 1,
                {**results, match_id: (winner, loser)},
                weight * branch_weight,
                [*path, row],
            )

    enumerate_paths(0, {}, 1.0, [])
    team_rows = []
    names = teams.set_index("team_id")["team_name"].to_dict()
    for seed, team_id in seeds.items():
        row = {
            "season": int(config["season"]),
            "as_of": as_of.isoformat(),
            "seed": seed,
            "team_id": team_id,
            "team_name": names[team_id],
            "champion_probability": champion_mass[team_id],
            "grand_final_probability": reach[champion_id, team_id],
        }
        for round_name in dict.fromkeys(match["round"] for match in bracket):
            row[f"{round_name}_reach_probability"] = sum(
                reach[match["match_id"], team_id]
                for match in bracket
                if match["round"] == round_name
            )
        team_rows.append(row)
    team_frame = pd.DataFrame(team_rows).sort_values("champion_probability", ascending=False)
    matchup_frame = pd.DataFrame(
        [
            {
                "match_id": match_id,
                "team_a_id": a,
                "team_b_id": b,
                "matchup_probability": mass,
                "team_a_win_probability": pair_a_win[match_id, a, b] / mass,
            }
            for (match_id, a, b), mass in sorted(pair_mass.items())
        ]
    )
    checks = {
        "champion_probability_sum": float(team_frame["champion_probability"].sum()),
        "grand_final_probability_sum": float(team_frame["grand_final_probability"].sum()),
        "terminal_probability_sum": total_mass,
    }
    if any(
        abs(checks[key] - expected) > 1e-9
        for key, expected in (
            ("champion_probability_sum", 1.0),
            ("grand_final_probability_sum", 2.0),
            ("terminal_probability_sum", 1.0),
        )
    ):
        raise ValueError("Playoff forecast failed probability normalization.")
    report = {
        "season": int(config["season"]),
        "as_of": as_of.isoformat(),
        "method": "exact_bracket_enumeration",
        "terminal_scenario_count": leaf_count,
        "model_name": artifact["model_name"],
        "format_confirmation": config["format_confirmation"],
        "regular_match_count": len(regular_schedule),
        "completed_playoff_match_count": len(fixed),
        "remaining_playoff_match_count": len(bracket) - len(fixed),
        "most_likely_scenario_probability": best_weight,
        "most_likely_scenario_champion": best_path[-1]["winner_team_id"],
        "confirmed_champion": fixed[champion_id][0] if champion_id in fixed else None,
        "validation": checks,
        "assumptions": [
            "Seluruh seed berasal dari klasemen akhir regular season.",
            "Pasangan dan BO mengikuti simulation_config; jadwal resmi dapat masih TBD.",
            "Probabilitas dibekukan pada state data yang tersedia pada cutoff.",
            "Hasil aktual memperbarui Elo/form; confidence scale memakai kalibrasi akhir RS.",
            "Prediksi pasangan babak lanjut bersyarat pada skenario; bukan jadwal peserta resmi.",
            "Skenario lengkap paling mungkin dapat berbeda dari favorit juara marginal.",
        ],
        "team_predictions": dataframe_records(team_frame),
        "scenario": dataframe_records(pd.DataFrame(best_path)),
        "champion_paths": {
            team_id: {
                "joint_probability": weight,
                "conditional_probability": weight / champion_mass[team_id],
                "matches": dataframe_records(pd.DataFrame(path)),
            }
            for team_id, (weight, path) in champion_paths.items()
        },
    }
    return {
        "teams": team_frame.reset_index(drop=True),
        "matches": pd.DataFrame(best_path),
        "matchups": matchup_frame,
    }, report


def write_playoff_forecast(
    frames: dict[str, pd.DataFrame],
    report: dict[str, Any],
    prediction_dir: Path,
    report_path: Path,
    archive_dir: Path | None = None,
) -> None:
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for key, frame in frames.items():
        frame.to_parquet(
            prediction_dir / f"season{report['season']}_playoff_{key}.parquet", index=False
        )
    write_json(report, report_path)
    if archive_dir is not None:
        archive_dir.mkdir(parents=True, exist_ok=True)
        for key, frame in frames.items():
            frame.to_csv(archive_dir / f"{key}.csv", index=False)
        write_json(report, archive_dir / "report.json")

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Streamlit Cloud may execute this file directly instead of installing the src-layout
# package. Add the src directory before importing mpl_predictor so both entry points work.
SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mpl_predictor.analysis.season_results import (  # noqa: E402
    build_championship_comparison,
    build_playoff_match_comparison,
    load_preplayoff_archive,
)
from mpl_predictor.config import ProjectPaths, get_project_paths  # noqa: E402

PLAYOFF_COMPARISON_STYLES = {
    "same_winner": ("Sesuai skenario", "#4ade80"),
    "different_winner": ("Tidak sesuai skenario", "#f87171"),
    "different_pairing": ("Pasangan berbeda", "#fbbf24"),
    "pending_result": ("Belum ada hasil", "#94a3b8"),
    "prediction_unavailable": ("Prediksi tidak tersedia", "#94a3b8"),
}


def _playoff_comparison_labels(row: dict[str, Any]) -> tuple[str, str, str, str]:
    prediction = "Tidak tersedia"
    if pd.notna(row["predicted_winner_team_id"]):
        prediction = (
            f"{row['predicted_team_a_id']} vs {row['predicted_team_b_id']} "
            f"→ {row['predicted_winner_team_id']}"
        )
    actual = "Belum ada hasil"
    if pd.notna(row["actual_winner_team_id"]):
        actual = (
            f"{row['actual_team_a_id']} {int(row['actual_team_a_score'])}-"
            f"{int(row['actual_team_b_score'])} {row['actual_team_b_id']}"
        )
    label, color = PLAYOFF_COMPARISON_STYLES[row["comparison_status"]]
    return prediction, actual, label, color


def build_playoff_comparison_table(comparison: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "play_in_1": "Babak awal 1",
        "play_in_2": "Babak awal 2",
        "upper_semifinal_1": "Semifinal atas 1",
        "upper_semifinal_2": "Semifinal atas 2",
        "lower_semifinal": "Semifinal bawah",
        "upper_final": "Final atas",
        "lower_final": "Final bawah",
        "grand_final": "Grand Final",
    }
    rows = []
    for record in comparison.to_dict(orient="records"):
        predicted, actual, status, _ = _playoff_comparison_labels(record)
        rows.append(
            {
                "Babak": labels.get(record["match_id"], record["match_id"]),
                "Prediksi awal": predicted,
                "Hasil aktual": actual,
                "Perbandingan": status,
            }
        )
    return pd.DataFrame(rows, columns=["Babak", "Prediksi awal", "Hasil aktual", "Perbandingan"])


def dashboard_file_paths(paths: ProjectPaths) -> dict[str, Path]:
    prediction_dir = paths.processed / "predictions"
    return {
        "predictions": prediction_dir / "season18_snapshot_predictions.parquet",
        "matches": prediction_dir / "season18_snapshot_match_probabilities.parquet",
        "global_importance": prediction_dir / "season18_global_feature_importance.parquet",
        "match_explanations": prediction_dir / "season18_match_explanations.parquet",
        "team_explanations": prediction_dir / "season18_team_explanations.parquet",
        "rosters": paths.data / "season18" / "rosters.csv",
        "model_report": paths.reports / "model_evaluation_report.json",
        "simulation_config": paths.root / "config" / "simulation_config.json",
    }


def load_dashboard_data(paths: ProjectPaths) -> tuple[dict[str, Any], list[Path]]:
    files = dashboard_file_paths(paths)
    missing = [path for path in files.values() if not path.exists()]
    if missing:
        return {}, missing
    with files["model_report"].open(encoding="utf-8") as handle:
        model_report = json.load(handle)
    with files["simulation_config"].open(encoding="utf-8") as handle:
        simulation_config = json.load(handle)
    frames = {
        "predictions": pd.read_parquet(files["predictions"]),
        "matches": pd.read_parquet(files["matches"]),
        "global_importance": pd.read_parquet(files["global_importance"]),
        "match_explanations": pd.read_parquet(files["match_explanations"]),
        "team_explanations": pd.read_parquet(files["team_explanations"]),
        "rosters": pd.read_csv(files["rosters"]),
        "model_comparison": pd.DataFrame.from_records(
            model_report["match_model"]["best_variant_by_family"]
        ),
        "simulation_config": simulation_config,
    }
    playoff_report = paths.reports / "season18_playoff_report.json"
    playoff_files = {
        key: paths.processed / "predictions" / f"season18_playoff_{key}.parquet"
        for key in ("teams", "matches", "matchups")
    }
    if playoff_report.exists() and all(path.exists() for path in playoff_files.values()):
        with playoff_report.open(encoding="utf-8") as handle:
            frames["playoff_report"] = json.load(handle)
        for key, path in playoff_files.items():
            frames[f"playoff_{key}"] = pd.read_parquet(path)
        archive = load_preplayoff_archive(
            paths.data / "season18", frames["playoff_matches"], season=18
        )
        if archive is not None:
            frames["preplayoff_teams"], frames["preplayoff_report"] = archive
            archived_config = (
                paths.data
                / "season18"
                / "playoff_prediction_snapshots"
                / frames["preplayoff_report"]["as_of"]
                / "simulation_config.json"
            )
            if archived_config.exists():
                with archived_config.open(encoding="utf-8") as handle:
                    frames["historical_simulation_config"] = json.load(handle)
    return frames, []


def _snapshot_label(row: pd.Series) -> str:
    day = pd.Timestamp(row["feature_cutoff_date"]).strftime("%d-%m-%Y")
    if row["prediction_type"] == "preseason":
        return f"Sebelum musim · {day}"
    if row["prediction_type"] == "as_of":
        return f"Minggu {int(row['partial_week'])} berjalan · {day}"
    return f"Akhir minggu {int(row['completed_week'])} · {day}"


def _percent(value: Any) -> str:
    return f"{float(value):.2%}"


def _prediction_chart(frame: pd.DataFrame):
    chart_data = frame.sort_values("champion_probability", ascending=True)
    figure = px.bar(
        chart_data,
        x="champion_probability",
        y="team_name",
        orientation="h",
        color="champion_probability",
        color_continuous_scale="Blues",
        text=chart_data["champion_probability"].map(_percent),
        labels={"champion_probability": "Peluang juara", "team_name": "Tim"},
    )
    figure.update_layout(
        height=340,
        margin={"l": 0, "r": 40, "t": 10, "b": 10},
        coloraxis_showscale=False,
        yaxis_title=None,
        xaxis_tickformat=".0%",
    )
    figure.update_traces(textposition="outside")
    return figure


def _history_chart(predictions: pd.DataFrame):
    labels = (
        predictions[
            [
                "snapshot_id",
                "snapshot_order",
                "prediction_type",
                "completed_week",
                "partial_week",
                "feature_cutoff_date",
            ]
        ]
        .drop_duplicates()
        .sort_values("snapshot_order")
    )
    label_lookup = {
        row.snapshot_id: "Pramusim"
        if row.prediction_type == "preseason"
        else (
            f"As-of {row.feature_cutoff_date}"
            if row.prediction_type == "as_of"
            else f"Week {int(row.completed_week)}"
        )
        for row in labels.itertuples(index=False)
    }
    chart_data = predictions.copy()
    chart_data["snapshot_label"] = chart_data["snapshot_id"].map(label_lookup)
    figure = px.line(
        chart_data.sort_values("snapshot_order"),
        x="snapshot_label",
        y="champion_probability",
        color="team_name",
        markers=True,
        labels={
            "snapshot_label": "Snapshot",
            "champion_probability": "Peluang juara",
            "team_name": "Tim",
        },
    )
    figure.update_layout(yaxis_tickformat=".0%")
    return figure


def build_projected_playoff_seeds(predictions: pd.DataFrame, snapshot_id: str) -> pd.DataFrame:
    """Build a unique projected top-six seeding from the selected simulation snapshot."""
    current = predictions.loc[predictions["snapshot_id"].eq(snapshot_id)].copy()
    if current.empty:
        raise ValueError(f"Snapshot prediksi tidak ditemukan: {snapshot_id}")
    projected = (
        current.sort_values(
            ["expected_regular_rank", "playoff_probability", "team_id"],
            ascending=[True, False, True],
        )
        .head(6)
        .reset_index(drop=True)
    )
    projected["projected_seed"] = range(1, len(projected) + 1)
    projected["projected_seed_probability"] = projected.apply(
        lambda row: row[f"seed_{int(row['projected_seed'])}_probability"], axis=1
    )
    return projected


def build_playoff_bracket_figure(
    projected_seeds: pd.DataFrame,
    simulation_config: dict[str, Any],
    scenario: pd.DataFrame | None = None,
    comparison: pd.DataFrame | None = None,
) -> go.Figure:
    """Render the configured six-team playoff flow as a responsive bracket figure."""
    required_matches = {
        "play_in_1",
        "play_in_2",
        "upper_semifinal_1",
        "upper_semifinal_2",
        "lower_semifinal",
        "upper_final",
        "lower_final",
        "grand_final",
    }
    bracket = {str(match["match_id"]): match for match in simulation_config["playoffs"]["bracket"]}
    missing_matches = required_matches - set(bracket)
    if missing_matches:
        raise ValueError(f"Konfigurasi bracket belum lengkap: {sorted(missing_matches)}")

    seed_lookup = {
        int(row.projected_seed): str(row.team_id) for row in projected_seeds.itertuples(index=False)
    }
    seed_probability_lookup = {
        int(row.projected_seed): float(row.projected_seed_probability)
        for row in projected_seeds.itertuples(index=False)
    }
    short_match_labels = {
        "play_in_1": "UB Quarterfinal 1",
        "play_in_2": "UB Quarterfinal 2",
        "upper_semifinal_1": "UB Semifinal 1",
        "upper_semifinal_2": "UB Semifinal 2",
        "lower_semifinal": "LB Semifinal",
        "upper_final": "UB Final",
        "lower_final": "LB Final",
        "grand_final": "Grand Final",
    }

    def participant_label(source: dict[str, Any]) -> tuple[str, str | None]:
        source_type = str(source["source"])
        if source_type == "seed":
            seed = int(source["seed"])
            team_id = seed_lookup.get(seed, "TBD")
            probability = seed_probability_lookup.get(seed)
            detail = _percent(probability) if probability is not None else None
            return f"Seed {seed} · {team_id}", detail
        match_label = short_match_labels[str(source["match_id"])]
        prefix = "Pemenang" if source_type == "winner" else "Kalah"
        return f"{prefix} {match_label}", None

    positions = {
        "play_in_1": (1.25, 8.70),
        "play_in_2": (1.25, 5.90),
        "upper_semifinal_1": (4.35, 8.70),
        "upper_semifinal_2": (4.35, 5.90),
        "upper_final": (7.45, 7.30),
        "lower_semifinal": (4.35, 2.10),
        "lower_final": (7.45, 3.50),
        "grand_final": (10.55, 5.40),
    }
    headers = [
        (1.25, 10.35, "Upper Bracket Quarterfinals"),
        (4.35, 10.35, "Upper Bracket Semifinals"),
        (7.45, 10.35, "Upper Bracket Final"),
        (10.55, 10.35, "Grand Final"),
        (4.35, 3.55, "Lower Bracket Semifinals"),
        (7.45, 4.95, "Lower Bracket Final"),
    ]
    card_width = 2.40
    card_height = 1.08
    half_width = card_width / 2
    half_height = card_height / 2
    figure = go.Figure()
    scenario_lookup = (
        scenario.set_index("match_id").to_dict(orient="index") if scenario is not None else {}
    )
    comparison_lookup = (
        comparison.set_index("match_id").to_dict(orient="index") if comparison is not None else {}
    )
    comparison_height = 0.95

    for x, y, title in headers:
        figure.add_shape(
            type="rect",
            x0=x - 1.35,
            x1=x + 1.35,
            y0=y - 0.27,
            y1=y + 0.27,
            line={"width": 0},
            fillcolor="#27272a",
            layer="below",
        )
        figure.add_annotation(
            x=x,
            y=y,
            text=f"<b>{title}</b>",
            showarrow=False,
            font={"size": 12, "color": "#e5e7eb"},
        )

    for match_id, (x, y) in positions.items():
        match = bracket[match_id]
        team_a, team_a_detail = participant_label(match["team_a"])
        team_b, team_b_detail = participant_label(match["team_b"])
        forecast = scenario_lookup.get(match_id)
        comparison_row = comparison_lookup.get(match_id)
        comparison_color = (
            PLAYOFF_COMPARISON_STYLES[comparison_row["comparison_status"]][1]
            if comparison_row is not None
            else "#4b5563"
        )
        if forecast is not None:
            team_a, team_b = str(forecast["team_a_id"]), str(forecast["team_b_id"])
            if forecast["outcome_type"] == "actual":
                team_a_detail = str(int(forecast["team_a_score"]))
                team_b_detail = str(int(forecast["team_b_score"]))
            else:
                probability = float(forecast["team_a_win_probability"])
                team_a_detail, team_b_detail = _percent(probability), _percent(1 - probability)
            marker = "✓" if forecast["outcome_type"] == "actual" else "▸"
            if forecast["winner_team_id"] == team_a:
                team_a = f"{marker} {team_a}"
            else:
                team_b = f"{marker} {team_b}"
        figure.add_shape(
            type="rect",
            name=f"match-{match_id}",
            x0=x - half_width,
            x1=x + half_width,
            y0=y - half_height - (comparison_height if comparison_row is not None else 0),
            y1=y + half_height,
            line={"color": comparison_color, "width": 1},
            fillcolor="#18181b",
            layer="below",
        )
        figure.add_shape(
            type="line",
            x0=x - half_width,
            x1=x + half_width,
            y0=y,
            y1=y,
            line={"color": "#3f3f46", "width": 1},
            layer="below",
        )
        figure.add_annotation(
            x=x - half_width + 0.12,
            y=y + 0.26,
            text=f"<b>{team_a}</b>",
            showarrow=False,
            xanchor="left",
            font={"size": 11, "color": "#f4f4f5"},
        )
        figure.add_annotation(
            x=x - half_width + 0.12,
            y=y - 0.26,
            text=f"<b>{team_b}</b>",
            showarrow=False,
            xanchor="left",
            font={"size": 11, "color": "#f4f4f5"},
        )
        if team_a_detail is not None:
            figure.add_annotation(
                x=x + half_width - 0.10,
                y=y + 0.26,
                text=team_a_detail,
                showarrow=False,
                xanchor="right",
                font={"size": 10, "color": "#93c5fd"},
            )
        if team_b_detail is not None:
            figure.add_annotation(
                x=x + half_width - 0.10,
                y=y - 0.26,
                text=team_b_detail,
                showarrow=False,
                xanchor="right",
                font={"size": 10, "color": "#93c5fd"},
            )
        if comparison_row is not None:
            prediction_text, actual_text, status_text, _ = _playoff_comparison_labels(
                comparison_row
            )
            figure.add_shape(
                type="line",
                name=f"comparison-divider-{match_id}",
                x0=x - half_width,
                x1=x + half_width,
                y0=y - half_height,
                y1=y - half_height,
                line={"color": "#3f3f46", "width": 1},
                layer="below",
            )
            for kind, offset, text, color in (
                ("prediction", 0.18, f"Prediksi: {prediction_text}", "#bfdbfe"),
                ("actual", 0.46, f"Hasil: {actual_text}", "#e5e7eb"),
                ("status", 0.74, f"<b>{status_text}</b>", comparison_color),
            ):
                figure.add_annotation(
                    name=f"comparison-{kind}-{match_id}",
                    x=x - half_width + 0.1,
                    y=y - half_height - offset,
                    text=text,
                    showarrow=False,
                    xanchor="left",
                    font={"size": 10, "color": color},
                )
        match_heading = f"{short_match_labels[match_id]} · BO{int(match['best_of'])}"
        if forecast is not None:
            date_label = pd.Timestamp(forecast["scheduled_at"]).tz_convert("Asia/Jakarta")
            match_heading += f" · {date_label.strftime('%d/%m %H:%M')}"
        figure.add_annotation(
            x=x - half_width,
            y=y + half_height + 0.13,
            text=match_heading,
            showarrow=False,
            xanchor="left",
            font={"size": 9, "color": "#a1a1aa"},
        )

    def add_connector(
        source: str,
        target: str,
        *,
        target_offset: float = 0.0,
        target_side: str,
        color: str = "#71717a",
    ) -> None:
        source_x, source_y = positions[source]
        target_x, target_y = positions[target]
        start_x = source_x + half_width
        end_x = target_x - half_width
        bend_x = (start_x + end_x) / 2
        start_y = source_y
        end_y = target_y + target_offset
        for segment, (x0, y0, x1, y1) in enumerate(
            (
                (start_x, start_y, bend_x, start_y),
                (bend_x, start_y, bend_x, end_y),
                (bend_x, end_y, end_x, end_y),
            )
        ):
            figure.add_shape(
                type="line",
                name=f"connector-{source}-{target}-{target_side}-{segment}",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                line={"color": color, "width": 2},
                layer="below",
            )

    upper_color = "#60a5fa"
    lower_color = "#f59e0b"
    final_color = "#facc15"
    for target, match in bracket.items():
        for side, offset in (("team_a", 0.26), ("team_b", -0.26)):
            participant = match[side]
            if participant["source"] == "seed":
                continue
            source = str(participant["match_id"])
            if participant["source"] == "loser":
                # Reference transfers keep same-column lower-bracket routes off the cards.
                x, y = positions[source]
                bottom = y - half_height - (comparison_height if source in comparison_lookup else 0)
                figure.add_shape(
                    type="line",
                    name=f"transfer-{source}-{target}-{side}",
                    x0=x,
                    x1=x,
                    y0=bottom,
                    y1=bottom - 0.22,
                    line={"color": lower_color, "width": 1.5, "dash": "dot"},
                    layer="below",
                )
                figure.add_annotation(
                    x=x,
                    y=bottom - 0.42,
                    text=f"Kalah → {short_match_labels[target]}",
                    showarrow=False,
                    font={"size": 10, "color": lower_color},
                )
                continue
            color = (
                final_color
                if target == "grand_final"
                else lower_color
                if target.startswith("lower_")
                else upper_color
            )
            add_connector(source, target, target_offset=offset, target_side=side, color=color)

    figure.update_layout(
        height=850 if comparison_lookup else 650,
        margin={"l": 5, "r": 5, "t": 5, "b": 5},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis={"visible": False, "range": [-0.20, 12.00], "fixedrange": True},
        yaxis={
            "visible": False,
            "range": [0.40 if comparison_lookup else 0.65, 10.85],
            "fixedrange": True,
        },
        hovermode=False,
    )
    return figure


def _latest_snapshot_id(data: dict[str, Any]) -> str:
    predictions = data["predictions"]
    return str(predictions.loc[predictions["snapshot_order"].idxmax(), "snapshot_id"])


def _select_regular_snapshot(data: dict[str, Any]) -> str:
    rows = data["predictions"].drop_duplicates("snapshot_id").sort_values("snapshot_order")
    labels = {str(row["snapshot_id"]): _snapshot_label(row) for _, row in rows.iterrows()}
    with st.expander("Lihat minggu sebelumnya"):
        st.caption("Opsional. Tampilan awal selalu memakai periode terbaru.")
        selected = st.selectbox(
            "Periode prediksi", list(labels), index=len(labels) - 1, format_func=labels.get
        )
    return str(selected)


def _evaluated_matches(data: dict[str, Any], snapshot_id: str | None = None) -> pd.DataFrame:
    matches = data["matches"]
    return matches.loc[
        matches["snapshot_id"].eq(snapshot_id or _latest_snapshot_id(data))
        & matches["status"].eq("completed")
        & matches["prediction_correct"].notna()
    ]


def build_regular_match_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Present pre-match predictions without changing their values or row order."""
    current = matches.copy()
    scheduled_at = pd.to_datetime(current["scheduled_at"], utc=True).dt.tz_convert("Asia/Jakarta")
    predicted_probability = current["team_a_win_probability"].where(
        current["predicted_winner_team_id"].eq(current["team_a_id"]),
        1 - current["team_a_win_probability"],
    )
    table = pd.DataFrame(index=current.index)
    table["Week"] = current["week"].astype(int)
    table["Jadwal"] = scheduled_at.dt.strftime("%d-%m-%Y %H:%M")
    table["Pertandingan"] = current["team_a_id"] + " vs " + current["team_b_id"]
    table["Prediksi"] = [
        f"{team} ({_percent(probability)})"
        for team, probability in zip(
            current["predicted_winner_team_id"], predicted_probability, strict=True
        )
    ]
    table["Skor"] = [
        f"{int(a)}-{int(b)}" if status == "completed" and pd.notna(a) and pd.notna(b) else "—"
        for a, b, status in zip(
            current["team_a_score"], current["team_b_score"], current["status"], strict=True
        )
    ]
    table["Status akurasi"] = current["accuracy_status"].map(
        {"correct": "✅ Benar", "incorrect": "❌ Salah", "pending_result": "Belum dimainkan"}
    )
    return table.reset_index(drop=True)


def _render_overview(data: dict[str, Any], snapshot_id: str) -> None:
    current = data["predictions"].loc[data["predictions"]["snapshot_id"].eq(snapshot_id)]
    current = current.sort_values("champion_probability", ascending=False)
    matches = data["matches"].loc[data["matches"]["snapshot_id"].eq(snapshot_id)]
    leader = current.iloc[0]
    completed = int(matches["status"].eq("completed").sum())
    first, second, third = st.columns(3)
    first.metric("Favorit pada periode ini", str(leader["team_id"]))
    second.metric("Peluang juara favorit", _percent(leader["champion_probability"]))
    third.metric("Pertandingan selesai", f"{completed}/{len(matches)}")

    st.subheader("Peluang juara saat regular season")
    st.caption(
        "Ini prediksi sebelum playoff, bukan hasil akhir musim. "
        "Lihat halaman Hasil Prediksi untuk membandingkannya dengan juara aktual."
    )
    st.plotly_chart(_prediction_chart(current), width="stretch", config={"displayModeBar": False})
    with st.expander("Rincian peluang setiap tim"):
        table = current[
            ["team_name", "champion_probability", "playoff_probability", "grand_final_probability"]
        ].rename(
            columns={
                "team_name": "Tim",
                "champion_probability": "Juara",
                "playoff_probability": "Lolos playoff",
                "grand_final_probability": "Grand Final",
            }
        )
        st.dataframe(
            table.style.format({column: "{:.2%}" for column in table.columns[1:]}),
            hide_index=True,
            width="stretch",
        )


def _render_matches(data: dict[str, Any], snapshot_id: str) -> None:
    matches = data["matches"].loc[data["matches"]["snapshot_id"].eq(snapshot_id)]
    evaluated = _evaluated_matches(data, snapshot_id)
    correct = int(evaluated["prediction_correct"].sum())
    st.subheader("Prediksi dan akurasi pertandingan")
    first, second = st.columns(2)
    first.metric("Akurasi prediksi", _percent(correct / len(evaluated)) if len(evaluated) else "—")
    second.metric("Prediksi benar", f"{correct}/{len(evaluated)}")
    st.caption("Benar atau salah dinilai dari prediksi sebelum pertandingan dimulai.")

    upcoming = matches.loc[matches["status"].eq("scheduled")].sort_values(
        ["scheduled_at", "official_match_id"]
    )
    if not upcoming.empty:
        st.subheader("Pertandingan berikutnya")
        st.dataframe(build_regular_match_table(upcoming.head(8)), hide_index=True, width="stretch")
        if len(upcoming) > 8:
            with st.expander(f"Semua jadwal berikutnya ({len(upcoming)})"):
                st.dataframe(
                    build_regular_match_table(upcoming),
                    hide_index=True,
                    width="stretch",
                    height=400,
                )

    completed = matches.loc[matches["status"].eq("completed")].sort_values(
        ["week", "scheduled_at", "official_match_id"], ascending=[False, False, False]
    )
    if not completed.empty:
        st.subheader("Pertandingan terbaru")
        st.dataframe(build_regular_match_table(completed.head(8)), hide_index=True, width="stretch")
        if len(completed) > 8:
            with st.expander(f"Semua hasil pertandingan ({len(completed)})"):
                st.dataframe(
                    build_regular_match_table(completed),
                    hide_index=True,
                    width="stretch",
                    height=400,
                )
    elif upcoming.empty:
        st.info("Belum ada jadwal pertandingan pada periode ini.")


def _render_explainability(data: dict[str, Any]) -> None:
    st.subheader("Explainability model")
    st.caption(
        "Kontribusi pertandingan menjelaskan raw logistic logit sebelum side-symmetry, "
        "kalibrasi Platt, dan simulasi Monte Carlo. Nilai ini bukan hubungan sebab-akibat."
    )
    global_importance = (
        data["global_importance"].head(12).sort_values("absolute_importance", ascending=True)
    )
    figure = px.bar(
        global_importance,
        x="absolute_importance",
        y="feature_label",
        orientation="h",
        color="coefficient",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        labels={"absolute_importance": "|Koefisien|", "feature_label": "Fitur"},
    )
    figure.update_layout(yaxis_title=None)
    st.plotly_chart(figure, width="stretch")

    explanations = data["match_explanations"]
    if explanations.empty:
        st.info(
            "Tidak ada pertandingan regular season yang menunggu dimainkan pada snapshot terbaru."
        )
    else:
        match_labels = (
            explanations[["match_id", "week", "team_a_id", "team_b_id"]]
            .drop_duplicates()
            .sort_values(["week", "match_id"])
        )
        label_lookup = {
            row.match_id: f"W{int(row.week)} · {row.team_a_id} vs {row.team_b_id}"
            for row in match_labels.itertuples(index=False)
        }
        selected_match = st.selectbox(
            "Pilih pertandingan", options=list(label_lookup), format_func=label_lookup.get
        )
        local = explanations.loc[explanations["match_id"].eq(selected_match)].nsmallest(
            10, "contribution_rank"
        )
        local = local.sort_values("contribution")
        local_figure = px.bar(
            local,
            x="contribution",
            y="feature_label",
            orientation="h",
            color="favors_team_id",
            labels={"contribution": "Kontribusi raw logit", "feature_label": "Fitur"},
        )
        local_figure.update_layout(yaxis_title=None)
        st.plotly_chart(local_figure, width="stretch")

    st.subheader("Perubahan peluang tim")
    team_table = data["team_explanations"][
        [
            "team_name",
            "preseason_champion_probability",
            "current_champion_probability",
            "champion_probability_change",
            "mean_remaining_match_win_probability",
        ]
    ].rename(
        columns={
            "team_name": "Tim",
            "preseason_champion_probability": "Pramusim",
            "current_champion_probability": "Terbaru",
            "champion_probability_change": "Perubahan",
            "mean_remaining_match_win_probability": "Rata-rata peluang match tersisa",
        }
    )
    st.dataframe(
        team_table.style.format(
            {
                "Pramusim": "{:.2%}",
                "Terbaru": "{:.2%}",
                "Perubahan": "{:+.2%}",
                "Rata-rata peluang match tersisa": "{:.2%}",
            },
            na_rep="—",
        ),
        hide_index=True,
        width="stretch",
    )


def _render_model_comparison(data: dict[str, Any]) -> None:
    st.subheader("Perbandingan model pertandingan")
    st.caption(
        "Semua model diuji pada 736 pertandingan dengan fold walk-forward Season 8-17. "
        "Nilai log loss, Brier, dan ECE yang lebih rendah lebih baik."
    )
    comparison = data["model_comparison"].copy()
    family_labels = {
        "logistic": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    comparison["Model"] = comparison["model_family"].map(family_labels)
    comparison["Status"] = comparison["model_family"].map(
        {
            "logistic": "✅ Model produksi",
            "random_forest": "🧪 Challenger",
            "xgboost": "🧪 Challenger",
        }
    )
    comparison["Varian"] = comparison["model_name"].str.removeprefix("match_")
    chart = px.bar(
        comparison.sort_values("log_loss", ascending=False),
        x="log_loss",
        y="Model",
        orientation="h",
        color="Status",
        text="log_loss",
        labels={"log_loss": "Log loss walk-forward"},
    )
    chart.update_traces(texttemplate="%{text:.6f}", textposition="outside")
    chart.update_layout(xaxis_rangemode="tozero")
    st.plotly_chart(chart, width="stretch")
    table = comparison[
        ["Model", "Status", "Varian", "log_loss", "brier_score", "accuracy", "ece"]
    ].rename(
        columns={
            "log_loss": "Log loss",
            "brier_score": "Brier",
            "accuracy": "Accuracy",
            "ece": "ECE",
        }
    )
    st.dataframe(
        table.style.format(
            {
                "Log loss": "{:.6f}",
                "Brier": "{:.6f}",
                "Accuracy": "{:.2%}",
                "ECE": "{:.6f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.info(
        "Random Forest raw unggul sangat tipis pada log loss dan Brier, tetapi Logistic "
        "Regression tetap dipakai karena accuracy dan kalibrasinya lebih kuat. Pemilihan "
        "model final tidak dilakukan otomatis."
    )


def _render_data_scope(data: dict[str, Any]) -> None:
    st.subheader("Cakupan data")
    first, second, third = st.columns(3)
    first.metric("Snapshot", data["predictions"]["snapshot_id"].nunique())
    second.metric("Tim", data["predictions"]["team_id"].nunique())
    third.metric("Anggota roster tercatat", len(data["rosters"]))
    st.markdown(
        "- Pramusim memakai hasil historis sampai S17 dan **0 hasil S18**.\n"
        "- Snapshot mingguan hanya membuka hasil sampai week terkait.\n"
        "- Snapshot as-of dapat berhenti di tengah week pada akhir hari WIB.\n"
        "- Hasil selesai dievaluasi dari prediksi pre-match, lalu memperbarui state "
        "Elo/form untuk match berikutnya.\n"
        "- Bracket playoff mengikuti konfigurasi playoff enam tim.\n"
        "- Probabilitas match dibekukan pada setiap snapshot simulasi."
    )


def _render_regular_season_page(data: dict[str, Any], snapshot_id: str | None = None) -> None:
    st.title("Regular Season")
    st.caption("MPL Indonesia Season 18 · Peluang tim dan hasil pertandingan regular season.")
    selected = snapshot_id or _select_regular_snapshot(data)
    period = data["predictions"].loc[data["predictions"]["snapshot_id"].eq(selected)].iloc[0]
    st.caption(f"Menampilkan: {_snapshot_label(period)}")
    _render_overview(data, selected)
    st.divider()
    _render_matches(data, selected)

    with st.expander("Riwayat peluang dari minggu ke minggu"):
        st.plotly_chart(_history_chart(data["predictions"]), width="stretch")
    with st.expander("Tentang model, pembelajaran, dan data"):
        matches = (
            data["matches"]
            .loc[data["matches"]["snapshot_id"].eq(selected)]
            .sort_values(["scheduled_at", "official_match_id"])
        )
        learned = int(matches["online_learning_update_applied"].sum())
        st.write(
            f"{learned} hasil pertandingan sudah dipelajari setelah prediksinya dibuat. "
            "Hasil memperbarui kekuatan tim dan penyesuaian probabilitas untuk laga berikutnya."
        )
        if not matches.empty:
            st.caption(
                "Penyesuaian keyakinan model terakhir: "
                f"{float(matches['online_learning_scale_after_update'].iloc[-1]):.3f}. "
                "Penjelasan fitur di bawah memakai pembaruan model terbaru."
            )
        _render_model_comparison(data)
        _render_explainability(data)
        _render_data_scope(data)


def _completed_grand_final(data: dict[str, Any]) -> pd.Series | None:
    matches = data.get("playoff_matches")
    if matches is None:
        return None
    final = matches.loc[matches["round"].eq("grand_final") & matches["outcome_type"].eq("actual")]
    return None if final.empty else final.iloc[0]


def _render_final_summary(data: dict[str, Any]) -> None:
    report = data["playoff_report"]
    final = _completed_grand_final(data)
    first, second, third = st.columns(3)
    first.metric("Juara aktual", report["confirmed_champion"] or "Belum ditentukan")
    second.metric(
        "Runner-up", str(final["loser_team_id"]) if final is not None else "Belum ditentukan"
    )
    score = (
        f"{final['team_a_id']} {int(final['team_a_score'])}-"
        f"{int(final['team_b_score'])} {final['team_b_id']}"
        if final is not None
        else "Belum dimainkan"
    )
    third.metric("Skor Grand Final", score)


def _render_playoff_forecast_tables(
    data: dict[str, Any], scenario: pd.DataFrame, *, show_details: bool = True
) -> None:
    report = data["playoff_report"]
    confirmed = bool(report["confirmed_champion"])
    st.subheader("Hasil pertandingan" if confirmed else "Jadwal dan prediksi")
    path = scenario.copy()
    rounds = {
        "play_ins": "Babak awal",
        "upper_semifinal": "Semifinal atas",
        "lower_semifinal": "Semifinal bawah",
        "upper_final": "Final atas",
        "lower_final": "Final bawah",
        "grand_final": "Grand Final",
    }
    table = pd.DataFrame(index=path.index)
    table["Tanggal"] = (
        pd.to_datetime(path["scheduled_at"], utc=True)
        .dt.tz_convert("Asia/Jakarta")
        .dt.strftime("%d-%m %H:%M")
    )
    table["Babak"] = path["round"].map(rounds)
    table["Pertandingan"] = path["team_a_id"] + " vs " + path["team_b_id"]
    table["Skor"] = [
        f"{int(a)}-{int(b)}" if kind == "actual" else "—"
        for a, b, kind in zip(
            path["team_a_score"], path["team_b_score"], path["outcome_type"], strict=True
        )
    ]
    table["Pemenang" if confirmed else "Pemenang / prediksi"] = path["winner_team_id"]
    if not confirmed:
        table["Status"] = path["outcome_type"].map(
            {"actual": "Hasil aktual", "prediction": "Prediksi, belum pasti"}
        )
        st.caption("Pasangan di babak berikutnya adalah contoh skenario, bukan peserta yang pasti.")
    st.dataframe(table, hide_index=True, width="stretch")
    if not show_details:
        return

    with st.expander("Detail peluang dan metode"):
        if not confirmed:
            columns = {
                "team_name": "Tim",
                "upper_semifinal_reach_probability": "Semifinal atas",
                "lower_semifinal_reach_probability": "Semifinal bawah",
                "upper_final_reach_probability": "Final atas",
                "lower_final_reach_probability": "Final bawah",
                "grand_final_probability": "Grand Final",
                "champion_probability": "Juara",
            }
            chances = data["playoff_teams"][list(columns)].rename(columns=columns)
            st.dataframe(
                chances.style.format({label: "{:.2%}" for label in list(columns.values())[1:]}),
                hide_index=True,
                width="stretch",
            )
            pairs = data["playoff_matchups"].rename(
                columns={
                    "match_id": "Babak",
                    "team_a_id": "Tim A",
                    "team_b_id": "Tim B",
                    "matchup_probability": "Peluang bertemu",
                    "team_a_win_probability": "Peluang Tim A menang jika bertemu",
                }
            )
            st.caption("Rincian pasangan yang mungkin bertemu pada setiap babak:")
            st.dataframe(
                pairs.style.format(
                    {
                        "Peluang bertemu": "{:.2%}",
                        "Peluang Tim A menang jika bertemu": "{:.2%}",
                    }
                ),
                hide_index=True,
                width="stretch",
                height=350,
            )
        else:
            st.write(
                "Seluruh hasil sudah dikonfirmasi. Tidak ada pertandingan tersisa untuk diprediksi."
            )
        st.caption(
            f"Model: {report['model_name']} · "
            f"{report['terminal_scenario_count']} kemungkinan hasil tersisa · "
            f"Status format: {report['format_confirmation']['status']}."
        )
        for assumption in report["assumptions"]:
            st.caption(assumption)


def _archived_playoff_match_comparison(data: dict[str, Any]) -> pd.DataFrame | None:
    teams = data.get("preplayoff_teams")
    report = data.get("preplayoff_report")
    actuals = data.get("playoff_matches")
    if teams is None or teams.empty or report is None or actuals is None:
        return None
    favorite = teams.sort_values(["champion_probability", "team_id"], ascending=[False, True]).iloc[
        0
    ]
    path = report.get("champion_paths", {}).get(str(favorite["team_id"]))
    if not path:
        return None
    return build_playoff_match_comparison(pd.DataFrame(path["matches"]), actuals)


def _render_archived_playoff_prediction(data: dict[str, Any]) -> bool:
    """Keep the original model choice visible even after the actual final is known."""
    teams = data.get("preplayoff_teams")
    report = data.get("preplayoff_report")
    if teams is None or teams.empty or report is None:
        st.warning(
            "Arsip prediksi sebelum playoff tidak tersedia. "
            "Hasil aktual tidak digunakan untuk membuat prediksi lama pengganti."
        )
        return False

    favorite = teams.sort_values(["champion_probability", "team_id"], ascending=[False, True]).iloc[
        0
    ]
    team_id = str(favorite["team_id"])
    champion = data["playoff_report"]["confirmed_champion"]
    st.subheader("Prediksi juara model sebelum playoff")
    first, second, third = st.columns(3)
    first.metric("Pilihan model", team_id)
    second.metric("Peluang juara sebelum playoff", _percent(favorite["champion_probability"]))
    third.metric("Juara aktual", champion or "Belum ditentukan")
    st.caption(f"Prediksi tersimpan tanggal {report['as_of']}; bukan prediksi ulang setelah final.")
    if champion:
        verdict = "tepat" if champion == team_id else "tidak tepat"
        st.write(f"Pilihan model {team_id} {verdict}; juara aktual adalah {champion}.")

    if report["format_confirmation"]["status"] != "official":
        st.warning(
            "Prediksi awal memakai asumsi jalur semifinal lama yang berbeda dari bracket aktual. "
            "Bagan arsip dipertahankan, bukan diubah mengikuti hasil."
        )
    config = report.get("simulation_config") or data.get("historical_simulation_config")
    champion_path = report.get("champion_paths", {}).get(team_id)
    if config is None or not champion_path:
        st.info("Pilihan model tersedia, tetapi bagan prediksi aslinya belum tersedia.")
        return True

    seeds = teams[["seed", "team_id"]].rename(columns={"seed": "projected_seed"}).copy()
    seeds["projected_seed_probability"] = 1.0  # Final RS seeds were known at this archive.
    scenario = pd.DataFrame(champion_path["matches"])
    comparison = _archived_playoff_match_comparison(data)
    st.subheader("Bagan prediksi model")
    st.caption(
        f"▸ Pemenang pada contoh jalur jika {team_id} menjadi juara, bukan hasil pasti. "
        "Persentase pada pertandingan adalah peluang menang jika pasangan tersebut bertemu."
    )
    st.plotly_chart(
        build_playoff_bracket_figure(seeds, config, scenario, comparison=comparison),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )
    if comparison is not None:
        st.subheader("Perbandingan prediksi dan hasil")
        st.caption(
            "Hijau: sesuai skenario. Merah: pemenang berbeda. Kuning: pasangan tim berbeda. "
            "Ini perbandingan skenario arsip, bukan akurasi prediksi pre-match playoff."
        )
        st.dataframe(build_playoff_comparison_table(comparison), hide_index=True, width="stretch")
    return True


def _render_playoff_page(data: dict[str, Any], snapshot_id: str | None = None) -> None:
    st.title("Playoff")
    st.caption(
        "MPL Indonesia Season 18 · Enam tim berebut juara; dua teratas langsung ke semifinal."
    )
    selected = snapshot_id or _latest_snapshot_id(data)
    current = data["predictions"].loc[data["predictions"]["snapshot_id"].eq(selected)]
    regular = data["matches"].loc[data["matches"]["snapshot_id"].eq(selected)]
    seeds = build_projected_playoff_seeds(data["predictions"], selected)
    regular_remaining = int(regular["status"].eq("scheduled").sum())
    report = (
        data.get("playoff_report")
        if (not regular_remaining and selected == _latest_snapshot_id(data))
        else None
    )
    scenario = None
    if report is not None:
        current = data["playoff_teams"]
    favorite = current.sort_values("champion_probability", ascending=False).iloc[0]
    confirmed = bool(report and report["confirmed_champion"])

    if confirmed and _render_archived_playoff_prediction(data):
        with st.expander("Lihat hasil aktual dan skor playoff"):
            _render_final_summary(data)
            st.caption(f"Hasil terkonfirmasi per {report['as_of']}, bukan prediksi model.")
            st.subheader("Bagan hasil aktual")
            st.plotly_chart(
                build_playoff_bracket_figure(
                    seeds,
                    data["simulation_config"],
                    data["playoff_matches"],
                    comparison=_archived_playoff_match_comparison(data),
                ),
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
            )
            _render_playoff_forecast_tables(data, data["playoff_matches"], show_details=False)
        return

    if confirmed:
        _render_final_summary(data)
        st.caption(
            f"Seluruh pertandingan playoff selesai · Data per {report['as_of']}. "
            "Persentase prediksi awal tersedia di halaman Hasil Prediksi."
        )
        scenario = data["playoff_matches"]
    else:
        first, second, third = st.columns(3)
        first.metric("Favorit juara", str(favorite["team_id"]))
        second.metric("Peluang juara", _percent(favorite["champion_probability"]))
        third.metric(
            "Pertandingan playoff tersisa" if report is not None else "Pertandingan RS tersisa",
            report["remaining_playoff_match_count"] if report is not None else regular_remaining,
        )
        if report is not None:
            path = report["champion_paths"][str(favorite["team_id"])]
            scenario = pd.DataFrame(path["matches"])
            st.info(
                f"Contoh jalur jika {favorite['team_id']} menjadi juara, bukan hasil pasti. "
                f"Data per {report['as_of']}."
            )
        else:
            st.info(
                "Tim di bagan masih berupa proyeksi. "
                "Hasil pertandingan regular season menentukan peserta dan urutannya."
                if regular_remaining
                else "Regular season selesai. Bagan memakai urutan akhir; "
                "hasil playoff belum tersedia."
            )

    st.subheader("Bagan playoff")
    st.caption(
        "✓ Hasil aktual · ▸ Prediksi pemenang, belum pasti"
        if scenario is not None
        else ("Angka di slot awal menunjukkan peluang tim menempati urutan tersebut.")
    )
    config = (
        data["simulation_config"]
        if report is not None
        else data.get("historical_simulation_config", data["simulation_config"])
    )
    st.plotly_chart(
        build_playoff_bracket_figure(seeds, config, scenario),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )
    if report is not None:
        _render_playoff_forecast_tables(data, scenario)
    else:
        table = seeds[["projected_seed", "team_name", "playoff_probability"]].rename(
            columns={
                "projected_seed": "Urutan",
                "team_name": "Tim",
                "playoff_probability": "Peluang lolos",
            }
        )
        with st.expander("Tim pada bagan"):
            st.dataframe(
                table.style.format({"Peluang lolos": "{:.2%}"}),
                hide_index=True,
                width="stretch",
            )
    with st.expander("Cara membaca bagan"):
        st.write(
            "Babak awal mempertemukan peringkat 3 lawan 6 dan 4 lawan 5. "
            "Tim yang kalah pada babak awal langsung gugur. "
            "Mulai semifinal, tim yang kalah di jalur atas mendapat kesempatan kedua "
            "di jalur bawah. Pemenang kedua jalur bertemu di Grand Final."
        )
        st.caption("BO5: menang 3 game. BO7: menang 4 game. Jadwal menggunakan waktu WIB.")


def _render_prediction_results_page(data: dict[str, Any]) -> None:
    st.title("Hasil Prediksi")
    st.caption(
        "MPL Indonesia Season 18 · Bandingkan prediksi sebelum playoff dengan hasil akhirnya."
    )
    report = data.get("playoff_report")
    if report is None:
        st.info("Ringkasan akan tersedia setelah data playoff diperbarui.")
        return
    _render_final_summary(data)
    st.caption(f"Data per {report['as_of']}. Hasil yang sudah pasti bukan ukuran akurasi prediksi.")

    archived = data.get("preplayoff_teams")
    comparison = None
    table = None
    if archived is None:
        st.warning(
            "Arsip prediksi sebelum playoff tidak tersedia. "
            "Evaluasi juara tidak dibuat dari hasil yang sudah diketahui."
        )
    else:
        comparison = build_championship_comparison(archived, data["playoff_matches"])
        archive_report = data["preplayoff_report"]
        st.subheader("Prediksi awal dibanding hasil akhir")
        st.caption(f"Prediksi asli {archive_report['as_of']}, dibuat sebelum playoff dimulai.")
        if archive_report["format_confirmation"]["status"] != "official":
            st.warning(
                "Prediksi awal memakai asumsi jalur semifinal lama yang berbeda dari "
                "bracket aktual. Angka aslinya tetap dipertahankan."
            )
        table = comparison[
            [
                "team_name",
                "champion_probability",
                "grand_final_probability",
                "actual_result",
            ]
        ].rename(
            columns={
                "team_name": "Tim",
                "champion_probability": "Prediksi juara",
                "grand_final_probability": "Prediksi Grand Final",
                "actual_result": "Hasil aktual",
            }
        )
        st.dataframe(
            table.style.format({"Prediksi juara": "{:.2%}", "Prediksi Grand Final": "{:.2%}"}),
            hide_index=True,
            width="stretch",
        )

    evaluated = _evaluated_matches(data)
    correct = int(evaluated["prediction_correct"].sum())
    st.subheader("Seberapa tepat prediksinya?")
    if comparison is not None:
        favorite = str(comparison.iloc[0]["team_id"])
        champion = report["confirmed_champion"]
        champion_row = comparison.loc[comparison["team_id"].eq(champion)]
        first, second, third = st.columns(3)
        first.metric("Prediksi Sebelum Playoff", favorite)
        second.metric(
            "Peluang awal juara aktual",
            _percent(champion_row.iloc[0]["champion_probability"])
            if not champion_row.empty
            else "—",
        )
        third.metric(
            "Akurasi pertandingan RS",
            _percent(correct / len(evaluated)) if len(evaluated) else "—",
        )
        if not champion_row.empty:
            outcome = "menjadi juara" if favorite == champion else "tidak menjadi juara"
            rank = int(champion_row.iloc[0]["forecast_rank"])
            st.write(
                f"Favorit awal {favorite} {outcome}. "
                f"Juara aktual {champion} berada di urutan favorit {rank} sebelum playoff."
            )
        else:
            st.caption("Pemenang turnamen belum diketahui; evaluasi juara menunggu hasil final.")
    else:
        st.metric(
            "Akurasi pertandingan RS",
            _percent(correct / len(evaluated)) if len(evaluated) else "—",
        )
    st.caption(
        f"{correct} dari {len(evaluated)} prediksi pertandingan regular season benar. "
        "Playoff tidak termasuk dalam angka akurasi ini."
    )

    with st.expander("Riwayat peluang dan unduh data"):
        st.plotly_chart(_history_chart(data["predictions"]), width="stretch")
        if table is not None:
            st.download_button(
                "Unduh perbandingan (CSV)",
                table.to_csv(index=False).encode("utf-8-sig"),
                file_name="season18_prediksi_dan_hasil.csv",
                mime="text/csv",
            )
        st.caption(
            "Setiap pertandingan RS dinilai satu kali dari prediksi sebelum laga. "
            "Skenario pasangan playoff tidak diperlakukan sebagai prediksi pre-match aktual. "
            "Angka 100% setelah final berarti hasil terkonfirmasi, bukan model akurat 100%."
        )
        st.caption(f"Model: {report['model_name']}. Evaluasi juara mencakup satu turnamen.")
    st.markdown("Sumber hasil: [Jadwal resmi MPL Indonesia](https://id-mpl.com/id/schedule).")


def main() -> None:
    st.set_page_config(
        page_title="MPL S18 Predictor",
        page_icon="🏆",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    data, missing = load_dashboard_data(get_project_paths())
    if missing:
        st.title("MPL Indonesia Season 18 Champion Predictor")
        st.error("Data prediksi belum siap. Jalankan pembaruan data terlebih dahulu.")
        with st.expander("Petunjuk pembaruan"):
            st.code("make update-season18")
            for path in missing:
                st.write(path)
        return

    pages = [
        st.Page(
            lambda: _render_regular_season_page(data),
            title="Regular Season",
            icon="📊",
            url_path="regular-season",
            default=True,
        ),
        st.Page(
            lambda: _render_playoff_page(data),
            title="Playoff",
            icon="🏆",
            url_path="playoff",
        ),
        st.Page(
            lambda: _render_prediction_results_page(data),
            title="Hasil Prediksi",
            icon="🎯",
            url_path="hasil-prediksi",
        ),
    ]
    st.navigation(pages, position="top").run()


if __name__ == "__main__":
    main()

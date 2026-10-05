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

from mpl_predictor.config import ProjectPaths, get_project_paths  # noqa: E402


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
    return frames, []


def _snapshot_label(row: pd.Series) -> str:
    if row["prediction_type"] == "preseason":
        return f"Pramusim · cutoff {row['feature_cutoff_date']}"
    if row["prediction_type"] == "as_of":
        return f"As-of {row['feature_cutoff_date']} · Week {int(row['partial_week'])} parsial"
    return f"Week {int(row['completed_week'])} · cutoff {row['feature_cutoff_date']}"


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
    figure.update_layout(coloraxis_showscale=False, yaxis_title=None, xaxis_tickformat=".0%")
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
    projected_seeds: pd.DataFrame, simulation_config: dict[str, Any]
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
        figure.add_shape(
            type="rect",
            x0=x - half_width,
            x1=x + half_width,
            y0=y - half_height,
            y1=y + half_height,
            line={"color": "#4b5563", "width": 1},
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
        figure.add_annotation(
            x=x - half_width,
            y=y + half_height + 0.13,
            text=f"{short_match_labels[match_id]} · BO{int(match['best_of'])}",
            showarrow=False,
            xanchor="left",
            font={"size": 9, "color": "#a1a1aa"},
        )

    def add_connector(
        source: str,
        target: str,
        *,
        source_offset: float = 0.0,
        target_offset: float = 0.0,
        lane: float | None = None,
        color: str = "#71717a",
    ) -> None:
        source_x, source_y = positions[source]
        target_x, target_y = positions[target]
        start_x = source_x + half_width
        end_x = target_x - half_width
        bend_x = lane if lane is not None else (start_x + end_x) / 2
        start_y = source_y + source_offset
        end_y = target_y + target_offset
        for x0, y0, x1, y1 in (
            (start_x, start_y, bend_x, start_y),
            (bend_x, start_y, bend_x, end_y),
            (bend_x, end_y, end_x, end_y),
        ):
            figure.add_shape(
                type="line",
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
    add_connector("play_in_1", "upper_semifinal_1", target_offset=-0.26, color=upper_color)
    add_connector("play_in_2", "upper_semifinal_2", target_offset=-0.26, color=upper_color)
    add_connector("upper_semifinal_1", "upper_final", target_offset=0.26, color=upper_color)
    add_connector("upper_semifinal_2", "upper_final", target_offset=-0.26, color=upper_color)
    add_connector(
        "upper_semifinal_1",
        "lower_semifinal",
        target_offset=0.26,
        lane=5.75,
        color=lower_color,
    )
    add_connector(
        "upper_semifinal_2",
        "lower_semifinal",
        target_offset=-0.26,
        lane=5.95,
        color=lower_color,
    )
    add_connector("upper_final", "lower_final", target_offset=0.26, color=lower_color)
    add_connector("lower_semifinal", "lower_final", target_offset=-0.26, color=lower_color)
    add_connector("upper_final", "grand_final", target_offset=0.26, color=final_color)
    add_connector("lower_final", "grand_final", target_offset=-0.26, color=final_color)

    figure.update_layout(
        height=650,
        margin={"l": 5, "r": 5, "t": 5, "b": 5},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis={"visible": False, "range": [-0.20, 12.00], "fixedrange": True},
        yaxis={"visible": False, "range": [0.65, 10.85], "fixedrange": True},
        hovermode=False,
    )
    return figure


def _render_overview(data: dict[str, Any], snapshot_id: str) -> None:
    predictions = data["predictions"]
    matches = data["matches"]
    current = predictions.loc[predictions["snapshot_id"].eq(snapshot_id)].sort_values(
        "champion_probability", ascending=False
    )
    current_matches = matches.loc[matches["snapshot_id"].eq(snapshot_id)]
    leader = current.iloc[0]

    first, second, third, fourth = st.columns(4)
    first.metric(
        "Favorit juara", str(leader["team_name"]), _percent(leader["champion_probability"])
    )
    second.metric("Match selesai", int(current_matches["status"].eq("completed").sum()), "/ 72")
    third.metric("Match tersisa", int(current_matches["status"].eq("scheduled").sum()))
    fourth.metric("Peluang playoff favorit", _percent(leader["playoff_probability"]))

    left, right = st.columns([1.45, 1])
    with left:
        st.subheader("Probabilitas juara")
        st.plotly_chart(_prediction_chart(current), width="stretch")
    with right:
        st.subheader("Ranking simulasi")
        table = current[
            [
                "champion_rank",
                "team_name",
                "champion_probability",
                "playoff_probability",
                "expected_regular_rank",
            ]
        ].rename(
            columns={
                "champion_rank": "Rank",
                "team_name": "Tim",
                "champion_probability": "Juara",
                "playoff_probability": "Playoff",
                "expected_regular_rank": "Ekspektasi rank RS",
            }
        )
        st.dataframe(
            table.style.format(
                {"Juara": "{:.2%}", "Playoff": "{:.2%}", "Ekspektasi rank RS": "{:.2f}"}
            ),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Perubahan probabilitas dari pramusim")
    st.plotly_chart(_history_chart(predictions), width="stretch")


def _render_matches(data: dict[str, Any], snapshot_id: str) -> None:
    matches = data["matches"]
    snapshot_matches = matches.loc[matches["snapshot_id"].eq(snapshot_id)].sort_values(
        ["scheduled_at", "official_match_id"]
    )
    evaluated = snapshot_matches.loc[
        snapshot_matches["status"].eq("completed") & snapshot_matches["prediction_correct"].notna()
    ]
    correct_count = int(evaluated["prediction_correct"].sum())
    accuracy = correct_count / len(evaluated) if len(evaluated) else None
    base_correct_count = int(evaluated["base_prediction_correct"].sum())
    base_accuracy = base_correct_count / len(evaluated) if len(evaluated) else None
    learned_result_count = int(snapshot_matches["online_learning_update_applied"].sum())
    final_scale = (
        float(snapshot_matches["online_learning_scale_after_update"].iloc[-1])
        if not snapshot_matches.empty
        else 1.0
    )

    st.subheader("Prediksi dan akurasi pertandingan")
    first, second, third, fourth = st.columns(4)
    first.metric("Prediksi dievaluasi", len(evaluated))
    second.metric("Prediksi benar", correct_count)
    third.metric("Akurasi", _percent(accuracy) if accuracy is not None else "-")
    fourth.metric("Hasil dipelajari", learned_result_count)
    st.caption(
        "Status akurasi membandingkan favorit pre-match dengan hasil aktual. Setelah hasil "
        "tersedia, residual kesalahan melatih kalibrasi online dan hasil memperbarui Elo/form "
        "untuk pertandingan berikutnya. Hasil pertandingan sendiri tidak pernah masuk ke "
        "prediksi pre-match-nya."
    )
    if base_accuracy is not None:
        st.caption(
            f"Akurasi base: {_percent(base_accuracy)} · akurasi adaptif: "
            f"{_percent(accuracy)} · confidence scale terbaru: {final_scale:.3f}. "
            "Koefisien model utama tetap dilatih ulang secara terpisah dan walk-forward."
        )

    filter_options = ["Semua", "Selesai", "Akan datang"]
    selected_filter = st.radio(
        "Tampilkan pertandingan",
        options=filter_options,
        index=1 if len(evaluated) else 2,
        horizontal=True,
        key=f"match_filter_{snapshot_id}",
    )
    if selected_filter == "Selesai":
        current = snapshot_matches.loc[snapshot_matches["status"].eq("completed")].copy()
    elif selected_filter == "Akan datang":
        current = snapshot_matches.loc[snapshot_matches["status"].eq("scheduled")].copy()
    else:
        current = snapshot_matches.copy()
    if current.empty:
        st.info("Tidak ada pertandingan untuk filter ini pada snapshot yang dipilih.")
        return
    current["accuracy_label"] = current["accuracy_status"].map(
        {
            "correct": "✅ Benar",
            "incorrect": "❌ Salah",
            "pending_result": "⏳ Belum dimainkan",
        }
    )
    current["learning_label"] = current["result_update_status"].map(
        {
            "incorporated_after_pre_match_prediction": "✅ Dipelajari setelah prediksi",
            "awaiting_result": "⏳ Menunggu hasil",
        }
    )
    table = current[
        [
            "week",
            "scheduled_at",
            "team_a_id",
            "team_b_id",
            "base_team_a_win_probability",
            "team_a_win_probability",
            "predicted_winner_team_id",
            "winner_team_id",
            "accuracy_label",
            "online_learning_observation_count",
            "learning_label",
        ]
    ].rename(
        columns={
            "week": "Week",
            "scheduled_at": "Jadwal",
            "team_a_id": "Team A",
            "team_b_id": "Team B",
            "base_team_a_win_probability": "P awal Team A",
            "team_a_win_probability": "P adaptif Team A",
            "predicted_winner_team_id": "Prediksi adaptif",
            "winner_team_id": "Hasil aktual",
            "accuracy_label": "Status akurasi",
            "online_learning_observation_count": "Hasil terdahulu dipelajari",
            "learning_label": "Pembaruan state",
        }
    )
    st.dataframe(
        table.style.format({"P awal Team A": "{:.2%}", "P adaptif Team A": "{:.2%}"}),
        hide_index=True,
        width="stretch",
    )


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
            }
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


def _render_regular_season_page(data: dict[str, Any], snapshot_id: str) -> None:
    st.title("MPL Indonesia Season 18 · Regular Season")
    st.caption("Prediksi leakage-safe · calibrated match model · Monte Carlo 20.000 iterasi")
    overview_tab, matches_tab, model_tab, explanation_tab, data_tab = st.tabs(
        ["Ringkasan", "Pertandingan", "Perbandingan model", "Explainability", "Data & asumsi"]
    )
    with overview_tab:
        _render_overview(data, snapshot_id)
    with matches_tab:
        _render_matches(data, snapshot_id)
    with model_tab:
        _render_model_comparison(data)
    with explanation_tab:
        _render_explainability(data)
    with data_tab:
        _render_data_scope(data)


def _render_playoff_page(data: dict[str, Any], snapshot_id: str) -> None:
    st.title("MPL Indonesia Season 18 · Playoff")
    st.caption("Bracket 6 tim · double elimination setelah quarterfinal · BO5 dan BO7")
    predictions = data["predictions"]
    matches = data["matches"]
    current = predictions.loc[predictions["snapshot_id"].eq(snapshot_id)].copy()
    current_matches = matches.loc[matches["snapshot_id"].eq(snapshot_id)]
    projected_seeds = build_projected_playoff_seeds(predictions, snapshot_id)
    remaining_matches = int(current_matches["status"].eq("scheduled").sum())
    champion_favorite = current.sort_values("champion_probability", ascending=False).iloc[0]
    final_favorite = current.sort_values("grand_final_probability", ascending=False).iloc[0]

    first, second, third, fourth = st.columns(4)
    first.metric(
        "Slot playoff", int(data["simulation_config"]["regular_season"]["playoff_team_count"])
    )
    second.metric("Match RS tersisa", remaining_matches)
    third.metric(
        "Favorit Grand Final",
        str(final_favorite["team_id"]),
        _percent(final_favorite["grand_final_probability"]),
    )
    fourth.metric(
        "Favorit juara",
        str(champion_favorite["team_id"]),
        _percent(champion_favorite["champion_probability"]),
    )

    if remaining_matches:
        st.info(
            "Regular season belum selesai. Tim pada bagan adalah proyeksi seed berdasarkan "
            "ekspektasi ranking simulasi, bukan peserta playoff yang sudah resmi."
        )
    else:
        st.success("Regular season selesai; seed pada bagan memakai ranking akhir simulasi.")

    st.subheader("Bracket playoff")
    st.caption(
        "Persentase pada slot seed adalah peluang tim finis tepat pada seed tersebut. "
        "Garis biru menunjukkan jalur upper bracket, oranye jalur lower bracket, dan kuning "
        "jalur menuju Grand Final."
    )
    bracket_figure = build_playoff_bracket_figure(projected_seeds, data["simulation_config"])
    st.plotly_chart(
        bracket_figure,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

    st.subheader("Proyeksi enam besar")
    seed_table = projected_seeds[
        [
            "projected_seed",
            "team_name",
            "projected_seed_probability",
            "playoff_probability",
            "grand_final_probability",
            "champion_probability",
            "expected_regular_rank",
        ]
    ].rename(
        columns={
            "projected_seed": "Seed",
            "team_name": "Tim",
            "projected_seed_probability": "Peluang seed",
            "playoff_probability": "Peluang playoff",
            "grand_final_probability": "Peluang Grand Final",
            "champion_probability": "Peluang juara",
            "expected_regular_rank": "Ekspektasi rank RS",
        }
    )
    st.dataframe(
        seed_table.style.format(
            {
                "Peluang seed": "{:.2%}",
                "Peluang playoff": "{:.2%}",
                "Peluang Grand Final": "{:.2%}",
                "Peluang juara": "{:.2%}",
                "Ekspektasi rank RS": "{:.2f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )


def main() -> None:
    st.set_page_config(page_title="MPL S18 Predictor", page_icon="🏆", layout="wide")
    paths = get_project_paths()
    data, missing = load_dashboard_data(paths)
    if missing:
        st.title("MPL Indonesia Season 18 Champion Predictor")
        st.error("Output prediksi belum lengkap.")
        st.code("make update-season18")
        with st.expander("File yang belum tersedia"):
            for path in missing:
                st.write(path)
        return

    snapshot_rows = (
        data["predictions"][
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
    labels = {str(row["snapshot_id"]): _snapshot_label(row) for _, row in snapshot_rows.iterrows()}
    selected_snapshot = st.sidebar.selectbox(
        "Snapshot prediksi",
        options=list(labels),
        index=len(labels) - 1,
        format_func=labels.get,
    )
    regular_season_page = st.Page(
        lambda: _render_regular_season_page(data, selected_snapshot),
        title="Regular Season",
        icon="📊",
        url_path="regular-season",
        default=True,
    )
    playoff_page = st.Page(
        lambda: _render_playoff_page(data, selected_snapshot),
        title="Playoff",
        icon="🏆",
        url_path="playoff",
    )
    navigation = st.navigation([regular_season_page, playoff_page], position="top")
    navigation.run()


if __name__ == "__main__":
    main()

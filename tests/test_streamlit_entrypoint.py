import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from mpl_predictor.config import get_project_paths
from mpl_predictor.dashboard import build_regular_match_table, load_dashboard_data
from streamlit_app import load_dashboard_main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_cloud_entrypoint_loads_src_package() -> None:
    dashboard_main = load_dashboard_main()
    assert dashboard_main.__module__ == "mpl_predictor.dashboard"
    assert dashboard_main.__name__ == "main"


def _app_for_page(renderer: str) -> AppTest:
    return AppTest.from_string(f"""
from mpl_predictor.config import get_project_paths
from mpl_predictor.dashboard import {renderer}, load_dashboard_data
data, missing = load_dashboard_data(get_project_paths())
assert not missing
{renderer}(data)
""").run(timeout=30)


def test_streamlit_default_page_is_simple_and_uses_latest_period() -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py")).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Regular Season"
    assert not app.sidebar.selectbox
    assert not app.radio
    assert not app.tabs
    assert app.selectbox[0].label == "Periode prediksi"
    assert app.selectbox[0].value == "S18_W09"
    assert any(expander.label == "Lihat minggu sebelumnya" for expander in app.expander)
    tables = [x.value for x in app.dataframe if "Status akurasi" in x.value.columns]
    assert len(tables[0]) == 8
    assert len(tables[1]) == 72
    assert list(tables[0].columns) == [
        "Week",
        "Jadwal",
        "Pertandingan",
        "Prediksi",
        "Skor",
        "Status akurasi",
    ]
    for table in tables:
        assert table["Week"].is_monotonic_decreasing
        for _, rows in table.groupby("Week"):
            dates = pd.to_datetime(rows["Jadwal"], format="%d-%m-%Y %H:%M")
            assert dates.is_monotonic_decreasing
    assert any(
        expander.label == "Tentang model, pembelajaran, dan data" for expander in app.expander
    )


def test_regular_history_is_optional_and_keeps_future_scores_hidden() -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py")).run(timeout=30)
    app.selectbox[0].set_value("S18_W03").run(timeout=30)
    assert not app.exception
    metrics = {x.label: x.value for x in app.metric}
    assert metrics["Pertandingan selesai"] == "24/72"
    tables = [x.value for x in app.dataframe if "Status akurasi" in x.value.columns]
    upcoming = next(
        table for table in tables if table["Status akurasi"].eq("Belum dimainkan").all()
    )
    assert upcoming["Week"].min() == 4
    assert upcoming["Skor"].eq("—").all()
    assert any("Akhir minggu 3" in x.value for x in app.caption)
    app.selectbox[0].set_value("S18_PRE").run(timeout=30)
    assert not app.exception
    metrics = {x.label: x.value for x in app.metric}
    assert metrics["Pertandingan selesai"] == "0/72"
    assert metrics["Akurasi prediksi"] == "—"


def test_compact_match_table_uses_the_predicted_teams_probability_without_mutating_input():
    data, _ = load_dashboard_data(get_project_paths())
    matches = data["matches"].head(1).copy()
    matches["team_a_win_probability"] = 0.2
    matches["predicted_winner_team_id"] = matches["team_b_id"]
    matches["status"] = "scheduled"
    matches["accuracy_status"] = "pending_result"
    matches["team_a_score"] = 0
    matches["team_b_score"] = 2
    original = matches.copy(deep=True)
    table = build_regular_match_table(matches)
    assert table.iloc[0]["Prediksi"] == f"{matches.iloc[0]['team_b_id']} (80.00%)"
    assert table.iloc[0]["Skor"] == "—"
    assert table.iloc[0]["Status akurasi"] == "Belum dimainkan"
    pd.testing.assert_frame_equal(matches, original)


def test_streamlit_playoff_keeps_original_model_choice_and_bracket_visible_after_final() -> None:
    app = _app_for_page("_render_playoff_page")
    assert not app.exception
    assert app.title[0].value == "Playoff"
    assert "Prediksi juara model sebelum playoff" in [x.value for x in app.subheader]
    assert "Bagan prediksi model" in [x.value for x in app.subheader]
    assert "Perbandingan prediksi dan hasil" in [x.value for x in app.subheader]
    assert not app.radio
    assert not app.selectbox
    assert not app.tabs
    assert len(app.get("plotly_chart")) == 2
    metrics = {x.label: x.value for x in app.metric}
    assert metrics["Pilihan model"] == "TLID"
    assert metrics["Peluang juara sebelum playoff"] == "27.46%"
    assert metrics["Juara aktual"] == "BTR"
    assert metrics["Runner-up"] == "NAVI"
    assert metrics["Skor Grand Final"] == "NAVI 3-4 BTR"
    table = next(x.value for x in app.dataframe if "Pemenang" in x.value.columns)
    assert len(table) == 8
    final = table.iloc[-1]
    assert final["Babak"] == "Grand Final"
    assert final["Pertandingan"] == "NAVI vs BTR"
    assert final["Skor"] == "3-4"
    assert "Peluang juara playoff" not in [x.value for x in app.subheader]
    assert any(x.label == "Lihat hasil aktual dan skor playoff" for x in app.expander)
    assert any("Pilihan model TLID tidak tepat" in x.value for x in app.markdown)
    assert any("jalur semifinal lama" in x.value for x in app.warning)
    figures = [json.loads(x.proto.spec) for x in app.get("plotly_chart")]
    for figure in figures:
        annotations = {item.get("name"): item for item in figure["layout"]["annotations"]}
        assert "Tidak sesuai skenario" in annotations["comparison-status-play_in_1"]["text"]
        assert "Sesuai skenario" in annotations["comparison-status-play_in_2"]["text"]
        assert "Pasangan berbeda" in annotations["comparison-status-grand_final"]["text"]
    comparison_table = next(x.value for x in app.dataframe if "Perbandingan" in x.value.columns)
    assert len(comparison_table) == 8
    assert comparison_table["Perbandingan"].eq("Pasangan berbeda").sum() == 5
    assert any("bukan akurasi prediksi pre-match" in x.value for x in app.caption)
    predicted_final = [
        annotation["text"]
        for annotation in figures[0]["layout"]["annotations"]
        if abs(annotation.get("x", 0) - 9.47) < 1e-8
        and min(abs(annotation.get("y", 0) - y) for y in (5.66, 5.14)) < 1e-8
    ]
    # The favorite's archived path ends in TLID winning, not the best single ONIC path
    # and not the now-known BTR result.
    assert any("▸ TLID" in text for text in predicted_final)
    assert all("✓" not in annotation["text"] for annotation in figures[0]["layout"]["annotations"])
    assert any("✓ BTR" in annotation["text"] for annotation in figures[1]["layout"]["annotations"])
    assert len(predicted_final) == 2
    actual_expander = next(
        x for x in app.expander if x.label == "Lihat hasil aktual dan skor playoff"
    )
    assert len(actual_expander.get("plotly_chart")) == 1
    assert all(x.label != "Pilihan model" for x in actual_expander.metric)


def test_playoff_without_archive_does_not_present_actual_champion_as_prediction() -> None:
    app = AppTest.from_string("""
from mpl_predictor.config import get_project_paths
from mpl_predictor.dashboard import _render_playoff_page, load_dashboard_data
data, _ = load_dashboard_data(get_project_paths())
data.pop("preplayoff_teams", None)
data.pop("preplayoff_report", None)
_render_playoff_page(data)
""").run(timeout=30)
    assert not app.exception
    assert any("Arsip prediksi sebelum playoff tidak tersedia" in x.value for x in app.warning)
    metrics = {x.label: x.value for x in app.metric}
    assert metrics["Juara aktual"] == "BTR"
    assert "Pilihan model" not in metrics
    assert len(app.get("plotly_chart")) == 1


def test_playoff_without_archived_config_does_not_use_corrected_format_for_old_forecast() -> None:
    app = AppTest.from_string("""
from mpl_predictor.config import get_project_paths
from mpl_predictor.dashboard import _render_playoff_page, load_dashboard_data
data, _ = load_dashboard_data(get_project_paths())
data.pop("historical_simulation_config", None)
data["preplayoff_report"].pop("simulation_config", None)
_render_playoff_page(data)
""").run(timeout=30)
    assert not app.exception
    metrics = {x.label: x.value for x in app.metric}
    assert metrics["Pilihan model"] == "TLID"
    assert metrics["Peluang juara sebelum playoff"] == "27.46%"
    assert any("bagan prediksi aslinya belum tersedia" in x.value for x in app.info)
    assert len(app.get("plotly_chart")) == 1  # Actual bracket only.


def test_playoff_missing_favorite_path_does_not_substitute_another_champions_path() -> None:
    app = AppTest.from_string("""
from mpl_predictor.config import get_project_paths
from mpl_predictor.dashboard import _render_playoff_page, load_dashboard_data
data, _ = load_dashboard_data(get_project_paths())
data["preplayoff_teams"] = data["preplayoff_teams"].sort_values("champion_probability")
data["preplayoff_report"]["champion_paths"] = {}
_render_playoff_page(data)
""").run(timeout=30)
    assert not app.exception
    metrics = {x.label: x.value for x in app.metric}
    assert metrics["Pilihan model"] == "TLID"
    assert any("bagan prediksi aslinya belum tersedia" in x.value for x in app.info)
    assert len(app.get("plotly_chart")) == 1


def test_streamlit_prediction_results_are_compact_and_use_archived_probabilities() -> None:
    app = _app_for_page("_render_prediction_results_page")
    assert not app.exception
    assert app.title[0].value == "Hasil Prediksi"
    assert not app.radio
    assert not app.selectbox
    assert not app.tabs
    metrics = {x.label: x.value for x in app.metric}
    assert metrics == {
        "Juara aktual": "BTR",
        "Runner-up": "NAVI",
        "Skor Grand Final": "NAVI 3-4 BTR",
        "Peluang awal juara aktual": "12.41%",
        "Favorit sebelum playoff": "TLID",
        "Akurasi pertandingan RS": "58.33%",
    }
    comparison = next(x.value for x in app.dataframe if "Prediksi juara" in x.value.columns)
    assert len(comparison.columns) == 4
    assert comparison["Prediksi juara"].max() < 0.3
    assert comparison.loc[comparison["Hasil aktual"].eq("Juara"), "Tim"].iloc[0] == (
        "Bigetron by Vitality"
    )
    assert any("jalur semifinal lama" in x.value for x in app.warning)
    assert any("bukan model akurat 100%" in x.value for x in app.caption)
    assert any("42 dari 72" in x.value for x in app.caption)
    assert any("Favorit awal TLID tidak menjadi juara" in x.value for x in app.markdown)


def test_streamlit_results_without_archive_do_not_invent_historical_accuracy() -> None:
    app = AppTest.from_string("""
from mpl_predictor.config import get_project_paths
from mpl_predictor.dashboard import _render_prediction_results_page, load_dashboard_data
data, _ = load_dashboard_data(get_project_paths())
data.pop("preplayoff_teams", None)
data.pop("preplayoff_report", None)
_render_prediction_results_page(data)
""").run(timeout=30)
    assert not app.exception
    assert any("Arsip prediksi sebelum playoff tidak tersedia" in x.value for x in app.warning)
    assert all(x.label != "Favorit sebelum playoff" for x in app.metric)


def test_streamlit_preplayoff_automatically_shows_one_clearly_labeled_prediction() -> None:
    app = AppTest.from_string("""
import json
import pandas as pd
from mpl_predictor.config import get_project_paths
from mpl_predictor.dashboard import _render_playoff_page, load_dashboard_data
paths = get_project_paths()
data, _ = load_dashboard_data(paths)
archive = paths.data / "season18/playoff_prediction_snapshots/2026-10-19"
data["playoff_report"] = json.loads((archive / "report.json").read_text())
data["simulation_config"] = json.loads((archive / "simulation_config.json").read_text())
for key in ("teams", "matches", "matchups"):
    data[f"playoff_{key}"] = pd.read_csv(archive / f"{key}.csv")
_render_playoff_page(data)
""").run(timeout=30)
    assert not app.exception
    assert not app.radio
    assert not app.selectbox
    metrics = {x.label: x.value for x in app.metric}
    assert metrics["Favorit juara"] == "TLID"
    assert metrics["Pertandingan playoff tersisa"] == "8"
    assert any("bukan hasil pasti" in x.value for x in app.info)
    table = next(x.value for x in app.dataframe if "Pemenang / prediksi" in x.value.columns)
    assert table["Status"].eq("Prediksi, belum pasti").all()
    assert table["Skor"].eq("—").all()
    assert table.iloc[-1]["Pemenang / prediksi"] == "TLID"


def test_playoff_projection_does_not_leak_final_results_into_regular_season_history() -> None:
    app = AppTest.from_string("""
from mpl_predictor.config import get_project_paths
from mpl_predictor.dashboard import _render_playoff_page, load_dashboard_data
data, _ = load_dashboard_data(get_project_paths())
_render_playoff_page(data, "S18_W08")
""").run(timeout=30)
    assert not app.exception
    assert any("masih berupa proyeksi" in x.value for x in app.info)
    assert all(x.label != "Juara aktual" for x in app.metric)
    assert len(app.get("plotly_chart")) == 1


def test_results_page_handles_unavailable_playoff_data():
    app = AppTest.from_string("""
from mpl_predictor.config import get_project_paths
from mpl_predictor.dashboard import _render_prediction_results_page, load_dashboard_data
data, _ = load_dashboard_data(get_project_paths())
data.pop("playoff_report", None)
_render_prediction_results_page(data)
""").run(timeout=30)
    assert not app.exception
    assert any("setelah data playoff diperbarui" in x.value for x in app.info)
    assert not app.metric

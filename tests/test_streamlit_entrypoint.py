from pathlib import Path

from streamlit.testing.v1 import AppTest

from streamlit_app import load_dashboard_main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_cloud_entrypoint_loads_src_package() -> None:
    dashboard_main = load_dashboard_main()

    assert dashboard_main.__module__ == "mpl_predictor.dashboard"
    assert dashboard_main.__name__ == "main"


def test_streamlit_default_page_renders_regular_season() -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py")).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "MPL Indonesia Season 18 · Regular Season"
    assert app.sidebar.selectbox[0].value == "S18_W07"
    for match_filter in ("Selesai", "Semua", "Akan datang"):
        app.radio[0].set_value(match_filter).run(timeout=30)
        assert not app.exception
        table = next(
            element.value for element in app.dataframe if "Status akurasi" in element.value.columns
        )
        assert table["Week"].is_monotonic_decreasing
        for _, week_rows in table.groupby("Week"):
            assert week_rows["Jadwal"].is_monotonic_decreasing


def test_streamlit_playoff_page_renders_bracket() -> None:
    app = AppTest.from_string(
        """
from mpl_predictor.config import get_project_paths
from mpl_predictor.dashboard import _render_playoff_page, load_dashboard_data

data, missing = load_dashboard_data(get_project_paths())
assert not missing
_render_playoff_page(data, "S18_W07")
"""
    ).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "MPL Indonesia Season 18 · Playoff"
    assert "Bracket playoff" in [heading.value for heading in app.subheader]
    assert "Proyeksi enam besar" in [heading.value for heading in app.subheader]
    assert len(app.get("plotly_chart")) == 1

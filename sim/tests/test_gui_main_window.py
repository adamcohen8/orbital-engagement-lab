from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtWidgets
except ImportError as exc:
    pytest.skip(f"PySide6 QtWidgets unavailable: {exc}", allow_module_level=True)
QApplication = QtWidgets.QApplication
QMessageBox = QtWidgets.QMessageBox

from sim.app.services import dump_config_text, validate_config  # noqa: E402
from sim.config.object_refs import configured_objects  # noqa: E402
from sim.gui.main_window import MainWindow  # noqa: E402


def _write_minimal_review_store(output_dir: Path) -> None:
    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True)
    (output_dir / "index.md").write_text("# Review Output\n", encoding="utf-8")
    with sqlite3.connect(review_dir / "run.sqlite") as conn:
        conn.execute(
            "CREATE TABLE run_metadata (scenario_name TEXT, duration_s REAL, dt_s REAL, samples INTEGER, "
            "oel_version TEXT, review_schema_version TEXT)"
        )
        conn.execute("INSERT INTO run_metadata VALUES ('gui_evidence_studio_smoke', 2.0, 1.0, 3, 'test', '0.3')")
        conn.execute("CREATE TABLE relative_state (time_s REAL, deputy_id TEXT, chief_id TEXT, range_km REAL)")
        conn.executemany(
            "INSERT INTO relative_state VALUES (?, 'chaser', 'target', ?)",
            [(0.0, 1.0), (1.0, 0.6), (2.0, 0.25)],
        )
        conn.execute("CREATE TABLE artifacts (artifact_type TEXT, artifact_id TEXT, path TEXT)")
        conn.execute("INSERT INTO artifacts VALUES ('summary', 'index', 'index.md')")


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def main_window(qt_app: QApplication) -> MainWindow:
    window = MainWindow()
    yield window
    window.deleteLater()


def test_gui_object_edits_update_canonical_objects(main_window: MainWindow) -> None:
    main_window.target_dry_mass.setValue(12345.0)
    main_window._set_combo_data_or_text(main_window.chaser_init_mode, "relative_ric_curv")
    main_window.chaser_init_values[1].setValue(-42.0)

    collected = main_window._collect_config_from_widgets()
    validated = validate_config(collected)
    objects = configured_objects(validated)

    assert objects["target"].specs["dry_mass_kg"] == 12345.0
    assert objects["chaser"].initial_state["relative_to_target_ric"]["state"][1] == -42.0
    assert collected["objects"]["target"]["specs"]["dry_mass_kg"] == 12345.0
    assert collected["objects"]["chaser"]["initial_state"]["relative_to_target_ric"]["state"][1] == -42.0


def test_save_applies_unapplied_advanced_yaml(main_window: MainWindow, tmp_path: Path) -> None:
    save_path = tmp_path / "yaml_saved.yaml"
    cfg = main_window._collect_config_from_widgets()
    cfg["scenario_name"] = "yaml_editor_wins"
    main_window.save_path_edit.setText(str(save_path))
    main_window.yaml_editor.setPlainText(dump_config_text(cfg))

    main_window._on_save()

    saved = save_path.read_text(encoding="utf-8")
    assert "scenario_name: yaml_editor_wins" in saved
    assert not main_window._yaml_has_unapplied_changes
    assert not main_window.is_dirty


def test_run_auto_save_updates_window_state(
    main_window: MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_path = tmp_path / "run_saved.yaml"
    started: list[Path] = []
    main_window.save_path_edit.setText(str(save_path))
    main_window.output_dir_edit.setText(str(tmp_path / "outputs"))
    main_window.scenario_name_edit.setText("run_auto_save")

    monkeypatch.setattr(main_window, "_start_run_worker", lambda path: started.append(Path(path)))

    main_window._on_run()

    assert started == [save_path]
    assert main_window.loaded_config_path == save_path
    assert main_window.current_config["scenario_name"] == "run_auto_save"
    assert not main_window.is_dirty


def test_advanced_yaml_tab_hidden_until_advanced_mode(main_window: MainWindow) -> None:
    yaml_index = 5

    assert not main_window.tabs.isTabVisible(yaml_index)

    main_window.advanced_mode_check.setChecked(True)

    assert main_window.tabs.isTabVisible(yaml_index)


def test_manual_run_confirms_existing_output_folder(
    main_window: MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("old", encoding="utf-8")
    cfg = main_window._collect_config_from_widgets()
    cfg["outputs"]["mode"] = "save"
    cfg["outputs"]["output_dir"] = str(output_dir)

    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.No)

    assert not main_window._confirm_output_overwrite(cfg)


def test_results_tab_selects_start_here_first(main_window: MainWindow, tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "control_effort.png").write_bytes(b"not really a png")
    (output_dir / "master_run_summary.json").write_text('{"scenario_name": "demo"}', encoding="utf-8")
    (output_dir / "index.md").write_text("# Start Here\n", encoding="utf-8")
    main_window.output_mode_combo.setCurrentText("save")
    main_window.output_dir_edit.setText(str(output_dir))

    main_window._refresh_output_files()

    assert main_window.output_files.currentItem().text() == "Start Here (md)"
    assert "Open index.md first" in main_window.results_summary.toPlainText()


def test_evidence_studio_advanced_query_saves_custom_figure_with_provenance(
    main_window: MainWindow, tmp_path: Path
) -> None:
    output_dir = tmp_path / "review_output"
    _write_minimal_review_store(output_dir)

    main_window.open_output_review(output_dir)
    main_window.review_query_editor.setPlainText("SELECT time_s, range_km FROM relative_state ORDER BY time_s")
    main_window._run_review_query()

    assert main_window.review_query_table.rowCount() == 3
    assert main_window.review_plot_x_combo.currentText() == "time_s"
    assert main_window.review_plot_y_combo.currentText() == "range_km"
    assert main_window.review_plot_preview_button.isEnabled()
    assert main_window.review_query_save_figure_button.isEnabled()

    main_window.review_plot_style_combo.setCurrentIndex(1)
    main_window.review_plot_artifact_id_edit.setText("gui_custom_range")
    main_window._save_review_query_figure()

    figure_path = output_dir / "review" / "figures" / "gui_custom_range.png"
    manifest_path = output_dir / "review" / "generated_artifacts.json"
    assert figure_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifacts"][-1]["artifact_id"] == "gui_custom_range"
    assert manifest["artifacts"][-1]["style_name"] == "oel_light"


def test_evidence_studio_review_only_mode_opens_studio(qt_app: QApplication, tmp_path: Path) -> None:
    output_dir = tmp_path / "review_output"
    _write_minimal_review_store(output_dir)
    window = MainWindow(output_dir=output_dir)
    try:
        assert window.review_only_mode is True
        assert window.windowTitle().endswith(" - OEL Evidence Studio")
        assert window.results_tabs.tabText(window.results_tabs.currentIndex()) == "Evidence Studio"
        assert window.review_query_run_button.isEnabled()
        assert not (output_dir / "evidence_studio_workspace").exists()
        explorer_items = [window.output_files.item(row).text() for row in range(window.output_files.count())]
        assert "Table: relative_state" in explorer_items
        assert any(item.startswith("Plot Recipe: Relative range") for item in explorer_items)
        assert window.evidence_agent_box.title() == "Custom Evidence Plots"
    finally:
        window.deleteLater()


def test_evidence_studio_plot_builder_button_and_fullscreen(
    qt_app: QApplication, tmp_path: Path
) -> None:
    output_dir = tmp_path / "review_output"
    _write_minimal_review_store(output_dir)
    window = MainWindow(output_dir=output_dir)
    try:
        window._open_evidence_plot_builder()

        assert window.results_tabs.tabText(window.results_tabs.currentIndex()) == "Advanced Query"
        assert "Advanced Query" in window.evidence_agent_status.text()

        window._toggle_evidence_viewer_fullscreen()

        assert not window.evidence_left_panel.isVisible()
        assert window.evidence_fullscreen_button.text() == "Exit Fullscreen"
    finally:
        window.deleteLater()


def test_evidence_studio_refresh_shows_api_generated_artifact(qt_app: QApplication, tmp_path: Path) -> None:
    output_dir = tmp_path / "review_output"
    _write_minimal_review_store(output_dir)
    window = MainWindow(output_dir=output_dir)
    try:
        figure_dir = output_dir / "review" / "figures"
        figure_dir.mkdir(parents=True)
        figure_path = figure_dir / "api_custom_note.md"
        figure_path.write_text("# API Output\n", encoding="utf-8")

        window._refresh_output_files()

        explorer_items = [window.output_files.item(row).text() for row in range(window.output_files.count())]
        assert "Api Custom Note (md)" in explorer_items
        window._select_output_path(figure_path)
        assert "API Output" in window.preview_text.toPlainText()
    finally:
        window.deleteLater()

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from sim.app.services import dump_config_text, validate_config  # noqa: E402
from sim.config.object_refs import configured_objects  # noqa: E402
from sim.gui.main_window import MainWindow  # noqa: E402


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

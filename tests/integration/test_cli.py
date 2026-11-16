import subprocess
import sys
from pathlib import Path


def test_status_runs_outside_project_with_default_settings(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "industrial_vision.cli", "status"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Industrial Vision AI | Phase 0 | log level: INFO" in result.stdout
    assert result.stderr == ""


def test_config_enables_debug_logging(tmp_path: Path) -> None:
    config_file = tmp_path / "application.toml"
    config_file.write_text('log_level = "DEBUG"\n', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "industrial_vision.cli",
            "--config",
            config_file.name,
            "status",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "log level: DEBUG" in result.stdout
    assert "DEBUG __main__: Configuration loaded" in result.stderr


def test_missing_config_reports_cli_error(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "industrial_vision.cli",
            "--config",
            "missing.toml",
            "status",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Cannot load configuration" in result.stderr
    assert "Traceback" not in result.stderr

import tomllib
from pathlib import Path

import pytest

from industrial_vision.config import load_config


def test_load_config_uses_defaults() -> None:
    assert load_config().log_level == "INFO"


def test_load_config_accepts_case_insensitive_log_level(tmp_path: Path) -> None:
    config_file = tmp_path / "application.toml"
    config_file.write_text('log_level = "debug"\n', encoding="utf-8")

    assert load_config(config_file).log_level == "DEBUG"


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ('log_level = "verbose"', "log_level must be one of"),
        ("log_level = 10", "log_level must be one of"),
        ('log_levle = "INFO"', "Unknown settings: log_levle"),
    ],
)
def test_load_config_rejects_invalid_settings(
    tmp_path: Path, contents: str, message: str
) -> None:
    config_file = tmp_path / "application.toml"
    config_file.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(config_file)


def test_load_config_rejects_malformed_toml(tmp_path: Path) -> None:
    config_file = tmp_path / "application.toml"
    config_file.write_text("log_level = [", encoding="utf-8")

    with pytest.raises(tomllib.TOMLDecodeError):
        load_config(config_file)

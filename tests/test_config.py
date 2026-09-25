from pathlib import Path

import pytest

from imdb_sentiment.config import ConfigError, load_data_config


def test_project_data_config() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_data_config()

    assert config.path.as_posix() == "data/aclImdb"
    assert config.validation_ratio == 0.10
    assert config.train_sizes == (1000, 5000, 10000, 22500)
    assert config.seed == 42
    assert config.dataset_path == (root / "data" / "aclImdb").resolve()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("data:\n  path: data/aclImdb\n", "Missing data configuration field"),
        (
            "data:\n  path: C:\\absolute\\aclImdb\n  validation_ratio: 0.10\n"
            "  train_sizes: [1000]\n  seed: 42\n",
            "data.path must be relative",
        ),
        (
            "data:\n  path: data/aclImdb\n  validation_ratio: 1.0\n"
            "  train_sizes: [1000]\n  seed: 42\n",
            "data.validation_ratio",
        ),
    ],
)
def test_invalid_config_has_clear_error(tmp_path: Path, content: str, message: str) -> None:
    (tmp_path / "config.yaml").write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_data_config(tmp_path)

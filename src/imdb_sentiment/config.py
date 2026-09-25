"""Read and validate the project's dataset configuration."""

from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """The project configuration is missing or invalid."""


@dataclass(frozen=True)
class DataConfig:
    path: Path
    dataset_path: Path
    validation_ratio: float
    train_sizes: tuple[int, ...]
    seed: int


def load_data_config(project_root: Path | None = None) -> DataConfig:
    """Load config.yaml and resolve data.path against the project root."""
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    config_file = root / "config.yaml"
    if not config_file.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_file}")

    try:
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_file}: {exc}") from exc

    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ConfigError("config.yaml must contain a 'data' mapping")
    data = config["data"]
    required = ("path", "validation_ratio", "train_sizes", "seed")
    missing = [key for key in required if key not in data]
    if missing:
        raise ConfigError(f"Missing data configuration field(s): {', '.join(missing)}")

    raw_path = data["path"]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ConfigError("data.path must be a nonempty relative path")
    path = Path(raw_path)
    if path.is_absolute() or path.drive:
        raise ConfigError("data.path must be relative to the project root")
    dataset_path = (root / path).resolve()
    if not dataset_path.is_relative_to(root):
        raise ConfigError("data.path must stay within the project root")

    ratio = data["validation_ratio"]
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 < ratio < 1:
        raise ConfigError("data.validation_ratio must be a number between 0 and 1")

    sizes = data["train_sizes"]
    if (not isinstance(sizes, list) or not sizes or
            any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % 2
                for size in sizes) or
            sizes != sorted(set(sizes))):
        raise ConfigError("data.train_sizes must be increasing, unique, positive even integers")

    seed = data["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError("data.seed must be an integer")

    return DataConfig(path, dataset_path, float(ratio), tuple(sizes), seed)

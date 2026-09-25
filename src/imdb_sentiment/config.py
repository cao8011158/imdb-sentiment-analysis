"""Read and validate the project's dataset configuration."""

from dataclasses import dataclass
from math import isfinite
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


@dataclass(frozen=True)
class TokenAnalysisConfig:
    models: dict[str, str]
    thresholds: tuple[int, ...]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_labels: int


@dataclass(frozen=True)
class TrainingConfig:
    max_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    learning_rate_candidates: tuple[float, ...]
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    max_epochs: int
    early_stopping_patience: int
    early_stopping_threshold: float
    metric_for_best_model: str
    seed: int
    bf16: bool
    group_by_length: bool
    save_total_limit: int

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


def _read_config(project_root: Path | None) -> tuple[Path, dict]:
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    config_file = root / "config.yaml"
    if not config_file.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_file}")

    try:
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_file}: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError("config.yaml must contain a mapping")
    return root, config


def load_data_config(project_root: Path | None = None) -> DataConfig:
    """Load config.yaml and resolve data.path against the project root."""
    root, config = _read_config(project_root)

    if not isinstance(config.get("data"), dict):
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


def load_token_analysis_config(project_root: Path | None = None) -> TokenAnalysisConfig:
    """Read candidate tokenizer names and reporting thresholds."""
    _, config = _read_config(project_root)
    analysis = config.get("token_analysis")
    if not isinstance(analysis, dict):
        raise ConfigError("config.yaml must contain a 'token_analysis' mapping")
    models = analysis.get("models")
    if (not isinstance(models, dict) or not models or
            any(not isinstance(key, str) or not key.strip() or
                not isinstance(value, str) or not value.strip()
                for key, value in models.items())):
        raise ConfigError("token_analysis.models must map names to nonempty model IDs")
    thresholds = analysis.get("thresholds")
    if (not isinstance(thresholds, list) or not thresholds or
            any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in thresholds) or
            thresholds != sorted(set(thresholds))):
        raise ConfigError("token_analysis.thresholds must be increasing positive integers")
    return TokenAnalysisConfig(dict(models), tuple(thresholds))


def load_model_config(project_root: Path | None = None) -> ModelConfig:
    """Read the selected sequence-classification model from config.yaml."""
    _, config = _read_config(project_root)
    model = config.get("model")
    if not isinstance(model, dict):
        raise ConfigError("config.yaml must contain a 'model' mapping")
    name = model.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("model.name must be a nonempty model ID")
    num_labels = model.get("num_labels")
    if isinstance(num_labels, bool) or not isinstance(num_labels, int) or num_labels <= 0:
        raise ConfigError("model.num_labels must be a positive integer")
    return ModelConfig(name, num_labels)


def load_training_config(project_root: Path | None = None) -> TrainingConfig:
    """Read training settings through the project's existing YAML reader."""
    _, config = _read_config(project_root)
    raw = config.get("training")
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml must contain a 'training' mapping")
    missing = [name for name in TrainingConfig.__dataclass_fields__ if name not in raw]
    if missing:
        raise ConfigError(f"Missing training configuration field(s): {', '.join(missing)}")

    positive_ints = (
        "max_length", "per_device_train_batch_size", "per_device_eval_batch_size",
        "gradient_accumulation_steps", "max_epochs", "save_total_limit",
    )
    for name in positive_ints:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"training.{name} must be a positive integer")
    patience = raw["early_stopping_patience"]
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
        raise ConfigError("training.early_stopping_patience must be a nonnegative integer")
    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError("training.seed must be an integer")

    def finite_number(name: str, *, positive: bool = False) -> float:
        value = raw[name]
        if isinstance(value, bool):
            raise ConfigError(f"training.{name} must be a finite number")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"training.{name} must be a finite number") from exc
        if not isfinite(number) or number < 0 or (positive and number == 0):
            qualifier = "positive" if positive else "nonnegative"
            raise ConfigError(f"training.{name} must be a finite {qualifier} number")
        return number

    learning_rate = finite_number("learning_rate", positive=True)
    rates = raw["learning_rate_candidates"]
    if not isinstance(rates, list) or not rates:
        raise ConfigError("training.learning_rate_candidates must be a nonempty list")
    try:
        candidates = tuple(float(rate) for rate in rates)
    except (TypeError, ValueError) as exc:
        raise ConfigError("training.learning_rate_candidates must contain numbers") from exc
    if (any(isinstance(rate, bool) for rate in rates) or
            any(not isfinite(rate) or rate <= 0 for rate in candidates) or
            len(candidates) != len(set(candidates))):
        raise ConfigError("training.learning_rate_candidates must be unique positive numbers")

    weight_decay = finite_number("weight_decay")
    warmup_ratio = finite_number("warmup_ratio")
    if warmup_ratio >= 1:
        raise ConfigError("training.warmup_ratio must be less than 1")
    threshold = finite_number("early_stopping_threshold")
    if raw["lr_scheduler_type"] != "linear":
        raise ConfigError("training.lr_scheduler_type must be 'linear'")
    if not isinstance(raw["metric_for_best_model"], str) or not raw["metric_for_best_model"].strip():
        raise ConfigError("training.metric_for_best_model must be a nonempty string")
    if raw["metric_for_best_model"] != "f1":
        raise ConfigError("training.metric_for_best_model must be 'f1' for this binary task")
    for name in ("bf16", "group_by_length"):
        if not isinstance(raw[name], bool):
            raise ConfigError(f"training.{name} must be a boolean")

    return TrainingConfig(
        raw["max_length"], raw["per_device_train_batch_size"],
        raw["per_device_eval_batch_size"], raw["gradient_accumulation_steps"],
        learning_rate, candidates, weight_decay, warmup_ratio,
        raw["lr_scheduler_type"], raw["max_epochs"], patience, threshold,
        raw["metric_for_best_model"], seed, raw["bf16"],
        raw["group_by_length"], raw["save_total_limit"],
    )

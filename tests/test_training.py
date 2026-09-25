from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from imdb_sentiment.config import ConfigError, load_training_config
from imdb_sentiment.dataset import Review
from imdb_sentiment.training import (
    EncodedReviews, SequentialEvalTrainer, _batch_settings, binary_metrics,
    compute_metrics,
)


def test_training_config_uses_existing_yaml_reader() -> None:
    config = load_training_config()
    assert config.max_length == 2048
    assert config.learning_rate_candidates == (1e-5, 2e-5, 3e-5)
    assert config.effective_batch_size == 32
    assert config.metric_for_best_model == "f1"
    assert config.save_total_limit == 2


@pytest.mark.parametrize("field,value", [
    ("max_length", "0"),
    ("per_device_train_batch_size", "0"),
    ("gradient_accumulation_steps", "0"),
    ("learning_rate", "0"),
    ("weight_decay", "-0.1"),
    ("warmup_ratio", "1.0"),
    ("early_stopping_patience", "-1"),
    ("save_total_limit", "0"),
])
def test_invalid_training_config(tmp_path: Path, field: str, value: str) -> None:
    project_config = Path(__file__).resolve().parents[1] / "config.yaml"
    content = project_config.read_text(encoding="utf-8")
    lines = content.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(f"  {field}:"):
            lines[index] = f"  {field}: {value}"
            break
    (tmp_path / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=field):
        load_training_config(tmp_path)


def test_empty_learning_rate_candidates(tmp_path: Path) -> None:
    project_config = Path(__file__).resolve().parents[1] / "config.yaml"
    content = project_config.read_text(encoding="utf-8")
    content = content.replace(
        "  learning_rate_candidates:\n    - 1.0e-5\n    - 2.0e-5\n    - 3.0e-5",
        "  learning_rate_candidates: []",
    )
    (tmp_path / "config.yaml").write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match="learning_rate_candidates"):
        load_training_config(tmp_path)


def test_binary_metrics_use_positive_class_one() -> None:
    result = binary_metrics([0, 0, 1, 1], [0, 1, 1, 0])
    assert result == {
        "accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5,
    }


def test_metric_predictions_argmax_raw_logits() -> None:
    evaluation = SimpleNamespace(
        predictions=np.array([[3.0, -2.0], [-1.0, 2.0]]),
        label_ids=np.array([0, 1]),
    )
    assert compute_metrics(evaluation) == {
        "accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
    }


def test_encoded_reviews_truncate_without_padding() -> None:
    calls = []

    def tokenizer(texts, **kwargs):
        calls.append(kwargs)
        return {
            "input_ids": [[1, 2] for _ in texts],
            "attention_mask": [[1, 1] for _ in texts],
        }

    reviews = (Review("a", "First", 0), Review("b", "Second", 1))
    encoded = EncodedReviews(reviews, tokenizer, 2048)
    assert len(encoded) == 2
    assert encoded[1] == {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": 1}
    assert calls[0]["max_length"] == 2048
    assert calls[0]["truncation"] is True
    assert calls[0]["padding"] is False


def test_micro_batch_must_preserve_effective_batch() -> None:
    config = load_training_config()
    assert _batch_settings(config, 8, 4) == (8, 4)
    assert _batch_settings(config, 16, 2) == (16, 2)
    assert _batch_settings(config, 32, 1) == (32, 1)
    with pytest.raises(ValueError, match="Effective batch size"):
        _batch_settings(config, 32, 2)


def test_validation_sampler_keeps_original_order() -> None:
    reviews = [0, 1, 2]
    sampler = SequentialEvalTrainer._get_eval_sampler(None, reviews)
    assert list(sampler) == [0, 1, 2]

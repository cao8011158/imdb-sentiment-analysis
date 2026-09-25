import csv
from pathlib import Path

import pytest

from imdb_sentiment.config import ConfigError, load_token_analysis_config
from imdb_sentiment.dataset import Review
from imdb_sentiment.token_analysis import (
    measure_lengths, summarize_lengths, summary_columns, write_summary,
)


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append(kwargs)
        # A tiny stand-in: one ID per word plus two special-token IDs.
        return {"input_ids": [[101] + list(range(len(text.split()))) + [102] for text in texts]}


def test_measure_lengths_uses_full_unpadded_text() -> None:
    reviews = (
        Review("train/neg/a.txt", "one two three", 0),
        Review("train/pos/b.txt", "one", 1),
        Review("train/pos/c.txt", "one two", 1),
    )
    tokenizer = RecordingTokenizer()
    assert measure_lengths(reviews, tokenizer, batch_size=2) == [5, 3, 4]
    assert len(tokenizer.calls) == 2
    for call in tokenizer.calls:
        assert call["truncation"] is False
        assert call["padding"] is False
        assert call["add_special_tokens"] is True


def test_percentiles_thresholds_and_csv_schema(tmp_path: Path) -> None:
    thresholds = (2, 4)
    summary = summarize_lengths([1, 2, 3, 4, 5], thresholds)
    assert summary == {
        "count": 5, "mean": 3, "median": 3,
        "p90": pytest.approx(4.6), "p95": pytest.approx(4.8),
        "p99": pytest.approx(4.96), "max": 5,
        "over_2_count": 3, "over_2_pct": 60,
        "over_4_count": 1, "over_4_pct": 20,
    }
    path = tmp_path / "summary.csv"
    write_summary([{"model": "example/model", "split": "validation", **summary}], thresholds, path)
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == summary_columns(thresholds)
        row = next(reader)
        assert row["model"] == "example/model"
        assert row["split"] == "validation"
        assert row["over_2_count"] == "3"
        assert row["over_4_pct"] == "20.0"


def test_token_analysis_config() -> None:
    config = load_token_analysis_config()
    assert config.models == {
        "modernbert": "answerdotai/ModernBERT-base",
        "deberta": "microsoft/deberta-v3-base",
    }
    assert config.thresholds == (512, 1024, 2048)


def test_missing_token_analysis_config_is_clear(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("data: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="token_analysis"):
        load_token_analysis_config(tmp_path)

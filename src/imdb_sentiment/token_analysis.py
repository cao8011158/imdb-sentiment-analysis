"""Measure untruncated IMDb review lengths with candidate tokenizers."""

import csv
from pathlib import Path
from statistics import mean, median

from .config import load_data_config, load_token_analysis_config
from .dataset import Review, prepare_dataset


def measure_lengths(reviews: tuple[Review, ...], tokenizer, batch_size: int = 128) -> list[int]:
    """Include special tokens, without padding or truncating any review."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    lengths = []
    for start in range(0, len(reviews), batch_size):
        batch = reviews[start:start + batch_size]
        encoded = tokenizer(
            [review.text for review in batch],
            truncation=False,
            padding=False,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        input_ids = encoded["input_ids"]
        if len(input_ids) != len(batch):
            raise ValueError("Tokenizer returned an unexpected number of reviews")
        lengths.extend(len(ids) for ids in input_ids)
    return lengths


def _percentile(sorted_lengths: list[int], percentile: int) -> float:
    """Linear interpolation between adjacent ranks (the usual p90/p95/p99 rule)."""
    rank = (len(sorted_lengths) - 1) * percentile / 100
    lower = int(rank)
    fraction = rank - lower
    upper = min(lower + 1, len(sorted_lengths) - 1)
    return sorted_lengths[lower] + fraction * (sorted_lengths[upper] - sorted_lengths[lower])


def summarize_lengths(lengths: list[int], thresholds: tuple[int, ...]) -> dict[str, int | float]:
    if not lengths:
        raise ValueError("Cannot summarize an empty set of token lengths")
    if any(length <= 0 for length in lengths):
        raise ValueError("Token lengths must be positive")
    ordered = sorted(lengths)
    summary: dict[str, int | float] = {
        "count": len(lengths),
        "mean": mean(lengths),
        "median": median(lengths),
        "p90": _percentile(ordered, 90),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "max": ordered[-1],
    }
    for threshold in thresholds:
        count = sum(length > threshold for length in lengths)
        summary[f"over_{threshold}_count"] = count
        summary[f"over_{threshold}_pct"] = 100 * count / len(lengths)
    return summary


def summary_columns(thresholds: tuple[int, ...]) -> list[str]:
    columns = ["model", "split", "count", "mean", "median", "p90", "p95", "p99", "max"]
    for threshold in thresholds:
        columns.extend((f"over_{threshold}_count", f"over_{threshold}_pct"))
    return columns


def write_summary(rows: list[dict], thresholds: tuple[int, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_columns(thresholds))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    from transformers import AutoTokenizer

    data_config = load_data_config()
    analysis_config = load_token_analysis_config()
    dataset = prepare_dataset(data_config)
    splits = {
        "training_pool": dataset.training_pool,
        "validation": dataset.validation,
        "official_test": dataset.official_test,
    }
    output = Path(__file__).resolve().parents[2] / "results" / "token_length_summary.csv"
    rows = []
    for name, model_id in analysis_config.models.items():
        print(f"Loading tokenizer: {name} ({model_id})", flush=True)
        try:
            # Prefer the native tokenizer where available. DeBERTa's fast
            # conversion does not implement SentencePiece byte fallback.
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        except Exception as exc:
            raise RuntimeError(f"Could not load tokenizer {name} ({model_id}): {exc}") from exc
        for split_name, reviews in splits.items():
            lengths = measure_lengths(reviews, tokenizer)
            rows.append({
                "model": model_id,
                "split": split_name,
                **summarize_lengths(lengths, analysis_config.thresholds),
            })
            print(f"  {split_name}: {len(lengths)} reviews", flush=True)
        write_summary(rows, analysis_config.thresholds, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

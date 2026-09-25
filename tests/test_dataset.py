from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from imdb_sentiment.config import load_data_config
from imdb_sentiment.dataset import DatasetError, load_official_dataset, prepare_dataset


def _ids(reviews) -> set[str]:
    return {review.sample_id for review in reviews}


@pytest.fixture(scope="module")
def raw_files() -> dict[str, set[str]]:
    """Count and identify the source files without using the dataset loader."""
    root = load_data_config().dataset_path
    result = {}
    for split in ("train", "test"):
        for class_name in ("pos", "neg"):
            directory = root / split / class_name
            assert directory.is_dir(), f"Missing source directory: {directory}"
            files = [path for path in directory.glob("*.txt") if path.is_file()]
            assert len(files) == 12_500, f"Unexpected raw .txt count in {directory}"
            result[f"{split}/{class_name}"] = {
                path.relative_to(root).as_posix() for path in files
            }
            assert len(result[f"{split}/{class_name}"]) == 12_500
    return result


def test_official_loader_matches_raw_files(raw_files: dict[str, set[str]]) -> None:
    official = load_official_dataset()
    raw_train = raw_files["train/pos"] | raw_files["train/neg"]
    raw_test = raw_files["test/pos"] | raw_files["test/neg"]

    assert len(official.train) == 25_000
    assert len(official.test) == 25_000
    assert _ids(official.train) == raw_train
    assert _ids(official.test) == raw_test
    assert Counter(review.label for review in official.train) == {0: 12_500, 1: 12_500}
    assert Counter(review.label for review in official.test) == {0: 12_500, 1: 12_500}
    for split, reviews in (("train", official.train), ("test", official.test)):
        assert {review.sample_id for review in reviews if review.label == 0} == raw_files[f"{split}/neg"]
        assert {review.sample_id for review in reviews if review.label == 1} == raw_files[f"{split}/pos"]


def test_splits_subsets_and_reproducibility(raw_files: dict[str, set[str]]) -> None:
    config = load_data_config()
    first = prepare_dataset(config)
    second = prepare_dataset(config)

    raw_train = raw_files["train/pos"] | raw_files["train/neg"]
    raw_test = raw_files["test/pos"] | raw_files["test/neg"]
    pool_ids = _ids(first.training_pool)
    validation_ids = _ids(first.validation)
    test_ids = _ids(first.official_test)
    assert len(first.official_train) == 25_000
    assert len(first.training_pool) == 22_500
    assert len(first.validation) == 2_500
    assert len(first.official_test) == 25_000
    assert Counter(review.label for review in first.training_pool) == {0: 11_250, 1: 11_250}
    assert Counter(review.label for review in first.validation) == {0: 1_250, 1: 1_250}
    assert Counter(review.label for review in first.official_test) == {0: 12_500, 1: 12_500}

    for reviews in (
        first.official_train, first.training_pool, first.validation, first.official_test,
        *first.train_subsets.values(),
    ):
        assert len(reviews) == len(_ids(reviews))
        assert {review.label for review in reviews} <= {0, 1}
    assert not pool_ids & validation_ids
    assert not pool_ids & test_ids
    assert not validation_ids & test_ids
    assert pool_ids | validation_ids == raw_train
    assert test_ids == raw_test

    expected_sizes = (1000, 5000, 10000, 22500)
    assert tuple(first.train_subsets) == expected_sizes
    previous_ids: set[str] = set()
    for size in expected_sizes:
        subset = first.train_subsets[size]
        subset_ids = _ids(subset)
        assert len(subset) == size
        assert Counter(review.label for review in subset) == {0: size // 2, 1: size // 2}
        assert previous_ids < subset_ids
        assert subset_ids <= pool_ids
        assert subset_ids == _ids(second.train_subsets[size])
        previous_ids = subset_ids
    assert previous_ids == pool_ids
    assert pool_ids == _ids(second.training_pool)
    assert validation_ids == _ids(second.validation)


def test_missing_dataset_path_is_clear(tmp_path: Path) -> None:
    config = replace(load_data_config(), dataset_path=tmp_path / "missing")
    with pytest.raises(DatasetError, match="dataset path does not exist"):
        load_official_dataset(config)


def test_missing_required_directory_is_clear(tmp_path: Path) -> None:
    config = replace(load_data_config(), dataset_path=tmp_path)
    with pytest.raises(DatasetError, match="Required IMDb directory does not exist.*train.*neg"):
        load_official_dataset(config)

"""Load official IMDb reviews and build reproducible labeled data splits."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import random

from .config import DataConfig, load_data_config


class DatasetError(ValueError):
    """The IMDb data is missing or does not match the expected official data."""


@dataclass(frozen=True)
class Review:
    sample_id: str
    text: str
    label: int


@dataclass(frozen=True)
class OfficialDataset:
    train: tuple[Review, ...]
    test: tuple[Review, ...]


@dataclass(frozen=True)
class PreparedDataset:
    official_train: tuple[Review, ...]
    training_pool: tuple[Review, ...]
    validation: tuple[Review, ...]
    official_test: tuple[Review, ...]
    train_subsets: dict[int, tuple[Review, ...]]


def _check_reviews(reviews: tuple[Review, ...], expected_each: int, name: str) -> None:
    counts = Counter(review.label for review in reviews)
    if set(counts) - {0, 1}:
        raise DatasetError(f"{name} contains a label other than 0 or 1: {counts}")
    if counts != {0: expected_each, 1: expected_each}:
        raise DatasetError(
            f"{name} must contain {expected_each} negative and {expected_each} positive "
            f"reviews; found {dict(counts)}"
        )
    ids = [review.sample_id for review in reviews]
    if len(ids) != len(set(ids)):
        raise DatasetError(f"{name} contains duplicate sample IDs")


def _read_class(dataset_path: Path, split: str, class_name: str, label: int) -> tuple[Review, ...]:
    directory = dataset_path / split / class_name
    if not directory.is_dir():
        raise DatasetError(f"Required IMDb directory does not exist: {directory}")
    files = sorted(
        (file for file in directory.glob("*.txt") if file.is_file()),
        key=lambda file: file.name,
    )
    if len(files) != 12_500:
        raise DatasetError(f"Expected 12500 .txt reviews in {directory}; found {len(files)}")
    try:
        return tuple(
            Review(file.relative_to(dataset_path).as_posix(), file.read_text(encoding="utf-8"), label)
            for file in files
        )
    except OSError as exc:
        raise DatasetError(f"Could not read IMDb review in {directory}: {exc}") from exc


def load_official_dataset(config: DataConfig | None = None) -> OfficialDataset:
    """Load only the four official labeled directories, leaving test intact."""
    config = config or load_data_config()
    root = config.dataset_path
    if not root.is_dir():
        raise DatasetError(f"IMDb dataset path does not exist: {root}")

    train = _read_class(root, "train", "neg", 0) + _read_class(root, "train", "pos", 1)
    test = _read_class(root, "test", "neg", 0) + _read_class(root, "test", "pos", 1)
    _check_reviews(train, 12_500, "Official train")
    _check_reviews(test, 12_500, "Official test")
    if len(train) != 25_000 or len(test) != 25_000:
        raise DatasetError("Official train and test must each contain 25000 reviews")
    if {r.sample_id for r in train} & {r.sample_id for r in test}:
        raise DatasetError("Official train and test sample IDs overlap")
    return OfficialDataset(train, test)


def prepare_dataset(config: DataConfig | None = None) -> PreparedDataset:
    """Create one stratified split and nested balanced training subsets."""
    config = config or load_data_config()
    official = load_official_dataset(config)
    validation_each = 12_500 * config.validation_ratio
    if not validation_each.is_integer() or int(validation_each) != 1_250:
        raise DatasetError(
            "data.validation_ratio must produce 1250 validation reviews per class "
            "from the official training data"
        )
    n_validation = int(validation_each)
    rng = random.Random(config.seed)
    class_reviews = {}
    for label in (0, 1):
        reviews = [review for review in official.train if review.label == label]
        rng.shuffle(reviews)
        class_reviews[label] = reviews

    validation = tuple(
        class_reviews[label][index]
        for label in (0, 1)
        for index in range(n_validation)
    )
    pool_by_label = {
        label: class_reviews[label][n_validation:] for label in (0, 1)
    }
    for label in (0, 1):
        rng.shuffle(pool_by_label[label])
    training_pool = tuple(pool_by_label[0] + pool_by_label[1])

    _check_reviews(training_pool, 11_250, "Training pool")
    _check_reviews(validation, 1_250, "Validation")
    if len(training_pool) != 22_500 or len(validation) != 2_500:
        raise DatasetError("Training pool must contain 22500 and validation 2500 reviews")

    subsets = {}
    for size in config.train_sizes:
        each = size // 2
        if each > 11_250:
            raise DatasetError(f"Training subset size {size} exceeds the 22500-review pool")
        subset = tuple(pool_by_label[0][:each] + pool_by_label[1][:each])
        _check_reviews(subset, each, f"Training subset {size}")
        subsets[size] = subset

    train_ids = {review.sample_id for review in official.train}
    pool_ids = {review.sample_id for review in training_pool}
    validation_ids = {review.sample_id for review in validation}
    test_ids = {review.sample_id for review in official.test}
    if pool_ids & validation_ids or pool_ids & test_ids or validation_ids & test_ids:
        raise DatasetError("Training pool, validation, and official test must not overlap")
    if pool_ids | validation_ids != train_ids:
        raise DatasetError("Training pool and validation must partition official train")
    previous_ids: set[str] = set()
    for size, subset in subsets.items():
        subset_ids = {review.sample_id for review in subset}
        if not previous_ids <= subset_ids or not subset_ids <= pool_ids:
            raise DatasetError(f"Training subset {size} is not nested within the pool")
        previous_ids = subset_ids

    return PreparedDataset(
        official.train, training_pool, validation, official.test, subsets
    )

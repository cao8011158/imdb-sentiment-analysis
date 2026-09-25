"""Evaluate fixed Jev sentiment classification on the existing IMDb splits."""

import argparse
import asyncio
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from math import isfinite
import os
from pathlib import Path
import random
import sqlite3
import statistics
import time

import httpx

from .dataset import PreparedDataset, Review, prepare_dataset
from .training import binary_metrics


BASE_URL = "https://api.typesafe.ai"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "jev"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 5
CSV_COLUMNS = (
    "sample_id", "true_label", "predicted_label", "correct", "confidence",
    "negative_probability", "positive_probability", "latency_ms", "input_tokens",
    "output_tokens", "requested_model", "actual_model", "attempts", "status", "error",
)
QUESTION = {
    "type": "choice",
    "instructions": "Classify the overall sentiment expressed in this IMDb movie review.",
    "criteria": {
        "negative": "The review expresses an overall negative sentiment toward the movie.",
        "positive": "The review expresses an overall positive sentiment toward the movie.",
    },
}


class JevError(RuntimeError):
    """A safe-to-display API, data, or evaluation error."""


def require_api_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise JevError("TYPESAFE_API_KEY environment variable is not set.")
    return key


def make_client(key: str, concurrency: int, *, transport=None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {key}"},
        timeout=httpx.Timeout(90.0, connect=20.0),
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
        transport=transport,
    )


def request_payload(model: str, review_text: str) -> dict:
    """Pass Review.text verbatim; no tokenizer, cleanup, or truncation."""
    return {"model": model, "state": review_text, "questions": {"sentiment": QUESTION}}


def parse_models(payload: object) -> list[dict[str, str]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise JevError("Invalid /v1/models response: models must be a list")
    models = []
    for item in payload["models"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
            raise JevError("Invalid /v1/models response: model name is missing")
        description = item.get("description", "")
        release_date = item.get("release_date", "")
        if not isinstance(description, str) or not isinstance(release_date, str):
            raise JevError("Invalid /v1/models response: model metadata must be text")
        models.append({"name": item["name"], "description": description, "release_date": release_date})
    return models


async def get_models(client: httpx.AsyncClient) -> list[dict[str, str]]:
    try:
        response = await client.get("/v1/models")
    except httpx.RequestError as exc:
        raise JevError(f"/v1/models request failed: {type(exc).__name__}") from None
    if response.status_code != 200:
        raise JevError(f"/v1/models returned HTTP {response.status_code}")
    try:
        return parse_models(response.json())
    except ValueError:
        raise JevError("/v1/models returned invalid JSON") from None


async def verify_model(client: httpx.AsyncClient, model: str) -> None:
    models = await get_models(client)
    if model not in {item["name"] for item in models}:
        raise JevError(f"Requested model {model!r} is unavailable in /v1/models")


def _unit_interval(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevError(f"Invalid sentiment response: {field} must be numeric")
    number = float(value)
    if not isfinite(number) or not 0 <= number <= 1:
        raise JevError(f"Invalid sentiment response: {field} must be in [0, 1]")
    return number


def _token_count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JevError(f"Invalid sentiment response: {field} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class ParsedAnswer:
    predicted_label: int
    confidence: float
    negative_probability: float
    positive_probability: float
    input_tokens: int
    output_tokens: int
    actual_model: str


def parse_answer(payload: object) -> ParsedAnswer:
    if not isinstance(payload, dict):
        raise JevError("Invalid sentiment response: root must be an object")
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise JevError("Invalid sentiment response: model is missing")
    answers = payload.get("answers")
    answer = answers.get("sentiment") if isinstance(answers, dict) else None
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise JevError("Invalid sentiment response: answers.sentiment.type must be choice")
    choice = answer.get("choice")
    if choice not in ("negative", "positive"):
        raise JevError("Invalid sentiment response: choice must be negative or positive")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict):
        raise JevError("Invalid sentiment response: probabilities are missing")
    negative = _unit_interval(probabilities.get("negative"), "probabilities.negative")
    positive = _unit_interval(probabilities.get("positive"), "probabilities.positive")
    confidence = _unit_interval(answer.get("confidence"), "confidence")
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise JevError("Invalid sentiment response: usage is missing")
    input_tokens = _token_count(usage.get("input_tokens"), "usage.input_tokens")
    output_tokens = _token_count(usage.get("output_tokens"), "usage.output_tokens")
    return ParsedAnswer(
        0 if choice == "negative" else 1, confidence, negative, positive,
        input_tokens, output_tokens, model,
    )


@dataclass(frozen=True)
class PredictionRow:
    sample_id: str
    true_label: int
    predicted_label: int | None
    correct: int | None
    confidence: float | None
    negative_probability: float | None
    positive_probability: float | None
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    requested_model: str
    actual_model: str | None
    attempts: int
    status: str
    error: str | None


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    delay = min(30.0, 0.5 * 2 ** (attempt - 1))
    if retry_after:
        try:
            delay = max(0.0, float(retry_after))
        except ValueError:
            try:
                date = parsedate_to_datetime(retry_after)
                delay = max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return min(delay, 60.0) + random.uniform(0, 0.25)


async def infer_review(
    client: httpx.AsyncClient, review: Review, model: str,
    semaphore: asyncio.Semaphore, prior_attempts: int = 0,
    prior_latency_ms: float = 0.0,
) -> PredictionRow:
    """Count only HTTP/parse time, excluding semaphore wait and retry sleep."""
    request = request_payload(model, review.text)
    latency_ms = prior_latency_ms
    error = "No API attempt was made"
    for attempt in range(1, MAX_RETRIES + 2):
        retry_after = None
        retryable = False
        async with semaphore:
            started = time.perf_counter()
            try:
                response = await client.post("/v1/systemone", json=request)
                if response.status_code == 200:
                    try:
                        parsed = parse_answer(response.json())
                    except ValueError:
                        raise JevError("Invalid sentiment response: JSON is malformed") from None
                    return PredictionRow(
                        review.sample_id, review.label, parsed.predicted_label,
                        int(parsed.predicted_label == review.label), parsed.confidence,
                        parsed.negative_probability, parsed.positive_probability,
                        latency_ms + (time.perf_counter() - started) * 1000,
                        parsed.input_tokens, parsed.output_tokens,
                        model, parsed.actual_model, prior_attempts + attempt, "success", None,
                    )
                error = f"HTTP {response.status_code} from /v1/systemone"
                retryable = response.status_code in RETRYABLE_STATUS
                retry_after = response.headers.get("Retry-After")
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                error = f"/v1/systemone request failed: {type(exc).__name__}"
                retryable = True
            except JevError as exc:
                error = str(exc)
            except httpx.RequestError as exc:
                error = f"/v1/systemone request failed: {type(exc).__name__}"
            finally:
                latency_ms += (time.perf_counter() - started) * 1000
        if not retryable or attempt > MAX_RETRIES:
            break
        await asyncio.sleep(_retry_delay(attempt, retry_after))
    return PredictionRow(
        review.sample_id, review.label, None, None, None, None, None,
        latency_ms, None, None, model, None, prior_attempts + attempt, "failed", error,
    )


class ResultStore:
    """Commit every completed sample so interrupted runs can resume safely."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS predictions ("
            "sample_id TEXT PRIMARY KEY, true_label INTEGER NOT NULL, predicted_label INTEGER, "
            "correct INTEGER, confidence REAL, negative_probability REAL, "
            "positive_probability REAL, latency_ms REAL NOT NULL, input_tokens INTEGER, "
            "output_tokens INTEGER, requested_model TEXT NOT NULL, actual_model TEXT, "
            "attempts INTEGER NOT NULL, status TEXT NOT NULL, error TEXT)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get(self, sample_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM predictions WHERE sample_id = ?", (sample_id,)
        ).fetchone()

    def mark_formal_start(self) -> None:
        """Keep the original wall-clock start across interrupted runs."""
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO metadata (key, value) VALUES ('formal_started_at_unix', ?)",
                (str(time.time()),),
            )

    def formal_wall_seconds(self) -> float:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'formal_started_at_unix'"
        ).fetchone()
        if row is None:
            raise JevError("Formal wall-clock start was not recorded")
        return max(0.0, time.time() - float(row["value"]))

    def upsert(self, prediction: PredictionRow) -> None:
        values = asdict(prediction)
        placeholders = ", ".join("?" for _ in CSV_COLUMNS)
        assignments = ", ".join(f"{field}=excluded.{field}" for field in CSV_COLUMNS[1:])
        with self.connection:
            self.connection.execute(
                f"INSERT INTO predictions ({', '.join(CSV_COLUMNS)}) VALUES ({placeholders}) "
                f"ON CONFLICT(sample_id) DO UPDATE SET {assignments}",
                tuple(values[field] for field in CSV_COLUMNS),
            )

    def rows_for(self, reviews: tuple[Review, ...]) -> list[dict]:
        rows = []
        for review in reviews:
            found = self.get(review.sample_id)
            if found is not None:
                rows.append(dict(found))
        return rows


def validate_formal_reviews(reviews: tuple[Review, ...]) -> None:
    if len(reviews) != 25_000:
        raise JevError(f"Formal evaluation requires exactly 25000 official test reviews; found {len(reviews)}")
    ids = {review.sample_id for review in reviews}
    if len(ids) != 25_000 or any(not sample_id.startswith("test/") for sample_id in ids):
        raise JevError("Formal evaluation requires 25000 unique official test sample IDs")
    if sum(review.label == 0 for review in reviews) != 12_500 or sum(review.label == 1 for review in reviews) != 12_500:
        raise JevError("Formal evaluation requires 12500 reviews per sentiment class")


def _pending_reviews(store: ResultStore, reviews: tuple[Review, ...], model: str) -> list[tuple[Review, int, float]]:
    pending = []
    for review in reviews:
        previous = store.get(review.sample_id)
        if previous and previous["requested_model"] != model:
            raise JevError(
                f"Existing result for {review.sample_id} uses a different requested model; "
                "use a separate results database for a different model"
            )
        if previous and previous["true_label"] != review.label:
            raise JevError(f"Existing result for {review.sample_id} has a different true label")
        if previous is None or previous["status"] != "success":
            pending.append((
                review, previous["attempts"] if previous else 0,
                previous["latency_ms"] if previous else 0.0,
            ))
    return pending


async def evaluate_reviews(
    client: httpx.AsyncClient, reviews: tuple[Review, ...], model: str,
    store: ResultStore, concurrency: int, *, formal: bool = False,
) -> None:
    if concurrency <= 0:
        raise JevError("--concurrency must be positive")
    if formal:
        validate_formal_reviews(reviews)
    await verify_model(client, model)
    pending = _pending_reviews(store, reviews, model)
    print(f"{len(reviews)} reviews; {len(reviews) - len(pending)} already successful; {len(pending)} pending")
    if not pending:
        return
    semaphore = asyncio.Semaphore(concurrency)
    queue: asyncio.Queue[tuple[Review, int, float]] = asyncio.Queue()
    for item in pending:
        queue.put_nowait(item)
    if formal:
        store.mark_formal_start()  # Before the first formal POST; persists across resumes.
    completed = 0

    async def worker() -> None:
        nonlocal completed
        while True:
            try:
                review, prior_attempts, prior_latency_ms = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                result = await infer_review(client, review, model, semaphore, prior_attempts, prior_latency_ms)
                store.upsert(result)
                completed += 1
                if completed % 100 == 0 or completed == len(pending):
                    print(f"Jev progress: {completed}/{len(pending)} pending reviews processed")
            finally:
                queue.task_done()

    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(pending)))))


def _write_predictions(rows: list[dict], path: Path) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in CSV_COLUMNS} for row in rows)
    temporary.replace(path)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def formal_summary(rows: list[dict], requested_model: str, concurrency: int, wall_seconds: float) -> dict:
    if len(rows) != 25_000 or any(row["status"] != "success" for row in rows):
        raise JevError("Formal evaluation incomplete: all 25000 official test predictions must succeed")
    if any(row["requested_model"] != requested_model for row in rows):
        raise JevError("Formal evaluation contains a different requested model")
    actual_models = {row["actual_model"] for row in rows}
    if len(actual_models) != 1:
        raise JevError(f"Formal evaluation returned mixed actual models: {sorted(actual_models)}")
    metrics = binary_metrics(
        [row["true_label"] for row in rows],
        [row["predicted_label"] for row in rows],
    )
    latencies = sorted(row["latency_ms"] for row in rows)
    total_input = sum(row["input_tokens"] for row in rows)
    total_output = sum(row["output_tokens"] for row in rows)
    return {
        "requested_model": requested_model,
        "actual_model": next(iter(actual_models)),
        "test_samples": 25_000,
        "successful_predictions": 25_000,
        "failed_predictions": 0,
        **metrics,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "mean_input_tokens": total_input / 25_000,
        "mean_output_tokens": total_output / 25_000,
        "mean_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "min_latency_ms": latencies[0],
        "max_latency_ms": latencies[-1],
        "total_wall_clock_seconds": wall_seconds,
        "throughput_reviews_per_second": 25_000 / wall_seconds if wall_seconds > 0 else 0.0,
        "concurrency": concurrency,
        "classification_config": {
            "instructions": QUESTION["instructions"],
            "negative_criterion": QUESTION["criteria"]["negative"],
            "positive_criterion": QUESTION["criteria"]["positive"],
        },
    }


def finish_formal(store: ResultStore, reviews: tuple[Review, ...], model: str, concurrency: int, output_dir: Path) -> dict:
    validate_formal_reviews(reviews)
    rows = store.rows_for(reviews)
    _write_predictions(rows, output_dir / "jev_predictions.csv")
    successful = sum(row["status"] == "success" for row in rows)
    if len(rows) != 25_000 or successful != 25_000:
        raise JevError(
            f"Formal evaluation incomplete: {successful}/25000 successful, "
            f"{len(rows) - successful} failed, {25_000 - len(rows)} missing. "
            "Partial SQLite/CSV results were preserved; rerun the same formal command to resume."
        )
    summary = formal_summary(rows, model, concurrency, store.formal_wall_seconds())
    path = output_dir / "jev_summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return summary


def _smoke_reviews(prepared: PreparedDataset, limit: int) -> tuple[Review, ...]:
    if not 1 <= limit <= len(prepared.validation):
        raise JevError(f"--limit must be between 1 and {len(prepared.validation)}")
    negatives = [review for review in prepared.validation if review.label == 0]
    positives = [review for review in prepared.validation if review.label == 1]
    return tuple(negatives[: (limit + 1) // 2] + positives[: limit // 2])


async def _run_cli(args: argparse.Namespace) -> None:
    key = require_api_key()
    if args.command == "models":
        async with make_client(key, 1) as client:
            for item in await get_models(client):
                print(f"{item['name']} | {item['description']} | {item['release_date']}")
        return

    if args.concurrency <= 0:
        raise JevError("--concurrency must be positive")
    prepared = prepare_dataset()
    reviews = _smoke_reviews(prepared, args.limit) if args.command == "smoke" else prepared.official_test
    if args.command == "formal":
        validate_formal_reviews(reviews)  # Before even querying the API.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    store = ResultStore(RESULTS_DIR / "jev_results.sqlite")
    try:
        async with make_client(key, args.concurrency) as client:
            await evaluate_reviews(client, reviews, args.model, store, args.concurrency, formal=args.command == "formal")
        if args.command == "formal":
            summary = finish_formal(store, reviews, args.model, args.concurrency, RESULTS_DIR)
            print(f"Formal Jev evaluation complete: {summary['accuracy']:.4f} accuracy, {summary['f1']:.4f} F1")
        else:
            rows = store.rows_for(reviews)
            successes = sum(row["status"] == "success" for row in rows)
            print(f"Validation smoke: {successes}/{len(reviews)} successful; results saved in {RESULTS_DIR / 'jev_results.sqlite'}")
            if successes != len(reviews):
                raise JevError("Validation smoke has failed predictions; inspect SQLite and rerun to retry")
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed Jev sentiment evaluation on IMDb")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("models", help="List available System One models")
    for command in ("smoke", "formal"):
        sub = commands.add_parser(command)
        sub.add_argument("--model", default="jev-latest")
        sub.add_argument("--concurrency", type=int, default=4 if command == "smoke" else 8)
        if command == "smoke":
            sub.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    try:
        asyncio.run(_run_cli(args))
    except JevError as exc:
        parser.exit(1, f"Jev evaluation error: {exc}\n")


if __name__ == "__main__":
    main()

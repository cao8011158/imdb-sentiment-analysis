"""All TypeSafe HTTP traffic in this module uses httpx.MockTransport."""

import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from imdb_sentiment.dataset import Review
from imdb_sentiment import jev_evaluation as jev


def answer(choice="positive", *, model="jev-actual"):
    return {
        "model": model,
        "answers": {"sentiment": {
            "type": "choice", "choice": choice, "confidence": 0.9,
            "probabilities": {"negative": 0.1, "positive": 0.9},
        }},
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }


def mock_client(handler):
    return jev.make_client("unit-test-token", 2, transport=httpx.MockTransport(handler))


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(jev.JevError, match="TYPESAFE_API_KEY environment variable is not set"):
        jev.require_api_key()


def test_model_list_parsing_and_mocked_get():
    def handler(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"models": [
            {"name": "jev-latest", "description": "Current", "release_date": "2026-01-01"}
        ]})

    async def scenario():
        async with mock_client(handler) as client:
            models = await jev.get_models(client)
            await jev.verify_model(client, "jev-latest")
            with pytest.raises(jev.JevError, match="unavailable"):
                await jev.verify_model(client, "wrong-model")
            return models

    assert asyncio.run(scenario()) == [
        {"name": "jev-latest", "description": "Current", "release_date": "2026-01-01"}
    ]
    with pytest.raises(jev.JevError, match="models must be a list"):
        jev.parse_models({"data": []})


def test_request_payload_keeps_raw_review():
    original = "<br />An UNchanged review!\nSecond line."
    assert jev.request_payload("jev-latest", original) == {
        "model": "jev-latest",
        "state": original,
        "questions": {"sentiment": {
            "type": "choice",
            "instructions": "Classify the overall sentiment expressed in this IMDb movie review.",
            "criteria": {
                "negative": "The review expresses an overall negative sentiment toward the movie.",
                "positive": "The review expresses an overall positive sentiment toward the movie.",
            },
        }},
    }


@pytest.mark.parametrize("choice,label", [("negative", 0), ("positive", 1)])
def test_choice_mapping(choice, label):
    parsed = jev.parse_answer(answer(choice))
    assert parsed.predicted_label == label
    assert parsed.confidence == 0.9
    assert parsed.negative_probability == 0.1
    assert parsed.positive_probability == 0.9
    assert parsed.input_tokens == 12
    assert parsed.output_tokens == 3
    assert parsed.actual_model == "jev-actual"


@pytest.mark.parametrize("mutation", [
    lambda data: data["answers"]["sentiment"].update(type="text"),
    lambda data: data["answers"]["sentiment"].update(choice="neutral"),
    lambda data: data["answers"]["sentiment"].update(confidence=True),
    lambda data: data["answers"]["sentiment"].update(confidence=1.1),
    lambda data: data["answers"]["sentiment"]["probabilities"].pop("negative"),
    lambda data: data["answers"]["sentiment"]["probabilities"].update(positive=-0.1),
    lambda data: data["usage"].update(input_tokens=-1),
    lambda data: data.update(model=""),
])
def test_invalid_choice_response_is_rejected(mutation):
    data = deepcopy(answer())
    mutation(data)
    with pytest.raises(jev.JevError, match="Invalid sentiment response"):
        jev.parse_answer(data)


def test_transient_http_error_retries_and_uses_retry_after(monkeypatch):
    attempts = 0
    delays = []

    def handler(request):
        nonlocal attempts
        attempts += 1
        assert json.loads(request.content)["state"] == "RAW"
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json=answer())

    async def no_wait(seconds):
        delays.append(seconds)

    monkeypatch.setattr(jev.asyncio, "sleep", no_wait)
    monkeypatch.setattr(jev.random, "uniform", lambda _a, _b: 0.0)

    async def scenario():
        async with mock_client(handler) as client:
            return await jev.infer_review(client, Review("train/pos/1.txt", "RAW", 1), "jev-latest", asyncio.Semaphore(1))

    result = asyncio.run(scenario())
    assert attempts == 2
    assert delays == [2.0]
    assert result.status == "success"
    assert result.attempts == 2
    assert result.correct == 1
    assert result.latency_ms >= 0


@pytest.mark.parametrize("status", [401, 403, 422])
def test_permanent_http_errors_do_not_retry(status):
    attempts = 0

    def handler(_request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(status)

    async def scenario():
        async with mock_client(handler) as client:
            return await jev.infer_review(client, Review("train/neg/1.txt", "bad", 0), "jev-latest", asyncio.Semaphore(1))

    result = asyncio.run(scenario())
    assert attempts == 1
    assert result.status == "failed"
    assert result.attempts == 1
    assert result.error == f"HTTP {status} from /v1/systemone"


def test_transient_retries_are_bounded(monkeypatch):
    attempts = 0

    def handler(_request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(503)

    async def no_wait(_seconds):
        pass

    monkeypatch.setattr(jev.asyncio, "sleep", no_wait)

    async def scenario():
        async with mock_client(handler) as client:
            return await jev.infer_review(client, Review("train/neg/1.txt", "text", 0), "jev-latest", asyncio.Semaphore(1))

    result = asyncio.run(scenario())
    assert attempts == jev.MAX_RETRIES + 1
    assert result.attempts == attempts
    assert result.status == "failed"


def test_resume_skips_success_and_retries_failure(tmp_path):
    reviews = (Review("train/neg/1.txt", "first", 0), Review("train/pos/1.txt", "second", 1))
    posts = []
    phase = 1

    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"models": [{"name": "jev-latest"}, {"name": "other-model"}]})
        payload = json.loads(request.content)
        posts.append(payload["state"])
        if payload["state"] == "second" and phase == 1:
            return httpx.Response(422)
        return httpx.Response(200, json=answer("negative" if payload["state"] == "first" else "positive"))

    store = jev.ResultStore(tmp_path / "results.sqlite")
    try:
        async def scenario():
            async with mock_client(handler) as client:
                await jev.evaluate_reviews(client, reviews, "jev-latest", store, 2)
                assert [row["status"] for row in store.rows_for(reviews)] == ["success", "failed"]
                nonlocal phase
                phase = 2
                await jev.evaluate_reviews(client, reviews, "jev-latest", store, 2)
                with pytest.raises(jev.JevError, match="different requested model"):
                    await jev.evaluate_reviews(client, reviews, "other-model", store, 2)

        asyncio.run(scenario())
        assert posts == ["first", "second", "second"]
        rows = store.rows_for(reviews)
        assert all(row["status"] == "success" for row in rows)
        assert [row["attempts"] for row in rows] == [1, 2]
        assert len(rows) == 2
    finally:
        store.close()


def test_formal_requires_complete_official_test_before_api(tmp_path):
    reviews = tuple(Review(f"test/neg/{index}.txt", "", 0) for index in range(12_500)) + tuple(
        Review(f"test/pos/{index}.txt", "", 1) for index in range(12_500)
    )
    jev.validate_formal_reviews(reviews)
    with pytest.raises(jev.JevError, match="exactly 25000"):
        jev.validate_formal_reviews(reviews[:-1])
    requests = []

    def handler(request):
        requests.append(request)
        raise AssertionError("Incomplete formal run must make no HTTP requests")

    async def scenario():
        async with mock_client(handler) as client:
            store = jev.ResultStore(tmp_path / "results.sqlite")
            try:
                with pytest.raises(jev.JevError, match="exactly 25000"):
                    await jev.evaluate_reviews(client, reviews[:-1], "jev-latest", store, 2, formal=True)
            finally:
                store.close()

    asyncio.run(scenario())
    assert requests == []


def test_formal_summary_uses_binary_metrics_and_rejects_mixed_models():
    rows = [
        {
            "status": "success", "true_label": index // 12_500,
            "predicted_label": index // 12_500, "actual_model": "jev-actual",
            "requested_model": "jev-latest",
            "latency_ms": 10.0, "input_tokens": 5, "output_tokens": 2,
        }
        for index in range(25_000)
    ]
    summary = jev.formal_summary(rows, "jev-latest", 8, 100.0)
    assert {name: summary[name] for name in ("accuracy", "precision", "recall", "f1")} == {
        "accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
    }
    assert summary["total_input_tokens"] == 125_000
    assert summary["total_output_tokens"] == 50_000
    assert summary["throughput_reviews_per_second"] == 250.0
    assert summary["classification_config"]["instructions"] == jev.QUESTION["instructions"]
    rows[0]["actual_model"] = "another-model"
    with pytest.raises(jev.JevError, match="mixed actual models"):
        jev.formal_summary(rows, "jev-latest", 8, 100.0)


def test_incomplete_formal_preserves_partial_csv_without_summary(tmp_path):
    reviews = tuple(Review(f"test/neg/{index}.txt", "", 0) for index in range(12_500)) + tuple(
        Review(f"test/pos/{index}.txt", "", 1) for index in range(12_500)
    )
    store = jev.ResultStore(tmp_path / "results.sqlite")
    try:
        store.upsert(jev.PredictionRow(
            reviews[0].sample_id, 0, 0, 1, 0.9, 0.9, 0.1,
            12.0, 10, 2, "jev-latest", "jev-actual", 1, "success", None,
        ))
        with pytest.raises(jev.JevError, match="incomplete"):
            jev.finish_formal(store, reviews, "jev-latest", 8, tmp_path)
        assert (tmp_path / "jev_predictions.csv").is_file()
        assert not (tmp_path / "jev_summary.json").exists()
    finally:
        store.close()

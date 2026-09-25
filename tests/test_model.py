from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from imdb_sentiment.config import ConfigError, ModelConfig, load_model_config
from imdb_sentiment import model


def test_model_config() -> None:
    config = load_model_config()
    assert config.name == "answerdotai/ModernBERT-base"
    assert config.num_labels == 2


@pytest.mark.parametrize("content", [
    "data: {}\n",
    "model: {name: '', num_labels: 2}\n",
    "model: {name: answerdotai/ModernBERT-base, num_labels: true}\n",
])
def test_invalid_model_config(tmp_path: Path, content: str) -> None:
    (tmp_path / "config.yaml").write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match="model"):
        load_model_config(tmp_path)


def test_loaders_use_config_and_label_mapping_without_download(monkeypatch) -> None:
    calls = []

    def fake_tokenizer(model_id):
        calls.append(("tokenizer", model_id))
        return "tokenizer"

    def fake_model(model_id, **kwargs):
        calls.append(("model", model_id, kwargs))
        return SimpleNamespace(config=SimpleNamespace(**kwargs))

    monkeypatch.setattr(model.AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(model.AutoModelForSequenceClassification, "from_pretrained", fake_model)
    config = load_model_config()
    assert model.load_tokenizer(config) == "tokenizer"
    loaded = model.load_model(config)
    assert calls[0] == ("tokenizer", config.name)
    assert calls[1] == ("model", config.name, {
        "num_labels": config.num_labels,
        "id2label": {0: "NEGATIVE", 1: "POSITIVE"},
        "label2id": {"NEGATIVE": 0, "POSITIVE": 1},
        "problem_type": "single_label_classification",
    })
    assert loaded.config.num_labels == 2
    assert loaded.config.id2label == {0: "NEGATIVE", 1: "POSITIVE"}
    assert loaded.config.label2id == {"NEGATIVE": 0, "POSITIVE": 1}


def test_num_labels_must_match_label_mapping(monkeypatch) -> None:
    monkeypatch.setattr(model.AutoModelForSequenceClassification, "from_pretrained",
                        lambda *args, **kwargs: pytest.fail("should not load"))
    with pytest.raises(ConfigError, match="label mapping"):
        model.load_model(ModelConfig("example/model", 3))


def test_encode_reviews_preserves_text_and_default_length_behavior() -> None:
    calls = []

    def tokenizer(texts, **kwargs):
        calls.append((texts, kwargs))
        return {"input_ids": [[10]], "attention_mask": [[1]]}

    text = "<br /> GREAT movie!"
    assert model.encode_reviews(text, tokenizer) == {
        "input_ids": [[10]], "attention_mask": [[1]],
    }
    assert calls[0][0] == text
    assert calls[0][1] == {
        "add_special_tokens": True, "return_attention_mask": True,
        "truncation": False, "max_length": None, "padding": False,
        "return_tensors": None,
    }
    model.encode_reviews([text, "Bad"], tokenizer, max_length=1024,
                         padding=True, return_tensors="pt")
    assert calls[1][1]["truncation"] is True
    assert calls[1][1]["max_length"] == 1024
    assert calls[1][1]["padding"] is True
    assert calls[1][1]["return_tensors"] == "pt"


def test_logits_prediction_probabilities_and_confidence() -> None:
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]])
    assert model.predict_ids(logits).tolist() == [0, 1]
    probabilities = model.class_probabilities(logits)
    assert probabilities.shape == logits.shape
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2))
    assert probabilities[0, 0] > probabilities[0, 1]
    assert probabilities[1, 1] > probabilities[1, 0]
    confidence = model.prediction_confidence(logits)
    assert torch.allclose(confidence, probabilities.max(dim=-1).values)
    assert torch.allclose(confidence, probabilities.gather(
        1, model.predict_ids(logits).unsqueeze(1)).squeeze(1))

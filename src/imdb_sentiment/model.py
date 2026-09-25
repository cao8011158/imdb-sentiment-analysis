"""ModernBERT sequence classification and small inference helpers."""

from collections.abc import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import ConfigError, ModelConfig, load_model_config


ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
LABEL2ID = {label: index for index, label in ID2LABEL.items()}


def load_tokenizer(config: ModelConfig | None = None):
    """Load the selected model's tokenizer without changing review text."""
    config = config or load_model_config()
    return AutoTokenizer.from_pretrained(config.name)


def load_model(config: ModelConfig | None = None):
    """Load Hugging Face's classification model; forward returns raw logits."""
    config = config or load_model_config()
    if config.num_labels != len(ID2LABEL):
        raise ConfigError("model.num_labels must match the IMDb label mapping")
    return AutoModelForSequenceClassification.from_pretrained(
        config.name,
        num_labels=config.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="single_label_classification",
    )


def encode_reviews(
    texts: str | Sequence[str], tokenizer, *, max_length: int | None = None,
    padding: bool = False, return_tensors: str | None = None,
):
    """Encode raw reviews; by default, neither truncate nor pad them.

    Set max_length to opt into truncation. For a tensor batch with unequal
    lengths, explicitly request padding=True and return_tensors="pt".
    """
    if max_length is not None and (isinstance(max_length, bool) or
                                   not isinstance(max_length, int) or max_length <= 0):
        raise ValueError("max_length must be a positive integer or None")
    return tokenizer(
        texts,
        add_special_tokens=True,
        return_attention_mask=True,
        truncation=max_length is not None,
        max_length=max_length,
        padding=padding,
        return_tensors=return_tensors,
    )


def predict_ids(logits: torch.Tensor) -> torch.Tensor:
    """Return class IDs using argmax over [negative, positive] logits."""
    return torch.argmax(logits, dim=-1)


def class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert raw logits to [P(negative), P(positive)] for reporting."""
    return torch.softmax(logits, dim=-1)


def prediction_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Return the predicted class's softmax probability."""
    return class_probabilities(logits).max(dim=-1).values

"""Optional CPU-only pretrained-model smoke test; run directly, not via pytest."""

import torch

from imdb_sentiment.config import load_model_config
from imdb_sentiment.model import (
    class_probabilities, encode_reviews, load_model, load_tokenizer,
    predict_ids, prediction_confidence,
)


def main() -> None:
    config = load_model_config()
    tokenizer = load_tokenizer(config)
    classifier = load_model(config).cpu().eval()
    encoded = encode_reviews(
        ["I loved this film!", "This movie was disappointing."],
        tokenizer, padding=True, return_tensors="pt",
    )
    with torch.inference_mode():
        logits = classifier(**encoded).logits
    assert logits.shape == (len(encoded["input_ids"]), config.num_labels)
    predictions = predict_ids(logits)
    probabilities = class_probabilities(logits)
    confidence = prediction_confidence(logits)
    print(f"model_class={type(classifier).__name__}")
    print(f"tokenizer_class={type(tokenizer).__name__}")
    print(f"classifier_out_features={classifier.classifier.out_features}")
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"predicted_ids={predictions.tolist()}")
    print(f"probabilities={probabilities.tolist()}")
    print(f"confidence={confidence.tolist()}")


if __name__ == "__main__":
    main()

# IMDb sentiment analysis

## Model interface

`config.yaml` selects `answerdotai/ModernBERT-base` with two sentiment labels:
`0 = NEGATIVE`, `1 = POSITIVE`. `load_tokenizer()` and `load_model()` in
`src/imdb_sentiment/model.py` use the Hugging Face Auto classes. The classifier
returns two raw logits in negative/positive order. `predict_ids()` uses argmax;
`class_probabilities()` applies softmax only for reporting; and
`prediction_confidence()` returns the predicted class's probability.

`encode_reviews(texts, tokenizer)` sends review text unchanged to the tokenizer
and returns `input_ids` and `attention_mask`. By default it does not truncate
or pad. Pass `max_length` to opt into truncation, and set `padding=True` and
`return_tensors="pt"` when a batch needs equal-length PyTorch tensors. The
training pipeline will choose its own length and padding strategy later.

Run the optional pretrained-weight CPU smoke test with
`$env:PYTHONPATH='src'; python tests/smoke_model.py` in PowerShell. It downloads
the model weights if they are not cached; the normal unit tests do not.

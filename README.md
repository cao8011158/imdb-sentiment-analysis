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
training pipeline uses the configured 2048-token truncation limit and pads only
to the longest item in each batch (rounded up to a multiple of 8).

The pretrained-weight CPU smoke test is `python tests/smoke_model.py`. Run it
in Colab after installation; it downloads model weights if they are not cached.
The normal unit tests do not download weights.

## Colab training sequence

Run the following in a Google Colab notebook with an A100 runtime. This is a
Colab procedure; the training commands are not intended for the local Windows
workspace. The Python package is installed from `pyproject.toml`, so this
procedure does not depend on the existing `uv.lock` file.

1. Clone your repository and install its training and test dependencies:

   ```python
   !git clone YOUR_REPOSITORY_URL imdb-sentiment-analysis
   %cd imdb-sentiment-analysis
   !pip install -e '.[test]'
   ```

2. Place the official IMDb archive in Colab and extract it so that
   `data/aclImdb/train/pos`, `data/aclImdb/train/neg`, `data/aclImdb/test/pos`,
   and `data/aclImdb/test/neg` exist. For an archive at `/content/aclImdb_v1.tar.gz`:

   ```python
   !mkdir -p data
   !tar -xzf /content/aclImdb_v1.tar.gz -C data
   ```

3. Run the existing unit tests, then the pretrained forward smoke test:

   ```python
   !pytest -q
   !python tests/smoke_model.py
   ```

4. Run the single-batch forward/backward/optimizer smoke test. Each command
   uses a balanced batch of the requested size, requires CUDA, and checks BF16
   when enabled in `config.yaml`. Try 8, 16, then 32; CUDA OOM is reported
   directly and never changes the batch size automatically.

   ```python
   !python tests/smoke_training.py single-batch --batch-size 8
   !python tests/smoke_training.py single-batch --batch-size 16
   !python tests/smoke_training.py single-batch --batch-size 32
   ```

5. Check that the same 16 training reviews can be memorized, then check the
   full training flow on 200 training and 100 validation reviews for 2 epochs.
   The tiny overfit check uses training reviews only and a test-only learning
   rate; neither check evaluates the official test split.

   ```python
   !python tests/smoke_training.py tiny-overfit
   !python tests/smoke_training.py e2e --batch-size 16 --accumulation-steps 2
   ```

6. Choose the largest stable A100 micro batch. Keep the effective batch size
   at 32 by setting one explicit pair for **every** later run: `8 × 4`,
   `16 × 2`, or `32 × 1`. Change the two `training` values in `config.yaml`
   together, or pass both CLI flags as shown below.

7. Run the learning-rate pilot using one training subset and the fixed 2,500
   review validation split. The three rates come from `config.yaml`; the
   official test set is not evaluated. Change the batch/accumulation pair in
   the command if needed.

   ```python
   !python -m imdb_sentiment.training lr-pilot --train-size 1000 --batch-size 16 --accumulation-steps 2
   !cat results/lr_search.csv
   ```

8. Inspect `results/lr_search.csv` and the pilot histories under
   `results/training/lr_pilot/`. Select by best validation F1, then set the
   single `training.learning_rate` value in `config.yaml` to that rate.
   Leave it fixed for all four scaling runs. Training starts from the same
   pretrained checkpoint and seed on every run.

9. Run the four formal experiments, keeping the same micro batch and
   accumulation flags in every command:

   ```python
   !python -m imdb_sentiment.training formal --train-size 1000 --batch-size 16 --accumulation-steps 2
   !python -m imdb_sentiment.training formal --train-size 5000 --batch-size 16 --accumulation-steps 2
   !python -m imdb_sentiment.training formal --train-size 10000 --batch-size 16 --accumulation-steps 2
   !python -m imdb_sentiment.training formal --train-size 22500 --batch-size 16 --accumulation-steps 2
   ```

The training tqdm bar shows epoch, global and within-epoch steps, loss, and
learning rate. Each validation epoch prints loss, accuracy, positive-class
precision, recall, and F1. TensorBoard stores training loss and learning rate
against optimizer step, plus validation loss, accuracy, and F1 against epoch:

```python
%load_ext tensorboard
%tensorboard --logdir results/runs
```

Each formal run writes `training_history.csv`, `metrics.json`,
`best_checkpoint.json`, and `final_test_metrics.json` under
`results/training/train_<size>/`. It validates and saves each epoch, stops
after the configured number of non-improving validation F1 checks, restores
the highest-F1 checkpoint, and only then evaluates the untouched official
25,000-review test split. Existing run directories are protected against
accidental overwrites. Later formal runs also reject changes to the selected
learning rate, training settings, micro batch, or fixed validation IDs.

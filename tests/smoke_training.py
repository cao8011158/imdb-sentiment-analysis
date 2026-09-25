"""Run only on Colab CUDA: training checks and worst-case memory stress."""

import argparse
import csv
from pathlib import Path

import torch
from transformers import DataCollatorWithPadding, set_seed

from imdb_sentiment.config import load_data_config, load_model_config, load_training_config
from imdb_sentiment.dataset import Review, prepare_dataset
from imdb_sentiment.model import load_model, load_tokenizer
from imdb_sentiment.training import EncodedReviews, _batch_settings, run_experiment


def _cuda_device(bf16: bool) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Colab training verification")
    if bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured BF16 is not supported by this CUDA device")
    return torch.device("cuda")


def _balanced_short(reviews: tuple[Review, ...], per_class: int) -> tuple[Review, ...]:
    """Use only the existing training split; choose short reviews for quick checks."""
    selected = []
    for label in (0, 1):
        candidates = sorted(
            (review for review in reviews if review.label == label),
            key=lambda review: len(review.text),
        )
        if len(candidates) < per_class:
            raise ValueError(f"Not enough label-{label} training reviews")
        selected.extend(candidates[:per_class])
    return tuple(selected)


def _batch(reviews, tokenizer, max_length):
    encoded = EncodedReviews(reviews, tokenizer, max_length)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    return collator([encoded[index] for index in range(len(encoded))])


def single_batch(reviews: tuple[Review, ...], batch_size: int) -> None:
    config = load_training_config()
    device = _cuda_device(config.bf16)
    set_seed(config.seed)
    tokenizer = load_tokenizer(load_model_config())
    model = load_model(load_model_config()).to(device).train()
    batch = _batch(_balanced_short(reviews, batch_size // 2),
                   tokenizer, config.max_length)
    batch = {key: value.to(device) for key, value in batch.items()}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    before = model.classifier.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.bf16):
        output = model(**batch)
    if output.logits.shape != (batch_size, load_model_config().num_labels):
        raise AssertionError(f"Unexpected logits shape: {tuple(output.logits.shape)}")
    if not torch.isfinite(output.loss):
        raise AssertionError("Single-batch loss is NaN or Inf")
    output.loss.backward()
    encoder_gradients = [parameter.grad for name, parameter in model.named_parameters()
                         if not name.startswith("classifier") and parameter.grad is not None]
    if not encoder_gradients or not any(torch.any(grad != 0) for grad in encoder_gradients):
        raise AssertionError("ModernBERT encoder did not receive gradients")
    if model.classifier.weight.grad is None:
        raise AssertionError("Classification head did not receive gradients")
    optimizer.step()
    if torch.equal(before, model.classifier.weight.detach()):
        raise AssertionError("Optimizer step did not change trainable head parameters")
    print(f"single-batch PASS: batch={batch_size}, bf16={config.bf16}, "
          f"loss={output.loss.item():.4f}, logits={tuple(output.logits.shape)}")


def memory_batch(batch_size: int) -> None:
    """One full optimizer step with every attention-mask position occupied."""
    config = load_training_config()
    if config.max_length != 2048 or not config.bf16:
        raise RuntimeError("memory-batch requires max_length=2048 and bf16=true")
    device = _cuda_device(config.bf16)
    set_seed(config.seed)
    model_config = load_model_config()
    tokenizer = load_tokenizer(model_config)
    model = load_model(model_config).to(device).train()
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("memory-batch requires full fine-tuning")

    ordinary_ids = tokenizer.encode("movie", add_special_tokens=False)
    ordinary_ids = [token_id for token_id in ordinary_ids
                    if token_id not in tokenizer.all_special_ids]
    if not ordinary_ids or tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise RuntimeError("Tokenizer did not provide required legal token IDs")
    token_id = ordinary_ids[0]
    sequence = [tokenizer.cls_token_id] + [token_id] * (config.max_length - 2)
    sequence.append(tokenizer.sep_token_id)
    vocabulary_size = model.get_input_embeddings().num_embeddings
    if any(token_id < 0 or token_id >= vocabulary_size for token_id in sequence):
        raise RuntimeError("Synthetic input contains a token ID outside model vocabulary")

    batch = {
        "input_ids": torch.tensor(sequence, dtype=torch.long).repeat(batch_size, 1),
        "attention_mask": torch.ones((batch_size, config.max_length), dtype=torch.long),
        "labels": torch.arange(batch_size, dtype=torch.long) % 2,
    }
    expected_shape = (batch_size, 2048)
    if (batch["input_ids"].shape != expected_shape or
            batch["attention_mask"].shape != expected_shape or
            batch["labels"].shape != (batch_size,) or
            not torch.all(batch["attention_mask"] == 1)):
        raise AssertionError("Synthetic batch does not have the required full-length shape")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            output = model(**batch)
        if output.logits.shape != (batch_size, model_config.num_labels):
            raise AssertionError(f"Unexpected logits shape: {tuple(output.logits.shape)}")
        if not torch.isfinite(output.loss):
            raise AssertionError("memory-batch loss is NaN or Inf")
        output.loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError:
        print(f"memory-batch OOM: batch_size={batch_size}, sequence_length=2048", flush=True)
        raise

    bytes_per_gb = 1_000_000_000
    print("memory-batch PASS")
    print(f"batch_size={batch_size}")
    print("sequence_length=2048")
    print(f"input_ids_dtype={batch['input_ids'].dtype}, "
          f"model_dtype={next(model.parameters()).dtype}, bf16_autocast=True")
    print(f"loss={output.loss.item():.4f}")
    print(f"peak_allocated_memory_GB={torch.cuda.max_memory_allocated(device) / bytes_per_gb:.3f}")
    print(f"peak_reserved_memory_GB={torch.cuda.max_memory_reserved(device) / bytes_per_gb:.3f}")
    print(f"gpu_name={torch.cuda.get_device_name(device)}")


def tiny_overfit(reviews: tuple[Review, ...], steps: int, learning_rate: float) -> None:
    config = load_training_config()
    device = _cuda_device(config.bf16)
    set_seed(config.seed)
    tokenizer = load_tokenizer(load_model_config())
    model = load_model(load_model_config()).to(device).train()
    batch = _batch(_balanced_short(reviews, 8), tokenizer, config.max_length)
    batch = {key: value.to(device) for key, value in batch.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    first_loss = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.bf16):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise AssertionError(f"Tiny-overfit loss is NaN or Inf at step {step}")
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        optimizer.step()
        if (step + 1) % 10 == 0:
            print(f"tiny-overfit step {step + 1}/{steps}: loss={loss.item():.4f}")
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                enabled=config.bf16):
        output = model(**batch)
    final_loss = output.loss.item()
    accuracy = (output.logits.argmax(dim=-1) == batch["labels"]).float().mean().item()
    if final_loss >= first_loss * 0.5 or accuracy < 0.90:
        raise AssertionError(
            f"Tiny overfit did not converge: loss {first_loss:.4f} -> "
            f"{final_loss:.4f}, training accuracy={accuracy:.3f}"
        )
    print(f"tiny-overfit PASS: loss {first_loss:.4f} -> {final_loss:.4f}, "
          f"training accuracy={accuracy:.3f}")


def end_to_end(train_reviews: tuple[Review, ...],
               validation_reviews: tuple[Review, ...],
               batch_size: int | None, accumulation_steps: int | None) -> None:
    config = load_training_config()
    _cuda_device(config.bf16)
    micro, accumulation = _batch_settings(config, batch_size, accumulation_steps)
    result = run_experiment(
        _balanced_short(train_reviews, 100),
        _balanced_short(validation_reviews, 50),
        run_name="smoke_e2e", config=config, learning_rate=config.learning_rate,
        batch_size=micro, accumulation_steps=accumulation, max_epochs=2,
    )
    output = Path(__file__).resolve().parents[1] / "results/training/smoke_e2e"
    for name in ("training_history.csv", "metrics.json", "best_checkpoint.json"):
        if not (output / name).is_file():
            raise AssertionError(f"Missing end-to-end output: {name}")
    with (output / "training_history.csv").open(encoding="utf-8", newline="") as file:
        if not any(row["eval_f1"] for row in csv.DictReader(file)):
            raise AssertionError("Validation metrics were not saved to history")
    if not Path(result["checkpoint"]).is_dir() or not result["best_checkpoint_restored"]:
        raise AssertionError("Best checkpoint was not saved and restored")
    print(f"end-to-end PASS: best epoch={result['best_epoch']}, "
          f"validation F1={result['best_validation_f1']:.4f}, "
          f"optimizer steps={result['total_optimizer_steps']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("single-batch")
    one.add_argument("--batch-size", type=int, choices=(8, 16, 32), required=True)
    memory = commands.add_parser("memory-batch")
    memory.add_argument("--batch-size", type=int, choices=(8, 16, 32), required=True)
    overfit = commands.add_parser("tiny-overfit")
    overfit.add_argument("--steps", type=int, default=100)
    overfit.add_argument("--learning-rate", type=float, default=1e-4)
    e2e = commands.add_parser("e2e")
    e2e.add_argument("--batch-size", type=int)
    e2e.add_argument("--accumulation-steps", type=int)
    options = parser.parse_args()
    if options.command == "memory-batch":
        memory_batch(options.batch_size)
        return
    if options.command == "tiny-overfit" and (options.steps <= 0 or options.learning_rate <= 0):
        parser.error("--steps and --learning-rate must be positive")
    prepared = prepare_dataset(load_data_config())
    train_reviews = prepared.train_subsets[min(prepared.train_subsets)]
    if options.command == "single-batch":
        single_batch(train_reviews, options.batch_size)
    elif options.command == "tiny-overfit":
        tiny_overfit(train_reviews, options.steps, options.learning_rate)
    else:
        end_to_end(train_reviews, prepared.validation,
                   options.batch_size, options.accumulation_steps)


if __name__ == "__main__":
    main()

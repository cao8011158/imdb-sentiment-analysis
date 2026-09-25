"""Reproducible ModernBERT full fine-tuning on the existing IMDb splits."""

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
import time

import torch
from safetensors import safe_open
from torch.utils.data import Dataset, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (
    DataCollatorWithPadding, EarlyStoppingCallback, ProgressCallback,
    Trainer, TrainerCallback, TrainingArguments, set_seed,
)

from .config import TrainingConfig, load_data_config, load_model_config, load_training_config
from .dataset import Review, prepare_dataset
from .model import encode_reviews, load_model, load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_COLUMNS = (
    "epoch", "global_step", "train_loss", "learning_rate", "eval_loss",
    "eval_accuracy", "eval_precision", "eval_recall", "eval_f1",
)


class EncodedReviews(Dataset):
    """Thin tokenized view of existing Review objects; no new split logic."""

    def __init__(self, reviews: tuple[Review, ...], tokenizer, max_length: int):
        self.examples = []
        for start in range(0, len(reviews), 128):
            batch = reviews[start:start + 128]
            encoded = encode_reviews(
                [review.text for review in batch], tokenizer,
                max_length=max_length, padding=False,
            )
            for index, review in enumerate(batch):
                self.examples.append({
                    "input_ids": encoded["input_ids"][index],
                    "attention_mask": encoded["attention_mask"][index],
                    "labels": review.label,
                })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


class SequentialEvalTrainer(Trainer):
    """Keep length grouping for shuffled training, but preserve eval/test order."""

    def _get_eval_sampler(self, eval_dataset):
        return SequentialSampler(eval_dataset)


def binary_metrics(labels, predictions) -> dict[str, float]:
    """Binary metrics with POSITIVE=1 and zero-division results of 0."""
    pairs = list(zip(labels, predictions, strict=True))
    if not pairs:
        raise ValueError("Cannot evaluate an empty set of predictions")
    tp = sum(true == 1 and pred == 1 for true, pred in pairs)
    tn = sum(true == 0 and pred == 0 for true, pred in pairs)
    fp = sum(true == 0 and pred == 1 for true, pred in pairs)
    fn = sum(true == 1 and pred == 0 for true, pred in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(pairs),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_metrics(eval_prediction) -> dict[str, float]:
    """Metrics use argmax of raw logits; softmax is not in the loss path."""
    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted = logits.argmax(axis=-1)
    return binary_metrics(eval_prediction.label_ids.tolist(), predicted.tolist())


class HistoryCallback(TrainerCallback):
    """Persist each Trainer loss/learning-rate and validation event immediately."""

    def __init__(self, path: Path):
        self.path = path
        self.epoch_writer = None
        with path.open("w", encoding="utf-8", newline="") as file:
            csv.DictWriter(file, fieldnames=HISTORY_COLUMNS).writeheader()

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.epoch_writer = SummaryWriter(log_dir=args.logging_dir)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        if not ("loss" in logs or "eval_loss" in logs):
            return
        row = {
            "epoch": logs.get("epoch", state.epoch),
            "global_step": state.global_step,
            "train_loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "eval_loss": logs.get("eval_loss"),
            "eval_accuracy": logs.get("eval_accuracy"),
            "eval_precision": logs.get("eval_precision"),
            "eval_recall": logs.get("eval_recall"),
            "eval_f1": logs.get("eval_f1"),
        }
        with self.path.open("a", encoding="utf-8", newline="") as file:
            csv.DictWriter(file, fieldnames=HISTORY_COLUMNS).writerow(row)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.epoch_writer is None or not metrics or "eval_f1" not in metrics:
            return
        epoch = round(state.epoch or 0)
        for key in ("loss", "accuracy", "f1"):
            self.epoch_writer.add_scalar(
                f"validation_epoch/{key}", metrics[f"eval_{key}"], epoch,
            )
        self.epoch_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.epoch_writer is not None:
            self.epoch_writer.close()
            self.epoch_writer = None


class EpochProgressCallback(ProgressCallback):
    """One tqdm bar with epoch, local step, latest loss, and learning rate."""

    def __init__(self, steps_per_epoch: int, max_epochs: int):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs
        self.loss = None
        self.learning_rate = None

    def _refresh(self, state):
        if self.training_bar is None:
            return
        epoch = min(self.max_epochs, max(1, ceil((state.epoch or 0) - 1e-9)))
        local_step = state.global_step - (epoch - 1) * self.steps_per_epoch
        self.training_bar.set_description(f"Epoch {epoch}/{self.max_epochs}")
        details = f"Step {local_step}/{self.steps_per_epoch} | Global {state.global_step}"
        if self.loss is not None:
            details += f" | Loss {self.loss:.4f}"
        if self.learning_rate is not None:
            details += f" | LR {self.learning_rate:.2e}"
        self.training_bar.set_postfix_str(details)

    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        if state.is_world_process_zero:
            self._refresh(state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        if "loss" in logs:
            self.loss = logs["loss"]
            self.learning_rate = logs.get("learning_rate", self.learning_rate)
            self._refresh(state)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        if not state.is_world_process_zero or not metrics or "eval_f1" not in metrics:
            return
        epoch = round(state.epoch or 0)
        tqdm.write(
            f"Epoch {epoch} validation\n"
            "------------------\n"
            f"eval_loss: {metrics['eval_loss']:.4f}\n"
            f"accuracy: {metrics['eval_accuracy']:.4f}\n"
            f"precision: {metrics['eval_precision']:.4f}\n"
            f"recall: {metrics['eval_recall']:.4f}\n"
            f"f1: {metrics['eval_f1']:.4f}"
        )


def _write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def _best_validation_record(trainer: Trainer) -> dict:
    checkpoint = trainer.state.best_model_checkpoint
    if not checkpoint:
        raise RuntimeError("Trainer did not save a best validation checkpoint")
    step = int(Path(checkpoint).name.removeprefix("checkpoint-"))
    matches = [row for row in trainer.state.log_history
               if row.get("step") == step and "eval_f1" in row]
    if len(matches) != 1:
        raise RuntimeError(f"Could not identify validation metrics for {checkpoint}")
    record = matches[0]
    return {
        "checkpoint": str(checkpoint),
        "best_epoch": round(record["epoch"]),
        "best_global_step": step,
        "best_validation_loss": record["eval_loss"],
        "best_validation_f1": record["eval_f1"],
    }


def run_experiment(
    train_reviews: tuple[Review, ...], validation_reviews: tuple[Review, ...],
    *, run_name: str, config: TrainingConfig, learning_rate: float,
    test_reviews: tuple[Review, ...] | None = None,
    batch_size: int | None = None, accumulation_steps: int | None = None,
    max_epochs: int | None = None,
) -> dict:
    """Train on one subset, select by validation F1, optionally report test once."""
    output_dir = PROJECT_ROOT / "results" / "training" / run_name
    logging_dir = PROJECT_ROOT / "results" / "runs" / run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Run already exists; choose a fresh run name: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    micro = config.per_device_train_batch_size if batch_size is None else batch_size
    accumulation = (config.gradient_accumulation_steps if accumulation_steps is None
                    else accumulation_steps)
    epochs = config.max_epochs if max_epochs is None else max_epochs
    if micro <= 0 or accumulation <= 0 or epochs <= 0:
        raise ValueError("Batch size, accumulation, and epochs must be positive")
    model_config = load_model_config()
    set_seed(config.seed)
    tokenizer = load_tokenizer(model_config)
    train_data = EncodedReviews(train_reviews, tokenizer, config.max_length)
    validation_data = EncodedReviews(validation_reviews, tokenizer, config.max_length)
    model = load_model(model_config)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Full fine-tuning requires every model parameter to be trainable")

    args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        logging_dir=str(logging_dir),
        report_to=["tensorboard"],
        per_device_train_batch_size=micro,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=accumulation,
        learning_rate=learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        num_train_epochs=epochs,
        optim="adamw_torch",
        bf16=config.bf16,
        group_by_length=config.group_by_length,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=config.save_total_limit,
        save_safetensors=True,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=True,
        load_best_model_at_end=True,
        logging_strategy="steps",
        logging_steps=1,
        disable_tqdm=False,
        auto_find_batch_size=False,
        label_smoothing_factor=0.0,
        seed=config.seed,
        data_seed=config.seed,
    )
    callbacks = [HistoryCallback(output_dir / "training_history.csv")]
    if config.early_stopping_patience:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold,
        ))
    trainer = SequentialEvalTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    trainer.remove_callback(ProgressCallback)
    steps_per_epoch = ceil(len(train_data) / (micro * accumulation))
    trainer.add_callback(EpochProgressCallback(steps_per_epoch, epochs))

    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - started
    best = _best_validation_record(trainer)
    with safe_open(str(Path(best["checkpoint"]) / "model.safetensors"),
                   framework="pt", device="cpu") as checkpoint_file:
        saved_head = checkpoint_file.get_tensor("classifier.weight")
    if not torch.equal(saved_head, trainer.model.classifier.weight.detach().cpu()):
        raise RuntimeError("Trainer did not restore the best checkpoint at train end")
    _write_json(output_dir / "best_checkpoint.json", best)
    summary = {
        "run_name": run_name,
        "train_size": len(train_data),
        "validation_size": len(validation_data),
        "validation_ids_sha256": sha256(
            "\n".join(sorted(review.sample_id for review in validation_reviews)).encode("utf-8")
        ).hexdigest(),
        "model": model_config.name,
        "training_config": asdict(config),
        "learning_rate": learning_rate,
        "micro_batch_size": micro,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": micro * accumulation,
        "training_time_seconds": elapsed,
        "total_optimizer_steps": trainer.state.global_step,
        "best_checkpoint_restored": True,
        **best,
    }
    _write_json(output_dir / "metrics.json", summary)

    if test_reviews is not None:
        test_data = EncodedReviews(test_reviews, tokenizer, config.max_length)
        test_metrics = trainer.predict(test_data, metric_key_prefix="test").metrics
        final = {
            "accuracy": test_metrics["test_accuracy"],
            "precision": test_metrics["test_precision"],
            "recall": test_metrics["test_recall"],
            "f1": test_metrics["test_f1"],
            "test_loss": test_metrics["test_loss"],
            "training_time_seconds": elapsed,
            "best_epoch": best["best_epoch"],
            "total_optimizer_steps": trainer.state.global_step,
        }
        _write_json(output_dir / "final_test_metrics.json", final)
        summary["final_test_metrics"] = final
    return summary


def _batch_settings(config: TrainingConfig, batch_size: int | None,
                    accumulation_steps: int | None) -> tuple[int, int]:
    if (batch_size is None) != (accumulation_steps is None):
        raise ValueError("Specify both --batch-size and --accumulation-steps, or neither")
    micro = config.per_device_train_batch_size if batch_size is None else batch_size
    accumulation = (config.gradient_accumulation_steps if accumulation_steps is None
                    else accumulation_steps)
    if micro <= 0 or accumulation <= 0:
        raise ValueError("Batch size and accumulation must be positive")
    if micro * accumulation != config.effective_batch_size:
        raise ValueError(
            f"Effective batch size must stay {config.effective_batch_size}; "
            f"got {micro} × {accumulation}"
        )
    return micro, accumulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("lr-pilot", "formal"):
        command = commands.add_parser(name)
        command.add_argument("--batch-size", type=int)
        command.add_argument("--accumulation-steps", type=int)
        if name == "formal":
            command.add_argument("--train-size", type=int, required=True)
        else:
            command.add_argument("--train-size", type=int, default=1000)
    options = parser.parse_args()
    config = load_training_config()
    micro, accumulation = _batch_settings(
        config, options.batch_size, options.accumulation_steps,
    )
    data_config = load_data_config()
    if options.train_size not in data_config.train_sizes:
        parser.error(f"--train-size must be one of {data_config.train_sizes}")
    prepared = prepare_dataset(data_config)
    train_reviews = prepared.train_subsets[options.train_size]

    if options.command == "formal":
        current_validation_hash = sha256(
            "\n".join(sorted(review.sample_id for review in prepared.validation)).encode("utf-8")
        ).hexdigest()
        for size in data_config.train_sizes:
            previous = PROJECT_ROOT / "results" / "training" / f"train_{size}" / "metrics.json"
            if previous.is_file():
                recorded = json.loads(previous.read_text(encoding="utf-8"))
                if (recorded["model"] != load_model_config().name or
                        json.dumps(recorded["training_config"], sort_keys=True) !=
                        json.dumps(asdict(config), sort_keys=True) or
                        recorded["micro_batch_size"] != micro or
                        recorded["gradient_accumulation_steps"] != accumulation or
                        recorded["validation_ids_sha256"] != current_validation_hash):
                    raise ValueError(
                        f"Formal run settings differ from completed train_{size}; "
                        "keep one selected learning rate and the same settings"
                    )
        result = run_experiment(
            train_reviews, prepared.validation,
            test_reviews=prepared.official_test,
            run_name=f"train_{options.train_size}",
            config=config, learning_rate=config.learning_rate,
            batch_size=micro, accumulation_steps=accumulation,
        )
        print(json.dumps(result["final_test_metrics"], indent=2))
        return

    path = PROJECT_ROOT / "results" / "lr_search.csv"
    if path.exists():
        raise FileExistsError(f"LR pilot results already exist: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "learning_rate", "best_epoch", "best_validation_loss", "best_validation_f1",
        ))
        writer.writeheader()
        for rate in config.learning_rate_candidates:
            result = run_experiment(
                train_reviews, prepared.validation,
                run_name=f"lr_pilot/train_{options.train_size}_lr_{rate:g}",
                config=config, learning_rate=rate,
                batch_size=micro, accumulation_steps=accumulation,
            )
            writer.writerow({key: result[key] for key in writer.fieldnames})
            file.flush()
    print(f"Review validation F1 in {path}, then set training.learning_rate in config.yaml")


if __name__ == "__main__":
    main()

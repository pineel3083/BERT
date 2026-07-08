import argparse
import json
import os
import random
import time

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


GLUE_DATASET_REPO = "nyu-mll/glue"
TASKS = {
    "mnli": {
        "dataset_name": "mnli",
        "train_split": "train",
        "validation_split": "validation_matched",
        "text_columns": ("premise", "hypothesis"),
        "num_labels": 3,
        "primary_metric": "accuracy",
        "id2label": {0: "entailment", 1: "neutral", 2: "contradiction"},
    },
    "mrpc": {
        "dataset_name": "mrpc",
        "train_split": "train",
        "validation_split": "validation",
        "text_columns": ("sentence1", "sentence2"),
        "num_labels": 2,
        "primary_metric": "f1",
        "id2label": {0: "not_equivalent", 1: "equivalent"},
    },
}


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, count):
        self.total += float(value) * int(count)
        self.count += int(count)

    @property
    def average(self):
        return self.total / self.count if self.count else 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dense fine-tune BERT on GLUE and save a reusable HF checkpoint."
    )
    parser.add_argument("--task", default="mnli", choices=sorted(TASKS))
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every-epoch", action="store_true", default=True)
    parser.add_argument("--no-eval-every-epoch", dest="eval_every_epoch", action="store_false")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-safe-serialization", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_duration(seconds):
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def load_glue_split(task_cfg, split, max_examples=None):
    dataset = load_dataset(GLUE_DATASET_REPO, task_cfg["dataset_name"], split=split)
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    return dataset


def tokenize_dataset(dataset, tokenizer, task_cfg, max_length):
    text_a, text_b = task_cfg["text_columns"]

    def preprocess(batch):
        encoded = tokenizer(
            batch[text_a],
            batch[text_b],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        encoded["labels"] = batch["label"]
        return encoded

    tokenized = dataset.map(preprocess, batched=True)
    keep_columns = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized.column_names:
        keep_columns.append("token_type_ids")
    remove_columns = [column for column in tokenized.column_names if column not in keep_columns]
    if remove_columns:
        tokenized = tokenized.remove_columns(remove_columns)
    tokenized.set_format(type="torch", columns=keep_columns)
    return tokenized


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(model_name, task_cfg):
    id2label = {int(k): v for k, v in task_cfg["id2label"].items()}
    label2id = {v: k for k, v in id2label.items()}
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=task_cfg["num_labels"],
        id2label=id2label,
        label2id=label2id,
    )
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )


def build_optimizer(model, learning_rate, weight_decay):
    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if param.requires_grad and not any(nd in name for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if param.requires_grad and any(nd in name for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(grouped, lr=learning_rate)


def compute_metrics(task_name, predictions, labels):
    if not labels:
        return {"accuracy": 0.0, "f1": None}
    correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
    accuracy = correct / len(labels)
    metrics = {"accuracy": accuracy}

    if task_name == "mrpc":
        tp = sum(int(pred == 1 and label == 1) for pred, label in zip(predictions, labels))
        fp = sum(int(pred == 1 and label == 0) for pred, label in zip(predictions, labels))
        fn = sum(int(pred == 0 and label == 1) for pred, label in zip(predictions, labels))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        metrics["f1"] = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    else:
        metrics["f1"] = None
    return metrics


def evaluate(model, loader, task_name, device):
    model.eval()
    loss_meter = AverageMeter()
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            batch_size = batch["labels"].size(0)
            loss_meter.update(outputs.loss.item(), batch_size)
            batch_predictions = outputs.logits.argmax(dim=-1)
            predictions.extend(batch_predictions.detach().cpu().tolist())
            labels.extend(batch["labels"].detach().cpu().tolist())

    metrics = compute_metrics(task_name, predictions, labels)
    metrics["loss"] = loss_meter.average
    return metrics


def save_checkpoint(model, tokenizer, output_dir, args, metrics, epoch, global_step, tag):
    os.makedirs(output_dir, exist_ok=True)
    save_kwargs = {"safe_serialization": not args.no_safe_serialization}
    try:
        model.save_pretrained(output_dir, **save_kwargs)
    except TypeError:
        model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    summary = {
        "tag": tag,
        "task": args.task,
        "source_model": args.model_name,
        "epoch": epoch,
        "global_step": global_step,
        "max_length": args.max_length,
        "metrics": metrics,
        "label_order": "dataset",
        "note": "Checkpoint logits follow GLUE dataset label ids; use --checkpoint-label-order dataset in eval scripts.",
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def main():
    args = parse_args()
    task_cfg = TASKS[args.task]
    if args.output_dir is None:
        args.output_dir = os.path.join("checkpoints", f"dense_{args.task}_len{args.max_length}")

    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be >= 1")

    set_seed(args.seed)
    device = torch.device(args.device)

    print("track: dense_glue_finetune")
    print("task:", args.task)
    print("model_name:", args.model_name)
    print("dataset_repo:", GLUE_DATASET_REPO)
    print("train_split:", task_cfg["train_split"])
    print("validation_split:", task_cfg["validation_split"])
    print("output_dir:", args.output_dir)
    print("device:", device)
    print("max_length:", args.max_length)
    print("batch_size:", args.batch_size)
    print("eval_batch_size:", args.eval_batch_size)
    print("epochs:", args.epochs)
    print("learning_rate:", args.learning_rate)
    print("fp16:", args.fp16)
    print("label_order: dataset")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    raw_train = load_glue_split(task_cfg, task_cfg["train_split"], args.max_train_examples)
    raw_eval = load_glue_split(task_cfg, task_cfg["validation_split"], args.max_eval_examples)
    train_dataset = tokenize_dataset(raw_train, tokenizer, task_cfg, args.max_length)
    eval_dataset = tokenize_dataset(raw_eval, tokenizer, task_cfg, args.max_length)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    eval_loader = make_loader(eval_dataset, args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model_name, task_cfg)
    model.to(device)

    updates_per_epoch = max(1, (len(train_loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps)
    total_update_steps = max(1, int(args.epochs * updates_per_epoch))
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    print("train_examples:", len(train_dataset))
    print("eval_examples:", len(eval_dataset))
    print("updates_per_epoch:", updates_per_epoch)
    print("total_update_steps:", total_update_steps)
    print("warmup_steps:", warmup_steps)

    best_score = None
    best_metrics = None
    global_step = 0
    start_time = time.time()
    model.zero_grad(set_to_none=True)

    full_epochs = int(args.epochs)
    extra_fraction = args.epochs - full_epochs
    epoch_count = full_epochs + (1 if extra_fraction > 0 else 0)

    for epoch in range(1, epoch_count + 1):
        model.train()
        loss_meter = AverageMeter()
        epoch_limit = len(train_loader)
        if epoch == epoch_count and extra_fraction > 0:
            epoch_limit = max(1, int(len(train_loader) * extra_fraction))

        for step, batch in enumerate(train_loader, start=1):
            if step > epoch_limit:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
            batch_size = batch["labels"].size(0)
            loss_meter.update(outputs.loss.item(), batch_size)

            scaler.scale(loss).backward()
            should_step = step % args.gradient_accumulation_steps == 0 or step == epoch_limit
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if args.log_every and global_step % args.log_every == 0:
                    elapsed = time.time() - start_time
                    steps_left = max(0, total_update_steps - global_step)
                    steps_per_second = global_step / elapsed if elapsed > 0 else 0.0
                    eta = steps_left / steps_per_second if steps_per_second > 0 else 0.0
                    print(
                        f"[train] epoch={epoch} step={global_step}/{total_update_steps} "
                        f"loss={loss_meter.average:.6f} lr={scheduler.get_last_lr()[0]:.3e} "
                        f"elapsed={format_duration(elapsed)} eta~{format_duration(eta)}",
                        flush=True,
                    )

            if global_step >= total_update_steps:
                break

        if args.eval_every_epoch or epoch == epoch_count:
            metrics = evaluate(model, eval_loader, args.task, device)
            score = metrics[task_cfg["primary_metric"]]
            print(
                f"[eval] epoch={epoch} step={global_step} "
                f"loss={metrics['loss']:.6f} accuracy={metrics['accuracy']:.6f} "
                f"f1={metrics['f1'] if metrics['f1'] is not None else '-'}",
                flush=True,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_metrics = dict(metrics)
                save_checkpoint(
                    model,
                    tokenizer,
                    args.output_dir,
                    args,
                    metrics,
                    epoch,
                    global_step,
                    tag="best",
                )
                print(f"[save] best checkpoint -> {args.output_dir}", flush=True)

        if global_step >= total_update_steps:
            break

    last_dir = os.path.join(args.output_dir, "last")
    last_metrics = evaluate(model, eval_loader, args.task, device)
    save_checkpoint(
        model,
        tokenizer,
        last_dir,
        args,
        last_metrics,
        epoch,
        global_step,
        tag="last",
    )
    print(f"[save] last checkpoint -> {last_dir}")
    print(
        "best: "
        f"{task_cfg['primary_metric']}={best_score:.6f} "
        f"metrics={best_metrics}"
    )
    print("load this checkpoint with:", args.output_dir)


if __name__ == "__main__":
    main()

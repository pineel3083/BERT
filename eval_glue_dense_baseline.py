import argparse
import csv
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from eval_glue_tpdcim import (
    GLUE_DATASET_REPO,
    TASKS,
    apply_prediction_label_map,
    collate_batch,
    compute_input_length_stats,
    compute_metrics,
    fmt_float,
    load_model,
    resolve_task_cfg,
    tokenize_dataset,
)


TRACK_NAME = "dense_glue_baseline"


PRESET_HELP = """
Recommended runs:

A. TextAttack MNLI dense baseline, GLUE-style max_length=128:
   python eval_glue_dense_baseline.py --task mnli --checkpoint textattack/bert-base-uncased-MNLI --checkpoint-label-order textattack --batch-size 32 --max-length 128

B. TextAttack MNLI dense baseline, long-sequence max_length=512:
   python eval_glue_dense_baseline.py --task mnli --checkpoint textattack/bert-base-uncased-MNLI --checkpoint-label-order textattack --batch-size 16 --max-length 512

C. MRPC dense baseline:
   python eval_glue_dense_baseline.py --task mrpc --checkpoint textattack/bert-base-uncased-MRPC --batch-size 32 --max-length 128

This script evaluates dense attention only. It does not enable the TPDCIM patch.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate dense GLUE baselines without TPDCIM sparse/TCS patch.",
        epilog=PRESET_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", default="mnli", choices=sorted(TASKS))
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint to evaluate. Defaults to the task checkpoint from eval_glue_tpdcim.py.",
    )
    parser.add_argument(
        "--checkpoint-label-order",
        choices=["default", "dataset", "textattack"],
        default="default",
        help="Use textattack for textattack/bert-base-uncased-MNLI; use dataset for locally trained checkpoints.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def make_loader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )


def format_duration(seconds):
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def maybe_print_progress(label, batch_index, total_batches, examples_done, total_examples, started_at, every, no_progress):
    if no_progress or every <= 0:
        return
    if batch_index != 1 and batch_index != total_batches and batch_index % every != 0:
        return

    elapsed = time.time() - started_at
    avg_per_batch = elapsed / max(1, batch_index)
    eta = avg_per_batch * max(0, total_batches - batch_index)
    examples_per_second = examples_done / elapsed if elapsed > 0 else 0.0
    print(
        f"[{label}] batch {batch_index}/{total_batches} "
        f"examples={examples_done}/{total_examples} "
        f"elapsed={format_duration(elapsed)} eta~{format_duration(eta)} "
        f"ex/s={examples_per_second:.2f}",
        flush=True,
    )


def evaluate_dense(model, task_name, task_cfg, loader, device, args):
    preds = []
    labels = []
    total_batches = len(loader)
    total_examples = len(loader.dataset) if hasattr(loader, "dataset") else "?"
    examples_done = 0
    started_at = time.time()

    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            labels_batch = batch.pop("labels")
            examples_done += int(labels_batch.shape[0])
            inputs = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**inputs)
            batch_preds = outputs.logits.detach().cpu().argmax(dim=-1)
            batch_preds = apply_prediction_label_map(batch_preds, task_cfg)
            preds.extend(batch_preds.tolist())
            labels.extend(labels_batch.tolist())
            maybe_print_progress(
                "dense_baseline",
                batch_index,
                total_batches,
                examples_done,
                total_examples,
                started_at,
                args.progress_every,
                args.no_progress,
            )

    return compute_metrics(task_name, preds, labels)


def default_output_csv(args):
    examples = "full" if args.max_examples is None else str(args.max_examples)
    return os.path.join(
        "results",
        f"dense_baseline_{args.task}_{examples}_len{args.max_length}.csv",
    )


def write_csv(path, row):
    if path is None:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    if args.output_csv is None:
        args.output_csv = default_output_csv(args)

    task_cfg = resolve_task_cfg(args.task, args)
    checkpoint = task_cfg["checkpoint"]

    print("track:", TRACK_NAME)
    print("task:", args.task)
    print("checkpoint:", checkpoint)
    print("checkpoint_label_order:", args.checkpoint_label_order)
    print("split:", task_cfg["split"])
    print("dataset_repo:", GLUE_DATASET_REPO)
    print("device:", args.device)
    print("batch_size:", args.batch_size)
    print("max_examples:", "full" if args.max_examples is None else args.max_examples)
    print("max_length:", args.max_length)
    label_map = task_cfg.get("prediction_label_map")
    print("prediction_label_map:", label_map if label_map is not None else "dataset order")
    print("output_csv:", args.output_csv)
    print("Note: dense attention only; TPDCIM sparse/int8/bitserial/TCS patch is not enabled.")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    dataset = tokenize_dataset(
        tokenizer=tokenizer,
        task_cfg=task_cfg,
        max_length=args.max_length,
        max_examples=args.max_examples,
    )
    input_stats = compute_input_length_stats(dataset, tile_n=args.max_length, max_length=args.max_length)
    print(
        "input length stats: "
        f"min/mean/max_nonpad={input_stats['min_nonpad_tokens']}/"
        f"{fmt_float(input_stats['mean_nonpad_tokens'])}/"
        f"{input_stats['max_nonpad_tokens']}"
    )

    loader = make_loader(dataset, args.batch_size)
    model = load_model(checkpoint, args.device)
    metrics = evaluate_dense(model, args.task, task_cfg, loader, args.device, args)
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    row = {
        "track": TRACK_NAME,
        "task": args.task,
        "checkpoint": checkpoint,
        "split": task_cfg["split"],
        "max_examples": "full" if args.max_examples is None else args.max_examples,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "primary_metric": task_cfg["primary_metric"],
        "accuracy": metrics["accuracy"],
        "f1": metrics.get("f1"),
        "score": metrics[task_cfg["primary_metric"]],
        "min_nonpad_tokens": input_stats["min_nonpad_tokens"],
        "mean_nonpad_tokens": input_stats["mean_nonpad_tokens"],
        "max_nonpad_tokens": input_stats["max_nonpad_tokens"],
    }
    write_csv(args.output_csv, row)

    print("\nDENSE GLUE BASELINE SUMMARY")
    print("task | checkpoint | max_length | accuracy | f1 | score")
    print("---- | ---------- | ---------- | -------- | -- | -----")
    print(
        " | ".join(
            [
                args.task,
                checkpoint,
                str(args.max_length),
                fmt_float(row["accuracy"]),
                fmt_float(row["f1"]),
                fmt_float(row["score"]),
            ]
        )
    )
    print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

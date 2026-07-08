import argparse
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tpdcim_bert_patch import enable_qk_tiling


MRPC_CHECKPOINT = "textattack/bert-base-uncased-MRPC"
MNLI_CHECKPOINT = "textattack/bert-base-uncased-MNLI"
GLUE_DATASET_REPO = "nyu-mll/glue"
BW_B = [22, 22, 20, 18, 16, 16, 14, 12]
BW_E = [28, 24, 22, 20, 18, 16, 14, 10]


@dataclass(frozen=True)
class EvalCase:
    name: str
    enable_patch: bool
    enable_sparse: bool = False
    qk_mode: str = "fp32"
    enable_tcs: bool = False
    tcs_thresholds_name: str = "none"
    tcs_thresholds: Optional[List[int]] = None


CASES = [
    EvalCase("baseline", enable_patch=False),
    EvalCase("dense_fp32", enable_patch=True, enable_sparse=False, qk_mode="fp32"),
    EvalCase("dense_int8", enable_patch=True, enable_sparse=False, qk_mode="int8"),
    EvalCase("dense_bitserial", enable_patch=True, enable_sparse=False, qk_mode="bitserial"),
    EvalCase("sparse_fp32", enable_patch=True, enable_sparse=True, qk_mode="fp32"),
    EvalCase("sparse_int8", enable_patch=True, enable_sparse=True, qk_mode="int8"),
    EvalCase("sparse_bitserial", enable_patch=True, enable_sparse=True, qk_mode="bitserial"),
    EvalCase(
        "sparse_bitserial_tcs_bw_B",
        enable_patch=True,
        enable_sparse=True,
        qk_mode="bitserial",
        enable_tcs=True,
        tcs_thresholds_name="bw_B",
        tcs_thresholds=BW_B,
    ),
    EvalCase(
        "sparse_bitserial_tcs_bw_E",
        enable_patch=True,
        enable_sparse=True,
        qk_mode="bitserial",
        enable_tcs=True,
        tcs_thresholds_name="bw_E",
        tcs_thresholds=BW_E,
    ),
]


TASKS = {
    "mrpc": {
        "checkpoint": MRPC_CHECKPOINT,
        "dataset_name": "mrpc",
        "split": "validation",
        "text_columns": ("sentence1", "sentence2"),
        "metric_names": ("accuracy", "f1"),
        "primary_metric": "f1",
        "prediction_label_map": None,
    },
    "mnli": {
        "checkpoint": MNLI_CHECKPOINT,
        "dataset_name": "mnli",
        "split": "validation_matched",
        "text_columns": ("premise", "hypothesis"),
        "metric_names": ("accuracy",),
        "primary_metric": "accuracy",
        # TextAttack MNLI logits follow the old Transformers GLUE order:
        # contradiction, entailment, neutral. HF GLUE ids are:
        # entailment, neutral, contradiction. This maps model ids to dataset ids.
        "prediction_label_map": [2, 0, 1],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned BERT checkpoints with TPDCIM attention patch on GLUE."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["mrpc", "mnli"],
        choices=sorted(TASKS),
        help="GLUE tasks to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_glue_split(dataset_name, split):
    """Load GLUE from its canonical HF dataset repo.

    Some datasets/huggingface_hub versions no longer resolve the historical
    short name `glue` cleanly, so prefer the namespace-qualified repo id.
    """
    return load_dataset(GLUE_DATASET_REPO, dataset_name, split=split)


def tokenize_dataset(tokenizer, task_cfg, max_length, max_examples=None):
    dataset = load_glue_split(task_cfg["dataset_name"], task_cfg["split"])
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

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
    remove_columns = [c for c in tokenized.column_names if c not in keep_columns]
    if remove_columns:
        tokenized = tokenized.remove_columns(remove_columns)
    return tokenized


def collate_batch(batch):
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        collated[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
    return collated


def load_model(checkpoint, device):
    # Force safetensors so torch 2.5 environments avoid transformers' torch.load CVE guard.
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        use_safetensors=True,
    )
    model.to(device)
    model.eval()
    return model


def apply_case_patch(model, case, args):
    if not case.enable_patch:
        return
    enable_qk_tiling(
        model,
        tile_n=args.tile_n,
        enable_sparse=case.enable_sparse,
        local_window=args.local_window,
        global_blocks=(0,),
        num_random_blocks=0,
        qk_mode=case.qk_mode,
        enable_tcs=case.enable_tcs,
        tcs_thresholds=case.tcs_thresholds,
    )


def empty_stats():
    return {
        "qk_mode": "baseline",
        "qk_computed_tiles": 0,
        "qk_skipped_tiles": 0,
        "qk_total_tiles": 0,
        "qk_sparse_density": 0.0,
        "qk_tile_skip_ratio": 0.0,
        "bit_ops_total": 0,
        "bit_ops_skipped_by_tcs": 0,
        "tcs_rows_total": 0,
        "tcs_rows_skipped": 0,
        "tcs_row_skip_ratio": 0.0,
    }


def collect_batch_stats(model):
    stats = empty_stats()
    qk_modes = set()

    for layer in model.bert.encoder.layer:
        layer_stats = getattr(layer.attention.self, "tpdcim_last_stats", None)
        if layer_stats is None:
            continue
        qk_modes.add(layer_stats.get("qk_mode", "fp32"))
        stats["qk_computed_tiles"] += layer_stats.get("computed_qk_tiles", 0)
        stats["qk_skipped_tiles"] += layer_stats.get("skipped_qk_tiles", 0)
        stats["qk_total_tiles"] += layer_stats.get("total_qk_tiles", 0)
        stats["bit_ops_total"] += layer_stats.get("bit_ops_total", 0)
        stats["bit_ops_skipped_by_tcs"] += layer_stats.get("bit_ops_skipped_by_tcs", 0)
        stats["tcs_rows_total"] += layer_stats.get("tcs_rows_total", 0)
        stats["tcs_rows_skipped"] += layer_stats.get("tcs_rows_skipped", 0)

    if qk_modes:
        stats["qk_mode"] = ",".join(sorted(qk_modes))
    return stats


def merge_stats(total, batch_stats):
    if batch_stats["qk_mode"] != "baseline":
        total["qk_mode"] = batch_stats["qk_mode"]
    for key in (
        "qk_computed_tiles",
        "qk_skipped_tiles",
        "qk_total_tiles",
        "bit_ops_total",
        "bit_ops_skipped_by_tcs",
        "tcs_rows_total",
        "tcs_rows_skipped",
    ):
        total[key] += batch_stats[key]


def finalize_stats(stats):
    total_tiles = stats["qk_total_tiles"]
    if total_tiles > 0:
        stats["qk_sparse_density"] = stats["qk_computed_tiles"] / total_tiles
        stats["qk_tile_skip_ratio"] = stats["qk_skipped_tiles"] / total_tiles
    if stats["tcs_rows_total"] > 0:
        stats["tcs_row_skip_ratio"] = stats["tcs_rows_skipped"] / stats["tcs_rows_total"]
    return stats


def apply_prediction_label_map(batch_preds, task_cfg):
    prediction_label_map = task_cfg.get("prediction_label_map")
    if prediction_label_map is None:
        return batch_preds
    mapping = torch.tensor(prediction_label_map, dtype=batch_preds.dtype, device=batch_preds.device)
    return mapping[batch_preds]


def compute_metrics(task_name, preds, labels):
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    accuracy = correct / len(labels) if labels else 0.0
    metrics = {"accuracy": accuracy}

    if task_name == "mrpc":
        tp = sum(int(p == 1 and y == 1) for p, y in zip(preds, labels))
        fp = sum(int(p == 1 and y == 0) for p, y in zip(preds, labels))
        fn = sum(int(p == 0 and y == 1) for p, y in zip(preds, labels))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        metrics["f1"] = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    else:
        metrics["f1"] = None

    return metrics


def evaluate_case(task_name, task_cfg, dataset, case, args):
    model = load_model(task_cfg["checkpoint"], args.device)
    apply_case_patch(model, case, args)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    preds = []
    labels = []
    logits_chunks = []
    total_stats = empty_stats()

    with torch.no_grad():
        for batch in loader:
            labels_batch = batch.pop("labels")
            inputs = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits.detach().cpu()
            batch_preds = logits.argmax(dim=-1)
            batch_preds = apply_prediction_label_map(batch_preds, task_cfg)

            preds.extend(batch_preds.tolist())
            labels.extend(labels_batch.tolist())
            logits_chunks.append(logits)
            merge_stats(total_stats, collect_batch_stats(model))

    metrics = compute_metrics(task_name, preds, labels)
    stats = finalize_stats(total_stats)
    logits = torch.cat(logits_chunks, dim=0)

    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    return metrics, stats, logits


def fmt_float(value, digits=6):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def print_table(rows):
    headers = [
        "task",
        "case",
        "accuracy",
        "f1",
        "accuracy_delta",
        "f1_delta",
        "qk_mode",
        "qk_computed_tiles",
        "qk_skipped_tiles",
        "qk_sparse_density",
        "qk_tile_skip_ratio",
        "bit_ops_total",
        "tcs_row_skip_ratio",
    ]
    print("\n" + "=" * 120)
    print("FINAL GLUE TPDCIM SUMMARY")
    print("=" * 120)
    print(" | ".join(headers))
    print(" | ".join("-" * len(h) for h in headers))
    for row in rows:
        print(
            " | ".join(
                [
                    row["task"],
                    row["case"],
                    fmt_float(row["accuracy"]),
                    fmt_float(row["f1"]),
                    fmt_float(row["accuracy_delta"]),
                    fmt_float(row["f1_delta"]),
                    row["qk_mode"],
                    str(row["qk_computed_tiles"]),
                    str(row["qk_skipped_tiles"]),
                    fmt_float(row["qk_sparse_density"]),
                    fmt_float(row["qk_tile_skip_ratio"]),
                    str(row["bit_ops_total"]),
                    fmt_float(row["tcs_row_skip_ratio"]),
                ]
            )
        )


def run_equivalence_checks(task_name, logits_by_case):
    checks = [
        ("dense_int8", "dense_bitserial"),
        ("sparse_int8", "sparse_bitserial"),
    ]
    for left, right in checks:
        if left not in logits_by_case or right not in logits_by_case:
            continue
        max_diff = (logits_by_case[left] - logits_by_case[right]).abs().max().item()
        print(f"[{task_name}] {left} vs {right} max logit diff: {max_diff:.8f}")
        if max_diff > 1e-6:
            raise AssertionError(f"{task_name}: {left} and {right} logits differ")


def main():
    args = parse_args()
    print("device:", args.device)
    print("batch_size:", args.batch_size)
    print("max_length:", args.max_length)
    print("tile_n:", args.tile_n)
    print("local_window:", args.local_window)
    print("max_examples:", args.max_examples)
    print("dataset_repo:", GLUE_DATASET_REPO)
    print("Note: GLUE task scores are trend/sanity checks, not exact TP-DCIM paper reproduction.")

    all_rows = []

    for task_name in args.tasks:
        task_cfg = TASKS[task_name]
        print("\n" + "=" * 80)
        print(f"TASK: {task_name} ({task_cfg['checkpoint']}, split={task_cfg['split']})")
        label_map = task_cfg.get("prediction_label_map")
        if label_map is not None:
            print(f"prediction_label_map: {label_map}")
        print("=" * 80)

        tokenizer = AutoTokenizer.from_pretrained(task_cfg["checkpoint"])
        dataset = tokenize_dataset(
            tokenizer=tokenizer,
            task_cfg=task_cfg,
            max_length=args.max_length,
            max_examples=args.max_examples,
        )

        baseline_metrics = None
        logits_by_case = {}

        for case in CASES:
            print(f"Running {task_name}/{case.name} ...")
            metrics, stats, logits = evaluate_case(task_name, task_cfg, dataset, case, args)
            logits_by_case[case.name] = logits

            if case.name == "baseline":
                baseline_metrics = metrics
            accuracy_delta = metrics["accuracy"] - baseline_metrics["accuracy"]
            f1_delta = None
            if metrics.get("f1") is not None and baseline_metrics.get("f1") is not None:
                f1_delta = metrics["f1"] - baseline_metrics["f1"]

            row = {
                "task": task_name,
                "case": case.name,
                "accuracy": metrics["accuracy"],
                "f1": metrics.get("f1"),
                "accuracy_delta": accuracy_delta,
                "f1_delta": f1_delta,
                "qk_mode": stats["qk_mode"],
                "qk_computed_tiles": stats["qk_computed_tiles"],
                "qk_skipped_tiles": stats["qk_skipped_tiles"],
                "qk_sparse_density": stats["qk_sparse_density"],
                "qk_tile_skip_ratio": stats["qk_tile_skip_ratio"],
                "bit_ops_total": stats["bit_ops_total"],
                "tcs_row_skip_ratio": stats["tcs_row_skip_ratio"],
            }
            all_rows.append(row)
            print(
                f"  accuracy={fmt_float(row['accuracy'])} "
                f"f1={fmt_float(row['f1'])} "
                f"qk_mode={row['qk_mode']} "
                f"computed/skipped={row['qk_computed_tiles']}/{row['qk_skipped_tiles']} "
                f"bit_ops={row['bit_ops_total']} "
                f"tcs_row_skip={fmt_float(row['tcs_row_skip_ratio'])}"
            )

        run_equivalence_checks(task_name, logits_by_case)

    print_table(all_rows)


if __name__ == "__main__":
    main()

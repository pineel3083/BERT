import argparse
import csv
import os

import torch
from transformers import AutoTokenizer

from eval_glue_tpdcim import (
    GLUE_DATASET_REPO,
    TASKS,
    ceil_div,
    compute_input_length_stats,
    compute_sparse_real_token_stats,
    fmt_float,
    get_num_layers,
    load_model,
    resolve_task_cfg,
    tokenize_dataset,
)
from eval_glue_tcs_greedy import (
    evaluate_loaded_model,
    format_thresholds,
    format_thresholds_msb_to_lsb,
    load_patched_sparse_model,
    make_loader,
    make_summary_row,
    parse_threshold_vector,
    set_tcs_thresholds,
)
from tpdcim_bert_patch import TCS_ACTIVE_RULE, TCS_THRESHOLD_ORDER


TRACK_NAME = "fixed_tcs_threshold_candidates"
DEFAULT_CANDIDATES = [
    ("tcs_conservative", [24, 32, 24, 24, 16, 16, 16, 16]),
    ("tcs_paper_like_mild", [40, 32, 24, 20, 16, 12, 8, 0]),
    ("tcs_paper_like_medium", [48, 36, 24, 20, 16, 12, 8, 0]),
    ("tcs_paper_like_aggressive", [48, 40, 36, 32, 24, 16, 8, 0]),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed TCS threshold vectors against the sparse_bitserial BigBird proxy."
    )
    parser.add_argument("--task", default="mnli", choices=sorted(TASKS))
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint override for this task, e.g. checkpoints/dense_mnli_len128.",
    )
    parser.add_argument(
        "--checkpoint-label-order",
        choices=["default", "dataset", "textattack"],
        default="default",
        help="Use 'dataset' for checkpoints trained by finetune_glue_dense.py.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=16)
    parser.add_argument("--local-window", type=int, default=1)
    parser.add_argument("--num-random-blocks", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--threshold-candidates-lsb-to-msb",
        nargs="*",
        default=None,
        help="Optional list of 8-value vectors in LSB-to-MSB order.",
    )
    parser.add_argument(
        "--threshold-names",
        nargs="*",
        default=None,
        help="Optional names for custom threshold vectors.",
    )
    parser.add_argument("--batch-progress-every", type=int, default=50)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def default_output_csv(args):
    if args.task == "mnli":
        examples = "full" if args.max_examples is None else str(args.max_examples)
        filename = (
            f"fixed_tcs_threshold_candidates_{args.task}_{examples}_"
            f"len{args.max_length}_tile{args.tile_n}_random{args.num_random_blocks}.csv"
        )
    else:
        filename = (
            f"fixed_tcs_threshold_candidates_{args.task}_"
            f"len{args.max_length}_tile{args.tile_n}_random{args.num_random_blocks}.csv"
        )
    return os.path.join("results", filename)


def get_threshold_candidates(args):
    if not args.threshold_candidates_lsb_to_msb:
        return [(name, list(values)) for name, values in DEFAULT_CANDIDATES]

    candidates = []
    names = args.threshold_names or []
    if names and len(names) != len(args.threshold_candidates_lsb_to_msb):
        raise ValueError("--threshold-names must match the number of threshold vectors")

    for index, raw_vector in enumerate(args.threshold_candidates_lsb_to_msb):
        name = names[index] if names else f"tcs_candidate_{index}"
        vector = parse_threshold_vector(raw_vector, f"--threshold-candidates-lsb-to-msb[{index}]")
        candidates.append((name, vector))
    return candidates


def make_output_row(args, task_cfg, threshold_name, method, row, software_score, sparse_score):
    primary_metric = task_cfg["primary_metric"]
    return {
        "track": TRACK_NAME,
        "task": args.task,
        "checkpoint": task_cfg["checkpoint"],
        "primary_metric": primary_metric,
        "method": method,
        "threshold_name": threshold_name,
        "max_examples": "full" if args.max_examples is None else args.max_examples,
        "max_length": args.max_length,
        "tile_n": args.tile_n,
        "num_blocks": ceil_div(args.max_length, args.tile_n),
        "num_random_blocks": args.num_random_blocks,
        "thresholds_lsb_to_msb": row["thresholds_lsb_to_msb"],
        "thresholds_msb_to_lsb": row["thresholds_msb_to_lsb"],
        "software_score": software_score,
        "sparse_reference_score": sparse_score,
        "score": row["primary_score"],
        "accuracy": row["accuracy"],
        "f1": row["f1"],
        "score_drop_vs_sparse_reference": row["score_drop_vs_sparse_reference"],
        "score_delta_vs_software_baseline": row["score_delta_vs_software_baseline"],
        "qk_sparse_density": row["qk_sparse_density"],
        "qk_tile_skip_ratio": row["qk_tile_skip_ratio"],
        "tcs_row_skip_ratio": row["tcs_row_skip_ratio"],
        "reduction_vs_dense_str": row["reduction_vs_dense_str"],
        "reduction_vs_bigbird_str": row["reduction_vs_bigbird_str"],
        "operation_saving_vs_dense": row["operation_saving_vs_dense"],
        "operation_saving_vs_bigbird": row["operation_saving_vs_bigbird"],
        "num_changed_predictions_vs_sparse_reference": row[
            "num_changed_predictions_vs_sparse_reference"
        ],
        "changed_prediction_ratio_vs_sparse_reference": row[
            "changed_prediction_ratio_vs_sparse_reference"
        ],
        "mean_logit_diff_vs_sparse_reference": row[
            "mean_logit_diff_vs_sparse_reference"
        ],
        "max_logit_diff_vs_sparse_reference": row[
            "max_logit_diff_vs_sparse_reference"
        ],
    }


def write_csv(path, rows):
    if not rows:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    columns = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def compact_score(row):
    return row["f1"] if row["primary_metric"] == "f1" else row["accuracy"]


def print_compact_table(rows):
    headers = [
        "Task",
        "Setting",
        "Method",
        "Score",
        "Drop vs Sparse",
        "TCS row skip",
        "Reduction vs Dense",
        "Reduction vs BigBird",
    ]
    print("\nCOMPACT FIXED TCS THRESHOLD SUMMARY")
    print("Note: QK activity proxy, BigBird proxy, and reduction_vs_bigbird proxy; not exact paper reproduction.")
    print(" | ".join(headers))
    print(" | ".join("-" * len(header) for header in headers))
    for row in rows:
        setting = f"len{row['max_length']} tile{row['tile_n']} rand{row['num_random_blocks']}"
        drop = "-" if row["method"] == "Software baseline" else fmt_float(row["score_drop_vs_sparse_reference"])
        print(
            " | ".join(
                [
                    row["task"].upper(),
                    setting,
                    row["method"],
                    fmt_float(compact_score(row), 6),
                    drop,
                    fmt_float(row["tcs_row_skip_ratio"], 6),
                    row["reduction_vs_dense_str"],
                    row["reduction_vs_bigbird_str"],
                ]
            )
        )


def main():
    args = parse_args()
    output_csv = args.output_csv or default_output_csv(args)
    candidates = get_threshold_candidates(args)
    task_cfg = resolve_task_cfg(args.task, args)

    print("track:", TRACK_NAME)
    print("task:", args.task)
    print("checkpoint:", task_cfg["checkpoint"])
    print("checkpoint_label_order:", args.checkpoint_label_order)
    print("split:", task_cfg["split"])
    print("dataset_repo:", GLUE_DATASET_REPO)
    print("device:", args.device)
    print("max_examples:", "full" if args.max_examples is None else args.max_examples)
    print("max_length:", args.max_length)
    print("tile_n:", args.tile_n)
    print("num_blocks:", ceil_div(args.max_length, args.tile_n))
    print("num_random_blocks:", args.num_random_blocks)
    print("tcs_threshold_order:", TCS_THRESHOLD_ORDER)
    print("tcs_active_rule:", TCS_ACTIVE_RULE)
    print("output_csv:", output_csv)
    print("Note: sparse_bitserial is the BigBird proxy reference; software baseline is context only.")
    print("Note: QK activity proxy, reduction_vs_dense proxy, and reduction_vs_bigbird proxy.")
    print("Note: not exact TP-DCIM paper reproduction, measured CUDA speedup, energy, or cycle-accurate simulation.")

    for name, vector in candidates:
        print(
            f"candidate {name}: lsb_to_msb={format_thresholds(vector)} "
            f"msb_to_lsb={format_thresholds_msb_to_lsb(vector)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(task_cfg["checkpoint"])
    num_layers = get_num_layers(task_cfg["checkpoint"])
    dataset = tokenize_dataset(
        tokenizer=tokenizer,
        task_cfg=task_cfg,
        max_length=args.max_length,
        max_examples=args.max_examples,
    )
    loader = make_loader(dataset, args)

    input_stats = compute_input_length_stats(dataset, args.tile_n, args.max_length)
    sparse_real_stats = compute_sparse_real_token_stats(dataset, args, num_layers)
    print(
        "input length stats: "
        f"min/mean/max_nonpad={input_stats['min_nonpad_tokens']}/"
        f"{fmt_float(input_stats['mean_nonpad_tokens'])}/"
        f"{input_stats['max_nonpad_tokens']} "
        f"min/mean/max_real_blocks={input_stats['min_real_blocks']}/"
        f"{fmt_float(input_stats['mean_real_blocks'])}/"
        f"{input_stats['max_real_blocks']}"
    )
    print(
        "sparse real-token estimate: "
        f"padding_ratio={fmt_float(sparse_real_stats['skipped_tile_padding_ratio'])} "
        f"real_ratio={fmt_float(sparse_real_stats['skipped_tile_real_ratio'])} "
        f"real_density={fmt_float(sparse_real_stats['real_qk_sparse_density'])} "
        f"real_skip={fmt_float(sparse_real_stats['real_qk_tile_skip_ratio'])}"
    )

    print("\nRunning software baseline ...", flush=True)
    software_model = load_model(task_cfg["checkpoint"], args.device)
    software_metrics, software_stats, software_logits, software_predictions = evaluate_loaded_model(
        software_model,
        args.task,
        task_cfg,
        loader,
        args.device,
        progress_label="software_baseline",
        progress_every=args.batch_progress_every,
        no_progress=args.no_progress,
    )
    software_score = software_metrics[task_cfg["primary_metric"]]
    del software_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("Running sparse_bitserial BigBird proxy reference ...", flush=True)
    sparse_model = load_patched_sparse_model(task_cfg, args, enable_tcs=False)
    sparse_metrics, sparse_stats, sparse_logits, sparse_predictions = evaluate_loaded_model(
        sparse_model,
        args.task,
        task_cfg,
        loader,
        args.device,
        progress_label="sparse_bitserial_reference",
        progress_every=args.batch_progress_every,
        no_progress=args.no_progress,
    )
    sparse_score = sparse_metrics[task_cfg["primary_metric"]]
    del sparse_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    software_row = make_summary_row(
        stage="software_baseline",
        task_name=args.task,
        task_cfg=task_cfg,
        args=args,
        metrics=software_metrics,
        stats=software_stats,
        logits=software_logits,
        predictions=software_predictions,
        software_score=software_score,
        sparse_score=sparse_score,
        sparse_logits=sparse_logits,
        sparse_predictions=sparse_predictions,
        thresholds=None,
        enable_tcs=False,
    )
    sparse_row = make_summary_row(
        stage="sparse_bitserial_reference",
        task_name=args.task,
        task_cfg=task_cfg,
        args=args,
        metrics=sparse_metrics,
        stats=sparse_stats,
        logits=sparse_logits,
        predictions=sparse_predictions,
        software_score=software_score,
        sparse_score=sparse_score,
        sparse_logits=sparse_logits,
        sparse_predictions=sparse_predictions,
        thresholds=None,
        enable_tcs=False,
    )

    output_rows = [
        make_output_row(
            args,
            task_cfg,
            "software_baseline",
            "Software baseline",
            software_row,
            software_score,
            sparse_score,
        ),
        make_output_row(
            args,
            task_cfg,
            "bigbird_proxy",
            "BigBird proxy",
            sparse_row,
            software_score,
            sparse_score,
        ),
    ]

    print(
        "reference scores: "
        f"software_{task_cfg['primary_metric']}={fmt_float(software_score)} "
        f"sparse_{task_cfg['primary_metric']}={fmt_float(sparse_score)}",
        flush=True,
    )

    first_name, first_vector = candidates[0]
    tcs_model = load_patched_sparse_model(task_cfg, args, enable_tcs=True, thresholds=first_vector)
    for name, vector in candidates:
        print(
            f"Running {name}: lsb_to_msb={format_thresholds(vector)} "
            f"msb_to_lsb={format_thresholds_msb_to_lsb(vector)}",
            flush=True,
        )
        set_tcs_thresholds(tcs_model, vector)
        metrics, stats, logits, predictions = evaluate_loaded_model(
            tcs_model,
            args.task,
            task_cfg,
            loader,
            args.device,
            progress_label=name,
            progress_every=args.batch_progress_every,
            no_progress=args.no_progress,
        )
        row = make_summary_row(
            stage=name,
            task_name=args.task,
            task_cfg=task_cfg,
            args=args,
            metrics=metrics,
            stats=stats,
            logits=logits,
            predictions=predictions,
            software_score=software_score,
            sparse_score=sparse_score,
            sparse_logits=sparse_logits,
            sparse_predictions=sparse_predictions,
            thresholds=vector,
            enable_tcs=True,
        )
        method = name.replace("tcs_", "TCS ").replace("_", " ")
        output_rows.append(make_output_row(args, task_cfg, name, method, row, software_score, sparse_score))

    del tcs_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    write_csv(output_csv, output_rows)
    print_compact_table(output_rows)
    print(f"\nSaved CSV: {output_csv}")


if __name__ == "__main__":
    main()

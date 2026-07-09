import argparse
import csv
import os
import time

import torch
from transformers import AutoTokenizer

from eval_glue_tpdcim import (
    GLUE_DATASET_REPO,
    TASKS,
    ceil_div,
    compute_input_length_stats,
    compute_metrics,
    fmt_float,
    resolve_task_cfg,
    tokenize_dataset,
)
from eval_glue_tcs_greedy import (
    evaluate_loaded_model,
    format_thresholds,
    format_thresholds_msb_to_lsb,
    load_patched_sparse_model,
    make_loader,
    parse_threshold_vector,
    set_tcs_thresholds,
)
from tpdcim_bert_patch import TCS_ACTIVE_RULE, TCS_THRESHOLD_ORDER


TRACK_NAME = "tcs_only_threshold_eval"


PRESET_HELP = """
Recommended run for after-FT MNLI TCS-medium full validation:

python eval_tcs_only.py \
  --task mnli \
  --checkpoint checkpoints/sparse_mnli_textattack_len128_tile16_rand1_full_e1 \
  --checkpoint-label-order textattack \
  --reference-score 0.8407539480387163 \
  --software-score 0.845746306673459 \
  --batch-size 4 \
  --max-length 128 \
  --tile-n 16 \
  --num-random-blocks 1 \
  --threshold-candidates-lsb-to-msb 48,36,24,20,16,12,8,0 \
  --threshold-names tcs_medium

This script evaluates only sparse_bitserial+TCS candidates. It does not run dense or sparse references.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate only TPDCIM sparse_bitserial+TCS threshold candidates on GLUE.",
        epilog=PRESET_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", default="mnli", choices=sorted(TASKS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-label-order",
        choices=["default", "dataset", "textattack"],
        default="default",
        help="Use textattack for textattack/bert-base-uncased-MNLI lineage; dataset for locally trained dataset-order checkpoints.",
    )
    parser.add_argument(
        "--reference-score",
        type=float,
        required=True,
        help="Sparse reference score to compare against, usually sparse-aware FT sparse full accuracy.",
    )
    parser.add_argument(
        "--software-score",
        type=float,
        default=None,
        help="Optional dense software baseline score for total delta reporting.",
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
        nargs="+",
        required=True,
        help="One or more 8-value threshold vectors in simulator LSB-to-MSB order.",
    )
    parser.add_argument(
        "--threshold-names",
        nargs="*",
        default=None,
        help="Optional names matching --threshold-candidates-lsb-to-msb.",
    )
    parser.add_argument("--batch-progress-every", type=int, default=50)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def default_output_csv(args):
    examples = "full" if args.max_examples is None else str(args.max_examples)
    return os.path.join(
        "results",
        f"tcs_only_{args.task}_{examples}_len{args.max_length}_tile{args.tile_n}_random{args.num_random_blocks}.csv",
    )


def get_threshold_candidates(args):
    names = args.threshold_names or []
    raw_vectors = args.threshold_candidates_lsb_to_msb
    if names and len(names) != len(raw_vectors):
        raise ValueError("--threshold-names must match the number of threshold vectors")

    candidates = []
    for index, raw_vector in enumerate(raw_vectors):
        name = names[index] if names else f"tcs_candidate_{index}"
        vector = parse_threshold_vector(raw_vector, f"--threshold-candidates-lsb-to-msb[{index}]")
        candidates.append((name, vector))
    return candidates


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


def format_duration(seconds):
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def activity_from_stats(stats):
    density = stats["qk_sparse_density"]
    tcs_keep_ratio = 1.0 - stats["tcs_row_skip_ratio"]
    reduction_dense = density * tcs_keep_ratio
    reduction_bigbird = tcs_keep_ratio
    return {
        "reduction_vs_dense": reduction_dense,
        "reduction_vs_dense_str": f"x{reduction_dense:.3f}",
        "reduction_vs_bigbird": reduction_bigbird,
        "reduction_vs_bigbird_str": f"x{reduction_bigbird:.3f}",
        "operation_saving_vs_dense": 1.0 - reduction_dense,
        "operation_saving_vs_bigbird": 1.0 - reduction_bigbird,
    }


def build_output_row(args, task_cfg, threshold_name, thresholds, metrics, stats):
    primary_metric = task_cfg["primary_metric"]
    score = metrics[primary_metric]
    activity = activity_from_stats(stats)
    software_delta = None if args.software_score is None else score - args.software_score
    return {
        "track": TRACK_NAME,
        "task": args.task,
        "checkpoint": task_cfg["checkpoint"],
        "primary_metric": primary_metric,
        "method": "TCS only",
        "threshold_name": threshold_name,
        "max_examples": "full" if args.max_examples is None else args.max_examples,
        "max_length": args.max_length,
        "tile_n": args.tile_n,
        "num_blocks": ceil_div(args.max_length, args.tile_n),
        "num_random_blocks": args.num_random_blocks,
        "thresholds_lsb_to_msb": format_thresholds(thresholds),
        "thresholds_msb_to_lsb": format_thresholds_msb_to_lsb(thresholds),
        "software_score": args.software_score,
        "sparse_reference_score": args.reference_score,
        "score": score,
        "accuracy": metrics["accuracy"],
        "f1": metrics.get("f1"),
        "score_drop_vs_sparse_reference": args.reference_score - score,
        "score_delta_vs_sparse_reference": score - args.reference_score,
        "score_delta_vs_software_baseline": software_delta,
        "qk_sparse_density": stats["qk_sparse_density"],
        "qk_tile_skip_ratio": stats["qk_tile_skip_ratio"],
        "qk_computed_tiles": stats["qk_computed_tiles"],
        "qk_skipped_tiles": stats["qk_skipped_tiles"],
        "bit_ops_total": stats["bit_ops_total"],
        "tcs_rows_total": stats["tcs_rows_total"],
        "tcs_rows_skipped": stats["tcs_rows_skipped"],
        "tcs_row_skip_ratio": stats["tcs_row_skip_ratio"],
        "reduction_vs_dense_str": activity["reduction_vs_dense_str"],
        "reduction_vs_bigbird_str": activity["reduction_vs_bigbird_str"],
        "operation_saving_vs_dense": activity["operation_saving_vs_dense"],
        "operation_saving_vs_bigbird": activity["operation_saving_vs_bigbird"],
    }


def print_compact_table(rows):
    headers = [
        "Task",
        "Threshold",
        "Score",
        "Drop vs Sparse",
        "Delta vs SW",
        "TCS row skip",
        "Reduction vs Dense",
        "Reduction vs BigBird",
    ]
    print("\nCOMPACT TCS-ONLY SUMMARY")
    print("Note: reference scores are provided by CLI; this script only evaluates TCS candidates.")
    print(" | ".join(headers))
    print(" | ".join("-" * len(header) for header in headers))
    for row in rows:
        print(
            " | ".join(
                [
                    row["task"].upper(),
                    row["threshold_name"],
                    fmt_float(row["score"], 6),
                    fmt_float(row["score_drop_vs_sparse_reference"], 6),
                    fmt_float(row["score_delta_vs_software_baseline"], 6),
                    fmt_float(row["tcs_row_skip_ratio"], 6),
                    row["reduction_vs_dense_str"],
                    row["reduction_vs_bigbird_str"],
                ]
            )
        )


def main():
    args = parse_args()
    args.output_csv = args.output_csv or default_output_csv(args)
    task_cfg = resolve_task_cfg(args.task, args)
    candidates = get_threshold_candidates(args)

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
    print("reference_score:", args.reference_score)
    print("software_score:", args.software_score)
    print("tcs_threshold_order:", TCS_THRESHOLD_ORDER)
    print("tcs_active_rule:", TCS_ACTIVE_RULE)
    print("output_csv:", args.output_csv)
    print("Note: sparse_bitserial+TCS candidates only; dense/sparse references are not re-evaluated.")

    for name, vector in candidates:
        print(
            f"candidate {name}: lsb_to_msb={format_thresholds(vector)} "
            f"msb_to_lsb={format_thresholds_msb_to_lsb(vector)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(task_cfg["checkpoint"])
    dataset = tokenize_dataset(
        tokenizer=tokenizer,
        task_cfg=task_cfg,
        max_length=args.max_length,
        max_examples=args.max_examples,
    )
    input_stats = compute_input_length_stats(dataset, args.tile_n, args.max_length)
    print(
        "input length stats: "
        f"min/mean/max_nonpad={input_stats['min_nonpad_tokens']}/"
        f"{fmt_float(input_stats['mean_nonpad_tokens'])}/"
        f"{input_stats['max_nonpad_tokens']} "
        f"min/mean/max_real_blocks={input_stats['min_real_blocks']}/"
        f"{fmt_float(input_stats['mean_real_blocks'])}/"
        f"{input_stats['max_real_blocks']}"
    )

    loader = make_loader(dataset, args)
    first_name, first_vector = candidates[0]
    model = load_patched_sparse_model(task_cfg, args, enable_tcs=True, thresholds=first_vector)

    output_rows = []
    started_at = time.time()
    for index, (name, vector) in enumerate(candidates, start=1):
        print(
            f"\nRunning {name} ({index}/{len(candidates)}): "
            f"lsb_to_msb={format_thresholds(vector)} "
            f"msb_to_lsb={format_thresholds_msb_to_lsb(vector)}",
            flush=True,
        )
        set_tcs_thresholds(model, vector)
        metrics, stats, logits, predictions = evaluate_loaded_model(
            model,
            args.task,
            task_cfg,
            loader,
            args.device,
            progress_label=name,
            progress_every=args.batch_progress_every,
            no_progress=args.no_progress,
        )
        row = build_output_row(args, task_cfg, name, vector, metrics, stats)
        output_rows.append(row)
        elapsed = time.time() - started_at
        print(
            f"done {name}: score={fmt_float(row['score'])} "
            f"drop_vs_sparse={fmt_float(row['score_drop_vs_sparse_reference'])} "
            f"tcs_skip={fmt_float(row['tcs_row_skip_ratio'])} "
            f"reduction_bigbird={row['reduction_vs_bigbird_str']} "
            f"elapsed={format_duration(elapsed)}",
            flush=True,
        )

    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    write_csv(args.output_csv, output_rows)
    print_compact_table(output_rows)
    print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

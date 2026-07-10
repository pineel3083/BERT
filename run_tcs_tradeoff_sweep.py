import argparse
import csv
import os
import time
from types import SimpleNamespace

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
    tokenize_dataset,
)
from eval_glue_tcs_greedy import (
    evaluate_loaded_model,
    format_duration,
    format_thresholds,
    format_thresholds_msb_to_lsb,
    get_start_thresholds,
    load_patched_sparse_model,
    make_loader,
    make_summary_row,
    parse_candidate_thresholds,
    set_tcs_thresholds,
    validate_candidate,
    candidate_sort_key,
)
from tpdcim_bert_patch import TCS_ACTIVE_RULE, TCS_THRESHOLD_ORDER


TRACK_NAME = "score_constrained_tcs_tradeoff"
DEFAULT_ALLOWED_DROP_SWEEP = "0.000,0.005,0.010,0.020,0.030"


def parse_float_list(raw_value):
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("allowed-drop-sweep must contain at least one value")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run score-constrained TCS trade-off sweeps against the BigBird proxy."
    )
    parser.add_argument("--task", default="mnli", choices=sorted(TASKS))
    parser.add_argument("--max-examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=16)
    parser.add_argument("--local-window", type=int, default=1)
    parser.add_argument("--num-random-blocks", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allowed-drop-sweep", default=DEFAULT_ALLOWED_DROP_SWEEP)
    parser.add_argument("--threshold-min", type=int, default=0)
    parser.add_argument("--threshold-max", type=int, default=48)
    parser.add_argument("--threshold-step", type=int, default=4)
    parser.add_argument(
        "--candidate-thresholds",
        default=None,
        help="Optional comma-separated scalar threshold grid. Overrides min/max/step.",
    )
    parser.add_argument("--greedy-order", choices=["msb_to_lsb", "lsb_to_msb"], default="msb_to_lsb")
    parser.add_argument("--start-thresholds", choices=["zero", "bw_B", "bw_E"], default="zero")
    parser.add_argument("--start-thresholds-lsb-to-msb", default=None)
    parser.add_argument("--start-thresholds-msb-to-lsb", default=None)
    parser.add_argument("--max-changed-pred-ratio", type=float, default=None)
    parser.add_argument("--max-mean-logit-diff", type=float, default=None)
    parser.add_argument("--batch-progress-every", type=int, default=50)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def default_output_csv(args):
    examples = "full" if args.max_examples is None else str(args.max_examples)
    return os.path.join(
        "results",
        f"tcs_tradeoff_{args.task}_{examples}_len{args.max_length}_tile{args.tile_n}_random{args.num_random_blocks}.csv",
    )


def clone_args_with_allowed_drop(args, allowed_drop):
    cloned = SimpleNamespace(**vars(args))
    cloned.allowed_drop = float(allowed_drop)
    # eval_glue_tcs_greedy expects this field when printing optional tables.
    cloned.print_candidates = False
    return cloned


def evaluate_thresholds_shared(
    raw_cache,
    model,
    task_name,
    task_cfg,
    loader,
    args,
    thresholds,
    software_score,
    sparse_score,
    sparse_logits,
    sparse_predictions,
    label,
):
    key = tuple(int(value) for value in thresholds)
    if key not in raw_cache:
        if not args.no_progress:
            print(
                f"[tradeoff] evaluating {label} thresholds_lsb={format_thresholds(key)} "
                f"thresholds_msb={format_thresholds_msb_to_lsb(key)}",
                flush=True,
            )
        set_tcs_thresholds(model, key)
        metrics, stats, logits, predictions = evaluate_loaded_model(
            model=model,
            task_name=task_name,
            task_cfg=task_cfg,
            loader=loader,
            device=args.device,
            progress_label=label,
            progress_every=args.batch_progress_every,
            no_progress=args.no_progress,
        )
        row = make_summary_row(
            stage="tcs_candidate",
            task_name=task_name,
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
            thresholds=key,
            enable_tcs=True,
        )
        row["_thresholds_tuple"] = key
        raw_cache[key] = row

    row = dict(raw_cache[key])
    row["_thresholds_tuple"] = key
    return validate_candidate(row, args)


def run_cached_greedy(
    raw_cache,
    model,
    task_name,
    task_cfg,
    loader,
    args,
    software_score,
    sparse_score,
    sparse_logits,
    sparse_predictions,
):
    thresholds = get_start_thresholds(args)
    candidate_values = parse_candidate_thresholds(args)
    bit_order = list(reversed(range(8))) if args.greedy_order == "msb_to_lsb" else list(range(8))
    step_rows = []

    started_at = time.time()
    if not args.no_progress:
        print(
            f"\n[tradeoff] allowed_drop={args.allowed_drop:.6f} "
            f"candidate_thresholds={format_thresholds(candidate_values)}",
            flush=True,
        )

    current = evaluate_thresholds_shared(
        raw_cache,
        model,
        task_name,
        task_cfg,
        loader,
        args,
        thresholds,
        software_score,
        sparse_score,
        sparse_logits,
        sparse_predictions,
        label=f"drop={args.allowed_drop:.3f} initial",
    )

    for step_index, bit in enumerate(bit_order):
        trials = []
        values = sorted(set(candidate_values + [thresholds[bit]]))
        if not args.no_progress:
            elapsed = format_duration(time.time() - started_at)
            print(
                f"[tradeoff] drop={args.allowed_drop:.6f} step={step_index + 1}/{len(bit_order)} "
                f"bit={bit} current={thresholds[bit]} values={format_thresholds(values)} elapsed={elapsed}",
                flush=True,
            )

        for value in values:
            trial_thresholds = list(thresholds)
            trial_thresholds[bit] = value
            row = evaluate_thresholds_shared(
                raw_cache,
                model,
                task_name,
                task_cfg,
                loader,
                args,
                trial_thresholds,
                software_score,
                sparse_score,
                sparse_logits,
                sparse_predictions,
                label=f"drop={args.allowed_drop:.3f} step={step_index} bit={bit} threshold={value}",
            )
            trials.append(row)

        valid_trials = [row for row in trials if row["valid"]]
        if valid_trials:
            current = max(valid_trials, key=candidate_sort_key)
            thresholds = list(current["_thresholds_tuple"])
            note = "accepted"
        else:
            current = evaluate_thresholds_shared(
                raw_cache,
                model,
                task_name,
                task_cfg,
                loader,
                args,
                thresholds,
                software_score,
                sparse_score,
                sparse_logits,
                sparse_predictions,
                label=f"drop={args.allowed_drop:.3f} keep_current",
            )
            note = "no_valid_candidate_keep_current"

        step_rows.append(
            {
                "allowed_drop": args.allowed_drop,
                "step": step_index,
                "bit": bit,
                "chosen_threshold": thresholds[bit],
                "thresholds_lsb_to_msb": format_thresholds(thresholds),
                "thresholds_msb_to_lsb": format_thresholds_msb_to_lsb(thresholds),
                "valid_candidates": len(valid_trials),
                "note": note,
                "primary_score": current["primary_score"],
                "score_drop_vs_sparse_reference": current["score_drop_vs_sparse_reference"],
                "tcs_row_skip_ratio": current["tcs_row_skip_ratio"],
                "reduction_vs_bigbird_str": current["reduction_vs_bigbird_str"],
            }
        )

    final_row = evaluate_thresholds_shared(
        raw_cache,
        model,
        task_name,
        task_cfg,
        loader,
        args,
        thresholds,
        software_score,
        sparse_score,
        sparse_logits,
        sparse_predictions,
        label=f"drop={args.allowed_drop:.3f} final",
    )
    return final_row, step_rows


def make_output_row(args, task_cfg, allowed_drop, software_score, sparse_score, final_row):
    return {
        "track": TRACK_NAME,
        "task": args.task,
        "primary_metric": task_cfg["primary_metric"],
        "max_examples": "full" if args.max_examples is None else args.max_examples,
        "max_length": args.max_length,
        "tile_n": args.tile_n,
        "num_blocks": ceil_div(args.max_length, args.tile_n),
        "num_random_blocks": args.num_random_blocks,
        "allowed_drop": allowed_drop,
        "threshold_step": args.threshold_step,
        "thresholds_lsb_to_msb": final_row["thresholds_lsb_to_msb"],
        "thresholds_msb_to_lsb": final_row["thresholds_msb_to_lsb"],
        "software_score": software_score,
        "sparse_reference_score": sparse_score,
        "tcs_score": final_row["primary_score"],
        "score_drop_vs_sparse_reference": final_row["score_drop_vs_sparse_reference"],
        "score_delta_vs_software_baseline": final_row["score_delta_vs_software_baseline"],
        "qk_sparse_density": final_row["qk_sparse_density"],
        "qk_tile_skip_ratio": final_row["qk_tile_skip_ratio"],
        "tcs_row_skip_ratio": final_row["tcs_row_skip_ratio"],
        "reduction_vs_dense_str": final_row["reduction_vs_dense_str"],
        "reduction_vs_bigbird_str": final_row["reduction_vs_bigbird_str"],
        "operation_saving_vs_dense": final_row["operation_saving_vs_dense"],
        "operation_saving_vs_bigbird": final_row["operation_saving_vs_bigbird"],
        "num_changed_predictions_vs_sparse_reference": final_row[
            "num_changed_predictions_vs_sparse_reference"
        ],
        "changed_prediction_ratio_vs_sparse_reference": final_row[
            "changed_prediction_ratio_vs_sparse_reference"
        ],
        "mean_logit_diff_vs_sparse_reference": final_row[
            "mean_logit_diff_vs_sparse_reference"
        ],
        "max_logit_diff_vs_sparse_reference": final_row[
            "max_logit_diff_vs_sparse_reference"
        ],
    }


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def print_compact_table(rows):
    headers = [
        "allowed_drop",
        "score",
        "drop_vs_sparse",
        "tcs_row_skip",
        "reduction_vs_bigbird",
        "thresholds_lsb_to_msb",
    ]
    print("\nCOMPACT SCORE-CONSTRAINED TCS TRADE-OFF")
    print("Note: BigBird proxy and reduction_vs_bigbird proxy; not exact paper reproduction.")
    print(" | ".join(headers))
    print(" | ".join("-" * len(header) for header in headers))
    for row in rows:
        print(
            " | ".join(
                [
                    fmt_float(row["allowed_drop"], 3),
                    fmt_float(row["tcs_score"], 6),
                    fmt_float(row["score_drop_vs_sparse_reference"], 6),
                    fmt_float(row["tcs_row_skip_ratio"], 6),
                    row["reduction_vs_bigbird_str"],
                    row["thresholds_lsb_to_msb"],
                ]
            )
        )


def main():
    args = parse_args()
    output_csv = args.output_csv or default_output_csv(args)
    allowed_drops = parse_float_list(args.allowed_drop_sweep)
    task_cfg = TASKS[args.task]

    print("track:", TRACK_NAME)
    print("task:", args.task)
    print("checkpoint:", task_cfg["checkpoint"])
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
    print("allowed_drop_sweep:", ",".join(f"{value:.3f}" for value in allowed_drops))
    print("output_csv:", output_csv)
    print("Note: QK activity proxy, BigBird proxy, and reduction_vs_bigbird proxy.")
    print("Note: not exact TP-DCIM paper reproduction, measured CUDA speedup, energy, or cycle-accurate simulation.")

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

    print(
        "reference scores: "
        f"software_{task_cfg['primary_metric']}={fmt_float(software_score)} "
        f"sparse_{task_cfg['primary_metric']}={fmt_float(sparse_score)}",
        flush=True,
    )

    search_args = clone_args_with_allowed_drop(args, allowed_drops[0])
    search_model = load_patched_sparse_model(
        task_cfg,
        search_args,
        enable_tcs=True,
        thresholds=get_start_thresholds(search_args),
    )
    raw_cache = {}
    output_rows = []

    for allowed_drop in allowed_drops:
        current_args = clone_args_with_allowed_drop(args, allowed_drop)
        final_row, _step_rows = run_cached_greedy(
            raw_cache=raw_cache,
            model=search_model,
            task_name=args.task,
            task_cfg=task_cfg,
            loader=loader,
            args=current_args,
            software_score=software_score,
            sparse_score=sparse_score,
            sparse_logits=sparse_logits,
            sparse_predictions=sparse_predictions,
        )
        output_rows.append(
            make_output_row(
                current_args,
                task_cfg,
                allowed_drop,
                software_score,
                sparse_score,
                final_row,
            )
        )

    del search_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    write_csv(output_csv, output_rows)
    print_compact_table(output_rows)
    print(f"\nSaved CSV: {output_csv}")


if __name__ == "__main__":
    main()

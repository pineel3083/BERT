import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from eval_glue_tpdcim import (
    BW_B,
    BW_E,
    GLUE_DATASET_REPO,
    TASKS,
    EvalCase,
    ceil_div,
    collate_batch,
    collect_batch_stats,
    compute_activity_metrics,
    compute_input_length_stats,
    compute_logit_diagnostics,
    compute_metrics,
    compute_sparse_real_token_stats,
    empty_stats,
    finalize_stats,
    fmt_cell,
    fmt_float,
    get_num_layers,
    load_model,
    merge_stats,
    tokenize_dataset,
    apply_prediction_label_map,
)
from tpdcim_bert_patch import (
    TCS_ACTIVE_RULE,
    TCS_THRESHOLD_ORDER,
    enable_qk_tiling,
)


TRACK_NAME = "task_level_tcs_threshold_greedy"
START_THRESHOLDS = {
    "zero": [0] * 8,
    "bw_B": BW_B,
    "bw_E": BW_E,
}

PRESET_HELP = """
Recommended first passes:

A. Paper-wise BERT classifier scaling, 8 sequence blocks:
   python eval_glue_tcs_greedy.py --task mnli --max-examples 128 --batch-size 4 --max-length 128 --tile-n 16 --threshold-step 4
   Then rerun the promising setting with --max-examples 512 and --threshold-step 2.

B. Full MRPC validation with the same 8-block classifier scaling:
   python eval_glue_tcs_greedy.py --task mrpc --batch-size 8 --max-length 128 --tile-n 16 --threshold-step 2

C. Padded 8-block diagnostic, closer to the paper's 8 Q/K blocks but mostly padding on GLUE:
   python eval_glue_tcs_greedy.py --task mnli --max-examples 512 --batch-size 4 --max-length 512 --tile-n 64 --threshold-step 4

Interpretation:
- The greedy reference is sparse_bitserial, i.e. the BigBird-like proxy before TCS.
- The objective is not to hit a target x0.71 ratio. It maximizes TCS row skipping while keeping task score drop within --allowed-drop.
- Thresholds are printed in the simulator's current LSB-to-MSB convention.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Greedy-search TPDCIM TCS thresholds with real GLUE task score constraints.",
        epilog=PRESET_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", default="mnli", choices=sorted(TASKS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=16)
    parser.add_argument("--local-window", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start-thresholds", choices=sorted(START_THRESHOLDS), default="zero")
    parser.add_argument("--greedy-order", choices=["msb_to_lsb", "lsb_to_msb"], default="msb_to_lsb")
    parser.add_argument("--threshold-min", type=int, default=0)
    parser.add_argument("--threshold-max", type=int, default=32)
    parser.add_argument("--threshold-step", type=int, default=2)
    parser.add_argument(
        "--candidate-thresholds",
        default=None,
        help="Optional comma-separated threshold list, e.g. 0,4,8,12,16,20,24,28,32.",
    )
    parser.add_argument(
        "--allowed-drop",
        type=float,
        default=0.005,
        help="Maximum allowed primary-metric drop vs sparse_bitserial reference.",
    )
    parser.add_argument(
        "--max-changed-pred-ratio",
        type=float,
        default=None,
        help="Optional extra constraint vs sparse_bitserial reference predictions.",
    )
    parser.add_argument(
        "--max-mean-logit-diff",
        type=float,
        default=None,
        help="Optional extra constraint on mean absolute logit diff vs sparse_bitserial reference.",
    )
    parser.add_argument(
        "--print-candidates",
        action="store_true",
        help="Print every candidate tried at each greedy step.",
    )
    return parser.parse_args()


def format_thresholds(thresholds):
    if thresholds is None:
        return "-"
    return "[" + ",".join(str(int(v)) for v in thresholds) + "]"


def parse_candidate_thresholds(args):
    if args.candidate_thresholds:
        values = []
        for item in args.candidate_thresholds.split(","):
            item = item.strip()
            if item:
                values.append(int(item))
    else:
        values = list(range(args.threshold_min, args.threshold_max + 1, args.threshold_step))

    if not values:
        raise ValueError("No candidate thresholds were provided.")
    if min(values) < 0:
        raise ValueError("TCS thresholds must be non-negative.")
    return sorted(set(values))


def get_attention_modules(model):
    if hasattr(model, "bert"):
        layers = model.bert.encoder.layer
    else:
        layers = model.encoder.layer
    return [layer.attention.self for layer in layers]


def set_tcs_thresholds(model, thresholds):
    """Update all patched BERT attention modules without reloading the model."""
    clean_thresholds = [int(v) for v in thresholds]
    if len(clean_thresholds) != 8:
        raise ValueError("TCS thresholds must contain 8 values in LSB-to-MSB order.")
    for attn in get_attention_modules(model):
        attn.tpdcim_enable_tcs = True
        attn.tpdcim_tcs_thresholds = list(clean_thresholds)
        attn.tpdcim_last_stats = None


def make_loader(dataset, args):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )


def evaluate_loaded_model(model, task_name, task_cfg, loader, device):
    preds = []
    labels = []
    logits_chunks = []
    total_stats = empty_stats()

    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels_batch = batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
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
    predictions = torch.tensor(preds, dtype=torch.long)
    return metrics, stats, logits, predictions


def load_patched_sparse_model(task_cfg, args, enable_tcs, thresholds=None):
    model = load_model(task_cfg["checkpoint"], args.device)
    enable_qk_tiling(
        model,
        tile_n=args.tile_n,
        enable_sparse=True,
        local_window=args.local_window,
        global_blocks=(0,),
        num_random_blocks=0,
        qk_mode="bitserial",
        enable_tcs=enable_tcs,
        tcs_thresholds=thresholds,
    )
    return model


def diagnostics_vs_sparse(logits, predictions, sparse_logits, sparse_predictions):
    diagnostics = compute_logit_diagnostics(
        logits=logits,
        predictions=predictions,
        baseline_logits=sparse_logits,
        baseline_predictions=sparse_predictions,
    )
    return {
        "max_logit_diff_vs_sparse_reference": diagnostics["max_logit_diff_vs_baseline"],
        "mean_logit_diff_vs_sparse_reference": diagnostics["mean_logit_diff_vs_baseline"],
        "num_changed_predictions_vs_sparse_reference": diagnostics[
            "num_changed_predictions_vs_baseline"
        ],
        "changed_prediction_ratio_vs_sparse_reference": diagnostics[
            "changed_prediction_ratio_vs_baseline"
        ],
    }


def make_summary_row(
    stage,
    task_name,
    task_cfg,
    args,
    metrics,
    stats,
    logits,
    predictions,
    software_score,
    sparse_score,
    sparse_logits,
    sparse_predictions,
    thresholds=None,
    enable_tcs=False,
):
    primary_metric = task_cfg["primary_metric"]
    primary_score = metrics[primary_metric]
    case = EvalCase(
        stage,
        enable_patch=(stage != "software_baseline"),
        enable_sparse=(stage != "software_baseline"),
        qk_mode="bitserial" if stage != "software_baseline" else "fp32",
        enable_tcs=enable_tcs,
        tcs_thresholds=list(thresholds) if thresholds is not None else None,
    )
    activity = compute_activity_metrics(case, stats)
    row = {
        "track": TRACK_NAME,
        "task": task_name,
        "model_name": task_cfg["checkpoint"],
        "split": task_cfg["split"],
        "max_length": args.max_length,
        "tile_n": args.tile_n,
        "num_blocks": ceil_div(args.max_length, args.tile_n),
        "stage": stage,
        "thresholds_lsb_to_msb": format_thresholds(thresholds),
        "primary_metric": primary_metric,
        "primary_score": primary_score,
        "accuracy": metrics["accuracy"],
        "f1": metrics.get("f1"),
        "score_delta_vs_software_baseline": primary_score - software_score,
        "score_delta_vs_sparse_reference": primary_score - sparse_score,
        "score_drop_vs_sparse_reference": sparse_score - primary_score,
        "qk_mode": stats["qk_mode"],
        "qk_computed_tiles": stats["qk_computed_tiles"],
        "qk_skipped_tiles": stats["qk_skipped_tiles"],
        "qk_sparse_density": stats["qk_sparse_density"],
        "qk_tile_skip_ratio": stats["qk_tile_skip_ratio"],
        "bit_ops_total": stats["bit_ops_total"],
        "tcs_rows_total": stats["tcs_rows_total"],
        "tcs_rows_skipped": stats["tcs_rows_skipped"],
        "tcs_row_skip_ratio": stats["tcs_row_skip_ratio"],
    }
    row.update(activity)
    row.update(diagnostics_vs_sparse(logits, predictions, sparse_logits, sparse_predictions))
    return row


def validate_candidate(row, args):
    reasons = []
    if row["score_drop_vs_sparse_reference"] > args.allowed_drop:
        reasons.append("score_drop")

    changed_ratio = row["changed_prediction_ratio_vs_sparse_reference"]
    if args.max_changed_pred_ratio is not None and changed_ratio is not None:
        if changed_ratio > args.max_changed_pred_ratio:
            reasons.append("changed_pred_ratio")

    mean_logit_diff = row["mean_logit_diff_vs_sparse_reference"]
    if args.max_mean_logit_diff is not None and mean_logit_diff is not None:
        if mean_logit_diff > args.max_mean_logit_diff:
            reasons.append("mean_logit_diff")

    row["valid"] = not reasons
    row["rejection_reason"] = ",".join(reasons) if reasons else "-"
    return row


def candidate_sort_key(row):
    # Primary reward: more TCS row skipping. Ties prefer better task score and smaller logit drift.
    mean_diff = row["mean_logit_diff_vs_sparse_reference"]
    mean_diff = 0.0 if mean_diff is None else mean_diff
    return (
        row["tcs_row_skip_ratio"],
        row["primary_score"],
        -mean_diff,
        -row["score_drop_vs_sparse_reference"],
    )


def evaluate_thresholds_cached(
    cache,
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
):
    key = tuple(int(v) for v in thresholds)
    if key in cache:
        return cache[key]

    set_tcs_thresholds(model, key)
    metrics, stats, logits, predictions = evaluate_loaded_model(
        model=model,
        task_name=task_name,
        task_cfg=task_cfg,
        loader=loader,
        device=args.device,
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
    cache[key] = validate_candidate(row, args)
    return cache[key]


def run_greedy_search(
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
    thresholds = list(START_THRESHOLDS[args.start_thresholds])
    candidate_values = parse_candidate_thresholds(args)
    bit_order = list(reversed(range(8))) if args.greedy_order == "msb_to_lsb" else list(range(8))
    cache = {}
    step_rows = []
    candidate_rows = []

    initial_row = evaluate_thresholds_cached(
        cache,
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
    )

    for step_index, bit in enumerate(bit_order):
        trials = []
        values = sorted(set(candidate_values + [thresholds[bit]]))
        for value in values:
            trial_thresholds = list(thresholds)
            trial_thresholds[bit] = value
            row = evaluate_thresholds_cached(
                cache,
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
            )
            display_row = dict(row)
            display_row["step"] = step_index
            display_row["bit"] = bit
            display_row["trial_threshold"] = value
            trials.append(row)
            candidate_rows.append(display_row)

        valid_trials = [row for row in trials if row["valid"]]
        if valid_trials:
            chosen = max(valid_trials, key=candidate_sort_key)
            thresholds = list(chosen["_thresholds_tuple"])
            note = "accepted"
        else:
            chosen = evaluate_thresholds_cached(
                cache,
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
            )
            note = "no_valid_candidate_keep_current"

        step_rows.append(
            {
                "step": step_index,
                "bit": bit,
                "chosen_threshold": thresholds[bit],
                "thresholds_lsb_to_msb": format_thresholds(thresholds),
                "valid_candidates": len(valid_trials),
                "note": note,
                "primary_score": chosen["primary_score"],
                "score_drop_vs_sparse_reference": chosen["score_drop_vs_sparse_reference"],
                "tcs_row_skip_ratio": chosen["tcs_row_skip_ratio"],
                "reduction_vs_dense_str": chosen["reduction_vs_dense_str"],
                "reduction_vs_bigbird_str": chosen["reduction_vs_bigbird_str"],
                "num_changed_predictions_vs_sparse_reference": chosen[
                    "num_changed_predictions_vs_sparse_reference"
                ],
                "changed_prediction_ratio_vs_sparse_reference": chosen[
                    "changed_prediction_ratio_vs_sparse_reference"
                ],
                "mean_logit_diff_vs_sparse_reference": chosen[
                    "mean_logit_diff_vs_sparse_reference"
                ],
            }
        )

    final_row = evaluate_thresholds_cached(
        cache,
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
    )
    return initial_row, final_row, step_rows, candidate_rows


def print_table(title, headers, rows):
    print("\n" + title)
    print(" | ".join(headers))
    print(" | ".join("-" * len(header) for header in headers))
    for row in rows:
        print(" | ".join(fmt_cell(row.get(header)) for header in headers))


def print_summary_table(rows):
    headers = [
        "stage",
        "thresholds_lsb_to_msb",
        "primary_metric",
        "primary_score",
        "accuracy",
        "f1",
        "score_delta_vs_software_baseline",
        "score_delta_vs_sparse_reference",
        "score_drop_vs_sparse_reference",
        "qk_sparse_density",
        "qk_tile_skip_ratio",
        "tcs_row_skip_ratio",
        "reduction_vs_dense_str",
        "reduction_vs_bigbird_str",
        "operation_saving_vs_dense",
        "operation_saving_vs_bigbird",
        "num_changed_predictions_vs_sparse_reference",
        "changed_prediction_ratio_vs_sparse_reference",
        "mean_logit_diff_vs_sparse_reference",
        "max_logit_diff_vs_sparse_reference",
    ]
    print_table("FINAL TCS CALIBRATION SUMMARY", headers, rows)


def print_step_table(rows):
    headers = [
        "step",
        "bit",
        "chosen_threshold",
        "thresholds_lsb_to_msb",
        "valid_candidates",
        "note",
        "primary_score",
        "score_drop_vs_sparse_reference",
        "tcs_row_skip_ratio",
        "reduction_vs_dense_str",
        "reduction_vs_bigbird_str",
        "num_changed_predictions_vs_sparse_reference",
        "changed_prediction_ratio_vs_sparse_reference",
        "mean_logit_diff_vs_sparse_reference",
    ]
    print_table("GREEDY SEARCH STEPS", headers, rows)


def print_candidate_table(rows):
    headers = [
        "step",
        "bit",
        "trial_threshold",
        "thresholds_lsb_to_msb",
        "valid",
        "rejection_reason",
        "primary_score",
        "score_drop_vs_sparse_reference",
        "tcs_row_skip_ratio",
        "reduction_vs_bigbird_str",
        "changed_prediction_ratio_vs_sparse_reference",
        "mean_logit_diff_vs_sparse_reference",
    ]
    print_table("GREEDY CANDIDATES", headers, rows)


def main():
    args = parse_args()
    task_cfg = TASKS[args.task]
    num_blocks = ceil_div(args.max_length, args.tile_n)

    print("track:", TRACK_NAME)
    print("task:", args.task)
    print("checkpoint:", task_cfg["checkpoint"])
    print("split:", task_cfg["split"])
    print("dataset_repo:", GLUE_DATASET_REPO)
    print("device:", args.device)
    print("batch_size:", args.batch_size)
    print("max_examples:", args.max_examples)
    print("max_length:", args.max_length)
    print("tile_n:", args.tile_n)
    print("num_blocks:", num_blocks)
    print("local_window:", args.local_window)
    print("tcs_threshold_order:", TCS_THRESHOLD_ORDER)
    print("tcs_active_rule:", TCS_ACTIVE_RULE)
    print("greedy_order:", args.greedy_order)
    print("start_thresholds:", args.start_thresholds, format_thresholds(START_THRESHOLDS[args.start_thresholds]))
    print("candidate_thresholds:", format_thresholds(parse_candidate_thresholds(args)))
    print("allowed_drop:", args.allowed_drop)
    print("max_changed_pred_ratio:", args.max_changed_pred_ratio)
    print("max_mean_logit_diff:", args.max_mean_logit_diff)
    print("Note: sparse_bitserial is the greedy reference; software baseline is reported for context.")
    print("Note: this is task-level trend calibration, not exact TP-DCIM paper reproduction.")

    if args.max_length % args.tile_n != 0:
        print("Warning: max_length is not divisible by tile_n; the last sequence block is partial.")

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
        f"min/mean/max_nonpad="
        f"{input_stats['min_nonpad_tokens']}/"
        f"{fmt_float(input_stats['mean_nonpad_tokens'])}/"
        f"{input_stats['max_nonpad_tokens']} "
        f"min/mean/max_real_blocks="
        f"{input_stats['min_real_blocks']}/"
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

    print("\nRunning software baseline ...")
    software_model = load_model(task_cfg["checkpoint"], args.device)
    software_metrics, software_stats, software_logits, software_predictions = evaluate_loaded_model(
        software_model,
        args.task,
        task_cfg,
        loader,
        args.device,
    )
    software_score = software_metrics[task_cfg["primary_metric"]]
    del software_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("Running sparse_bitserial reference ...")
    sparse_model = load_patched_sparse_model(task_cfg, args, enable_tcs=False)
    sparse_metrics, sparse_stats, sparse_logits, sparse_predictions = evaluate_loaded_model(
        sparse_model,
        args.task,
        task_cfg,
        loader,
        args.device,
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

    print(
        "reference scores: "
        f"software_{task_cfg['primary_metric']}={fmt_float(software_score)} "
        f"sparse_{task_cfg['primary_metric']}={fmt_float(sparse_score)}"
    )

    print("Running GLUE-constrained TCS greedy search ...")
    search_model = load_patched_sparse_model(
        task_cfg,
        args,
        enable_tcs=True,
        thresholds=START_THRESHOLDS[args.start_thresholds],
    )
    initial_row, final_row, step_rows, candidate_rows = run_greedy_search(
        model=search_model,
        task_name=args.task,
        task_cfg=task_cfg,
        loader=loader,
        args=args,
        software_score=software_score,
        sparse_score=sparse_score,
        sparse_logits=sparse_logits,
        sparse_predictions=sparse_predictions,
    )
    del search_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    initial_row = dict(initial_row)
    initial_row["stage"] = "tcs_initial"
    final_row = dict(final_row)
    final_row["stage"] = "tcs_greedy_final"

    print_step_table(step_rows)
    print_summary_table([software_row, sparse_row, initial_row, final_row])
    if args.print_candidates:
        print_candidate_table(candidate_rows)


if __name__ == "__main__":
    main()

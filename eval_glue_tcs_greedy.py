import argparse
import time

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
Recommended runs:

A. Quick sanity run with a coarse candidate grid:
   python eval_glue_tcs_greedy.py --task mnli --max-examples 128 --batch-size 4 --max-length 128 --tile-n 16 --candidate-thresholds 0,8,16,24,36,48

B. Full MNLI calibration with the same candidate grid:
   python eval_glue_tcs_greedy.py --task mnli --batch-size 4 --max-length 128 --tile-n 16 --candidate-thresholds 0,8,16,24,36,48

C. Evaluate a paper-reported threshold vector without using it as a candidate grid:
   python eval_glue_tcs_greedy.py --task mnli --batch-size 4 --max-length 128 --tile-n 16 --fixed-thresholds-msb-to-lsb <8 comma-separated values> --skip-greedy

Interpretation:
- The greedy reference is sparse_bitserial, i.e. the BigBird-like proxy before TCS.
- Candidate thresholds are scalar values tried independently for each bit.
- Fixed/start threshold vectors must contain 8 values.
- Paper vectors reported as MSB-to-LSB are reversed internally because the simulator stores thresholds as LSB-to-MSB.
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
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap for quick calibration. Omit for the full validation split.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start-thresholds", choices=sorted(START_THRESHOLDS), default="zero")
    parser.add_argument(
        "--start-thresholds-lsb-to-msb",
        default=None,
        help="Custom 8-value starting vector in simulator order. Overrides --start-thresholds.",
    )
    parser.add_argument(
        "--start-thresholds-msb-to-lsb",
        default=None,
        help="Custom 8-value starting vector in paper order. Reversed internally.",
    )
    parser.add_argument(
        "--fixed-thresholds-lsb-to-msb",
        default=None,
        help="Evaluate one fixed 8-value vector in simulator order before optional greedy search.",
    )
    parser.add_argument(
        "--fixed-thresholds-msb-to-lsb",
        default=None,
        help="Evaluate one fixed 8-value vector in paper order before optional greedy search.",
    )
    parser.add_argument(
        "--skip-greedy",
        action="store_true",
        help="Only evaluate baseline/sparse/fixed thresholds; do not run greedy search.",
    )
    parser.add_argument("--greedy-order", choices=["msb_to_lsb", "lsb_to_msb"], default="msb_to_lsb")
    parser.add_argument("--threshold-min", type=int, default=0)
    parser.add_argument("--threshold-max", type=int, default=48)
    parser.add_argument("--threshold-step", type=int, default=4)
    parser.add_argument(
        "--candidate-thresholds",
        default=None,
        help="Optional comma-separated scalar candidate list, e.g. 0,8,16,24,36,48. These are not an 8-bit vector.",
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
        "--progress-every",
        type=int,
        default=1,
        help="Print one progress line every N uncached candidate evaluations.",
    )
    parser.add_argument(
        "--batch-progress-every",
        type=int,
        default=50,
        help="Print one batch progress line every N batches inside each model evaluation. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable live greedy-search and batch progress logging.",
    )
    parser.add_argument(
        "--print-candidates",
        action="store_true",
        help="Print every candidate tried at each greedy step.",
    )
    return parser.parse_args()


def parse_threshold_vector(raw_value, name):
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if len(values) != 8:
        raise ValueError(f"{name} must contain exactly 8 comma-separated integers")
    if min(values) < 0:
        raise ValueError(f"{name} must contain non-negative thresholds")
    return values


def thresholds_from_args(lsb_to_msb, msb_to_lsb, arg_name):
    if lsb_to_msb is not None and msb_to_lsb is not None:
        raise ValueError(f"Pass only one of {arg_name}-lsb-to-msb or {arg_name}-msb-to-lsb")
    if lsb_to_msb is not None:
        return parse_threshold_vector(lsb_to_msb, f"{arg_name}-lsb-to-msb")
    if msb_to_lsb is not None:
        return list(reversed(parse_threshold_vector(msb_to_lsb, f"{arg_name}-msb-to-lsb")))
    return None


def get_start_thresholds(args):
    custom = thresholds_from_args(
        args.start_thresholds_lsb_to_msb,
        args.start_thresholds_msb_to_lsb,
        "--start-thresholds",
    )
    if custom is not None:
        return custom
    return list(START_THRESHOLDS[args.start_thresholds])


def get_fixed_thresholds(args):
    return thresholds_from_args(
        args.fixed_thresholds_lsb_to_msb,
        args.fixed_thresholds_msb_to_lsb,
        "--fixed-thresholds",
    )


def format_thresholds(thresholds):
    if thresholds is None:
        return "-"
    return "[" + ",".join(str(int(v)) for v in thresholds) + "]"


def format_thresholds_msb_to_lsb(thresholds):
    if thresholds is None:
        return "-"
    return format_thresholds(list(reversed(thresholds)))


def format_duration(seconds):
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def progress_line(args, message):
    if not args.no_progress:
        print(message, flush=True)


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
        raise ValueError("TCS candidate thresholds must be non-negative.")
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


def maybe_print_batch_progress(
    progress_label,
    batch_index,
    total_batches,
    examples_done,
    total_examples,
    started_at,
    progress_every,
    no_progress,
):
    if no_progress or progress_label is None or progress_every <= 0:
        return

    should_print = (
        batch_index == 1
        or batch_index == total_batches
        or batch_index % progress_every == 0
    )
    if not should_print:
        return

    elapsed = time.time() - started_at
    avg_per_batch = elapsed / max(1, batch_index)
    eta = avg_per_batch * max(0, total_batches - batch_index)
    examples_per_second = examples_done / elapsed if elapsed > 0 else 0.0
    print(
        f"[{progress_label}] batch {batch_index}/{total_batches} "
        f"examples={examples_done}/{total_examples} "
        f"elapsed={format_duration(elapsed)} eta~{format_duration(eta)} "
        f"ex/s={examples_per_second:.2f}",
        flush=True,
    )


def evaluate_loaded_model(
    model,
    task_name,
    task_cfg,
    loader,
    device,
    progress_label=None,
    progress_every=0,
    no_progress=False,
):
    preds = []
    labels = []
    logits_chunks = []
    total_stats = empty_stats()
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
            logits = outputs.logits.detach().cpu()
            batch_preds = logits.argmax(dim=-1)
            batch_preds = apply_prediction_label_map(batch_preds, task_cfg)

            preds.extend(batch_preds.tolist())
            labels.extend(labels_batch.tolist())
            logits_chunks.append(logits)
            merge_stats(total_stats, collect_batch_stats(model))
            maybe_print_batch_progress(
                progress_label=progress_label,
                batch_index=batch_index,
                total_batches=total_batches,
                examples_done=examples_done,
                total_examples=total_examples,
                started_at=started_at,
                progress_every=progress_every,
                no_progress=no_progress,
            )

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
        "thresholds_msb_to_lsb": format_thresholds_msb_to_lsb(thresholds),
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


def maybe_print_candidate_start(args, progress, label, key):
    if progress is None or args.no_progress:
        return None

    progress["started"] += 1
    candidate_index = progress["started"]
    should_print = (
        args.progress_every <= 1
        or candidate_index == 1
        or candidate_index % args.progress_every == 0
    )
    if should_print:
        elapsed = time.time() - progress["started_at"]
        print(
            f"[candidate {candidate_index}/~{progress['approx_total']}] start "
            f"{label} thresholds_lsb={format_thresholds(key)} "
            f"thresholds_msb={format_thresholds_msb_to_lsb(key)} "
            f"elapsed={format_duration(elapsed)}",
            flush=True,
        )
    return should_print


def maybe_print_candidate_done(args, progress, row, should_print):
    if progress is None or args.no_progress:
        return

    progress["finished"] += 1
    if not should_print:
        return

    elapsed = time.time() - progress["started_at"]
    avg = elapsed / max(1, progress["finished"])
    remaining = max(0, progress["approx_total"] - progress["finished"])
    eta = avg * remaining
    print(
        f"[candidate {progress['finished']}/~{progress['approx_total']}] done "
        f"score={fmt_float(row['primary_score'])} "
        f"drop_vs_sparse={fmt_float(row['score_drop_vs_sparse_reference'])} "
        f"tcs_skip={fmt_float(row['tcs_row_skip_ratio'])} "
        f"valid={fmt_cell(row['valid'])} "
        f"reason={row['rejection_reason']} "
        f"elapsed={format_duration(elapsed)} eta~{format_duration(eta)}",
        flush=True,
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
    progress=None,
    progress_label="candidate",
):
    key = tuple(int(v) for v in thresholds)
    if key in cache:
        return cache[key]

    should_print = maybe_print_candidate_start(args, progress, progress_label, key)
    set_tcs_thresholds(model, key)
    metrics, stats, logits, predictions = evaluate_loaded_model(
        model=model,
        task_name=task_name,
        task_cfg=task_cfg,
        loader=loader,
        device=args.device,
        progress_label=progress_label,
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
    cache[key] = validate_candidate(row, args)
    maybe_print_candidate_done(args, progress, cache[key], should_print)
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
    thresholds = get_start_thresholds(args)
    candidate_values = parse_candidate_thresholds(args)
    bit_order = list(reversed(range(8))) if args.greedy_order == "msb_to_lsb" else list(range(8))
    approx_total = 1 + len(bit_order) * len(candidate_values)
    progress = {
        "approx_total": approx_total,
        "started": 0,
        "finished": 0,
        "started_at": time.time(),
    }
    cache = {}
    step_rows = []
    candidate_rows = []

    progress_line(
        args,
        f"Greedy search will run up to about {approx_total} uncached TCS candidate forward passes.",
    )
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
        progress=progress,
        progress_label="initial",
    )

    for step_index, bit in enumerate(bit_order):
        trials = []
        values = sorted(set(candidate_values + [thresholds[bit]]))
        progress_line(
            args,
            f"[step {step_index + 1}/{len(bit_order)}] bit={bit} "
            f"current_threshold={thresholds[bit]} trying={format_thresholds(values)}",
        )
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
                progress=progress,
                progress_label=f"step={step_index} bit={bit} trial_threshold={value}",
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
                progress=progress,
                progress_label=f"step={step_index} bit={bit} keep_current",
            )
            note = "no_valid_candidate_keep_current"

        progress_line(
            args,
            f"[step {step_index + 1}/{len(bit_order)}] chosen_threshold={thresholds[bit]} "
            f"valid_candidates={len(valid_trials)} score={fmt_float(chosen['primary_score'])} "
            f"drop_vs_sparse={fmt_float(chosen['score_drop_vs_sparse_reference'])} "
            f"tcs_skip={fmt_float(chosen['tcs_row_skip_ratio'])} "
            f"thresholds_lsb={format_thresholds(thresholds)} "
            f"thresholds_msb={format_thresholds_msb_to_lsb(thresholds)} note={note}",
        )
        step_rows.append(
            {
                "step": step_index,
                "bit": bit,
                "chosen_threshold": thresholds[bit],
                "thresholds_lsb_to_msb": format_thresholds(thresholds),
                "thresholds_msb_to_lsb": format_thresholds_msb_to_lsb(thresholds),
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
        progress=progress,
        progress_label="final",
    )
    return initial_row, final_row, step_rows, candidate_rows


def evaluate_fixed_thresholds(task_name, task_cfg, loader, args, software_score, sparse_score, sparse_logits, sparse_predictions, fixed_thresholds):
    print(
        "Running fixed TCS vector: "
        f"lsb_to_msb={format_thresholds(fixed_thresholds)} "
        f"msb_to_lsb={format_thresholds_msb_to_lsb(fixed_thresholds)}",
        flush=True,
    )
    model = load_patched_sparse_model(task_cfg, args, enable_tcs=True, thresholds=fixed_thresholds)
    metrics, stats, logits, predictions = evaluate_loaded_model(
        model,
        task_name,
        task_cfg,
        loader,
        args.device,
        progress_label="tcs_fixed_reference",
        progress_every=args.batch_progress_every,
        no_progress=args.no_progress,
    )
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return make_summary_row(
        stage="tcs_fixed_reference",
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
        thresholds=fixed_thresholds,
        enable_tcs=True,
    )


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
        "thresholds_msb_to_lsb",
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
    if not rows:
        return
    headers = [
        "step",
        "bit",
        "chosen_threshold",
        "thresholds_lsb_to_msb",
        "thresholds_msb_to_lsb",
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
        "thresholds_msb_to_lsb",
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
    candidate_thresholds = parse_candidate_thresholds(args)
    start_thresholds = get_start_thresholds(args)
    fixed_thresholds = get_fixed_thresholds(args)

    print("track:", TRACK_NAME)
    print("task:", args.task)
    print("checkpoint:", task_cfg["checkpoint"])
    print("split:", task_cfg["split"])
    print("dataset_repo:", GLUE_DATASET_REPO)
    print("device:", args.device)
    print("batch_size:", args.batch_size)
    print("max_examples:", "full" if args.max_examples is None else args.max_examples)
    print("max_length:", args.max_length)
    print("tile_n:", args.tile_n)
    print("num_blocks:", num_blocks)
    print("local_window:", args.local_window)
    print("tcs_threshold_order:", TCS_THRESHOLD_ORDER)
    print("tcs_active_rule:", TCS_ACTIVE_RULE)
    print("greedy_order:", args.greedy_order)
    print("start_thresholds_lsb_to_msb:", format_thresholds(start_thresholds))
    print("start_thresholds_msb_to_lsb:", format_thresholds_msb_to_lsb(start_thresholds))
    print("fixed_thresholds_lsb_to_msb:", format_thresholds(fixed_thresholds))
    print("fixed_thresholds_msb_to_lsb:", format_thresholds_msb_to_lsb(fixed_thresholds))
    print("candidate_thresholds:", format_thresholds(candidate_thresholds))
    print("allowed_drop:", args.allowed_drop)
    print("max_changed_pred_ratio:", args.max_changed_pred_ratio)
    print("max_mean_logit_diff:", args.max_mean_logit_diff)
    print("progress_every:", args.progress_every)
    print("batch_progress_every:", args.batch_progress_every)
    print("skip_greedy:", args.skip_greedy)
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
        progress_label="software_baseline",
        progress_every=args.batch_progress_every,
        no_progress=args.no_progress,
    )
    software_score = software_metrics[task_cfg["primary_metric"]]
    print(f"software baseline done: {task_cfg['primary_metric']}={fmt_float(software_score)}", flush=True)
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
        progress_label="sparse_bitserial_reference",
        progress_every=args.batch_progress_every,
        no_progress=args.no_progress,
    )
    sparse_score = sparse_metrics[task_cfg["primary_metric"]]
    print(f"sparse_bitserial reference done: {task_cfg['primary_metric']}={fmt_float(sparse_score)}", flush=True)
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
        f"sparse_{task_cfg['primary_metric']}={fmt_float(sparse_score)}",
        flush=True,
    )

    summary_rows = [software_row, sparse_row]

    if fixed_thresholds is not None:
        fixed_row = evaluate_fixed_thresholds(
            task_name=args.task,
            task_cfg=task_cfg,
            loader=loader,
            args=args,
            software_score=software_score,
            sparse_score=sparse_score,
            sparse_logits=sparse_logits,
            sparse_predictions=sparse_predictions,
            fixed_thresholds=fixed_thresholds,
        )
        summary_rows.append(fixed_row)

    step_rows = []
    candidate_rows = []
    if not args.skip_greedy:
        print("Running GLUE-constrained TCS greedy search ...", flush=True)
        search_model = load_patched_sparse_model(
            task_cfg,
            args,
            enable_tcs=True,
            thresholds=start_thresholds,
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
        summary_rows.extend([initial_row, final_row])

    print_step_table(step_rows)
    print_summary_table(summary_rows)
    if args.print_candidates:
        print_candidate_table(candidate_rows)


if __name__ == "__main__":
    main()

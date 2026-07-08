import argparse

import torch

from tpdcim_bert_patch import make_sparse_block_mask


TRACK_NAME = "paper_like_hw_scaling"
BW_B = [22, 22, 20, 18, 16, 16, 14, 12]
BW_E = [28, 24, 22, 20, 18, 16, 14, 10]
THRESHOLD_SETS = {
    "bw_B": BW_B,
    "bw_E": BW_E,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Report paper-like TPDCIM QK/TCS activity proxies without task accuracy."
    )
    parser.add_argument("--N-list", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--local-window", type=int, default=1)
    parser.add_argument("--global-blocks", nargs="+", type=int, default=[0])
    parser.add_argument("--num-random-blocks", type=int, default=0)
    parser.add_argument("--threshold-sets", nargs="+", default=["bw_B", "bw_E"], choices=sorted(THRESHOLD_SETS))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run-sweep", action="store_true", help="Run a uniform [t]*8 threshold sweep.")
    parser.add_argument("--run-greedy-search", action="store_true", help="Search bit-wise thresholds for a target BigBird-relative reduction.")
    parser.add_argument("--target-reduction-vs-bigbird", type=float, default=0.71)
    parser.add_argument("--sweep-min-threshold", type=int, default=0)
    parser.add_argument("--sweep-max-threshold", type=int, default=None)
    parser.add_argument("--sweep-step", type=int, default=1)
    parser.add_argument("--sweep-top-k", type=int, default=12)
    parser.add_argument("--greedy-order", choices=["lsb_to_msb", "msb_to_lsb"], default="lsb_to_msb")
    parser.add_argument("--greedy-start", choices=["zero", "bw_B", "bw_E"], default="zero")
    return parser.parse_args()


def ceil_div(value, divisor):
    return (int(value) + int(divisor) - 1) // int(divisor)


def fmt_cell(value, digits=6):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_reduction(value):
    if value is None:
        return "-"
    return f"x{value:.3f}"


def format_thresholds(thresholds):
    return "[" + ",".join(str(int(v)) for v in thresholds) + "]"


def sparse_block_counts(num_blocks, args):
    block_mask = make_sparse_block_mask(
        num_blocks=num_blocks,
        local_window=args.local_window,
        global_blocks=tuple(args.global_blocks),
        num_random_blocks=args.num_random_blocks,
        device=None,
    )
    total_per_layer_head = num_blocks * num_blocks
    computed_per_layer_head = int(block_mask.sum().item())
    skipped_per_layer_head = total_per_layer_head - computed_per_layer_head
    return block_mask, total_per_layer_head, computed_per_layer_head, skipped_per_layer_head


def assert_expected_counts(N, args, total_per_layer_head, computed_per_layer_head):
    if N == 2048 and args.tile_n == 256:
        assert total_per_layer_head == 64, total_per_layer_head
        assert computed_per_layer_head == 34, computed_per_layer_head
    if N == 4096 and args.tile_n == 256:
        assert total_per_layer_head == 256, total_per_layer_head
        assert computed_per_layer_head == 74, computed_per_layer_head


def build_tcs_histograms(N, block_mask, args, max_threshold):
    """Build per-bit cumulative TCS skipped-row counts.

    Thresholds follow the current simulator convention: LSB-to-MSB, active when
    bit_sum > threshold. Counts are weighted by the number of computed K tiles
    for the Q block, so they match the existing row-skip activity proxy scale.
    """
    generator = torch.Generator().manual_seed(args.seed + N)
    num_blocks = ceil_div(N, args.tile_n)
    clipped_max = max(max_threshold, args.head_dim)
    histograms = [torch.zeros(clipped_max + 1, dtype=torch.long) for _ in range(8)]
    rows_total_by_bit = [0 for _ in range(8)]

    for _layer in range(args.layers):
        for _head in range(args.heads):
            for qi in range(num_blocks):
                q0 = qi * args.tile_n
                q1 = min(q0 + args.tile_n, N)
                rows = q1 - q0
                computed_k_tiles = int(block_mask[qi].sum().item())
                if computed_k_tiles == 0:
                    continue

                q_int8 = torch.randint(
                    low=-127,
                    high=128,
                    size=(rows, args.head_dim),
                    dtype=torch.int16,
                    generator=generator,
                )
                q_uint8 = torch.bitwise_and(q_int8, 255)

                for bit in range(8):
                    q_bit = torch.bitwise_and(torch.bitwise_right_shift(q_uint8, bit), 1)
                    bit_sum = q_bit.sum(dim=-1).to(torch.long)
                    counts = torch.bincount(bit_sum, minlength=args.head_dim + 1)
                    histograms[bit][: args.head_dim + 1] += counts * computed_k_tiles
                    rows_total_by_bit[bit] += rows * computed_k_tiles

    cumulative = [hist.cumsum(dim=0) for hist in histograms]
    return cumulative, rows_total_by_bit


def tcs_stats_from_histograms(cumulative, rows_total_by_bit, thresholds):
    tcs_rows_total = 0
    tcs_rows_skipped = 0
    max_index = cumulative[0].numel() - 1

    for bit, raw_threshold in enumerate(thresholds):
        threshold = int(raw_threshold)
        tcs_rows_total += rows_total_by_bit[bit]
        if threshold < 0:
            continue
        if threshold >= max_index:
            tcs_rows_skipped += rows_total_by_bit[bit]
        else:
            tcs_rows_skipped += int(cumulative[bit][threshold].item())

    ratio = 0.0 if tcs_rows_total == 0 else tcs_rows_skipped / tcs_rows_total
    return tcs_rows_total, tcs_rows_skipped, ratio


def make_activity_fields(qk_sparse_density, tcs_row_skip_ratio=None, dense=False):
    if dense:
        reduction_dense = 1.0
        reduction_bigbird = None
        dense_activity_ratio = 1.0
        bigbird_activity_ratio = None
    elif tcs_row_skip_ratio is None:
        reduction_dense = qk_sparse_density
        reduction_bigbird = 1.0
        dense_activity_ratio = qk_sparse_density
        bigbird_activity_ratio = 1.0
    else:
        keep_ratio = 1.0 - tcs_row_skip_ratio
        reduction_dense = qk_sparse_density * keep_ratio
        reduction_bigbird = keep_ratio
        dense_activity_ratio = reduction_dense
        bigbird_activity_ratio = reduction_bigbird

    saving_dense = 1.0 - reduction_dense
    saving_bigbird = None if reduction_bigbird is None else 1.0 - reduction_bigbird
    return {
        "dense_activity_ratio": dense_activity_ratio,
        "bigbird_activity_ratio": bigbird_activity_ratio,
        "computation_reduction_vs_dense": reduction_dense,
        "reduction_vs_dense_str": format_reduction(reduction_dense),
        "computation_reduction_vs_bigbird": reduction_bigbird,
        "reduction_vs_bigbird_str": format_reduction(reduction_bigbird),
        "operation_saving_vs_dense": saving_dense,
        "operation_saving_vs_bigbird": saving_bigbird,
    }


def make_row(args, N, mode, method, qk_total_tiles, qk_computed_tiles, qk_skipped_tiles, bit_ops_total, tcs_rows_total=0, tcs_rows_skipped=0, tcs_row_skip_ratio=0.0, thresholds=None):
    qk_sparse_density = qk_computed_tiles / qk_total_tiles if qk_total_tiles else 0.0
    qk_tile_skip_ratio = qk_skipped_tiles / qk_total_tiles if qk_total_tiles else 0.0
    dense = method == "dense_baseline"
    tcs_ratio = tcs_row_skip_ratio if method.startswith("prop_tcs") else None
    activity = make_activity_fields(qk_sparse_density, tcs_ratio, dense=dense)
    return {
        "track": TRACK_NAME,
        "N": N,
        "tile_n": args.tile_n,
        "num_blocks": ceil_div(N, args.tile_n),
        "layers": args.layers,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "mode": mode,
        "method": method,
        "thresholds_lsb_to_msb": None if thresholds is None else format_thresholds(thresholds),
        "qk_total_tiles": qk_total_tiles,
        "qk_computed_tiles": qk_computed_tiles,
        "qk_skipped_tiles": qk_skipped_tiles,
        "qk_sparse_density": qk_sparse_density,
        "qk_tile_skip_ratio": qk_tile_skip_ratio,
        "bit_ops_total": bit_ops_total,
        "tcs_rows_total": tcs_rows_total,
        "tcs_rows_skipped": tcs_rows_skipped,
        "tcs_row_skip_ratio": tcs_row_skip_ratio,
        **activity,
    }


def sweep_rows_for_N(args, N, qk_sparse_density, cumulative, rows_total_by_bit):
    max_threshold = args.sweep_max_threshold if args.sweep_max_threshold is not None else args.head_dim
    candidates = []
    for threshold in range(args.sweep_min_threshold, max_threshold + 1, args.sweep_step):
        thresholds = [threshold] * 8
        total, skipped, ratio = tcs_stats_from_histograms(cumulative, rows_total_by_bit, thresholds)
        activity = make_activity_fields(qk_sparse_density, ratio, dense=False)
        reduction_bigbird = activity["computation_reduction_vs_bigbird"]
        target_error = abs(reduction_bigbird - args.target_reduction_vs_bigbird)
        candidates.append(
            {
                "search": "uniform_sweep",
                "N": N,
                "thresholds_lsb_to_msb": format_thresholds(thresholds),
                "tcs_rows_total": total,
                "tcs_rows_skipped": skipped,
                "tcs_row_skip_ratio": ratio,
                "computation_reduction_vs_dense": activity["computation_reduction_vs_dense"],
                "reduction_vs_dense_str": activity["reduction_vs_dense_str"],
                "computation_reduction_vs_bigbird": reduction_bigbird,
                "reduction_vs_bigbird_str": activity["reduction_vs_bigbird_str"],
                "operation_saving_vs_bigbird": activity["operation_saving_vs_bigbird"],
                "target_reduction_vs_bigbird": args.target_reduction_vs_bigbird,
                "target_error": target_error,
            }
        )
    return sorted(candidates, key=lambda row: row["target_error"])[: args.sweep_top_k]


def greedy_search_for_N(args, N, qk_sparse_density, cumulative, rows_total_by_bit):
    max_threshold = args.sweep_max_threshold if args.sweep_max_threshold is not None else args.head_dim
    if args.greedy_start == "zero":
        thresholds = [0] * 8
    else:
        thresholds = list(THRESHOLD_SETS[args.greedy_start])

    bit_order = list(range(8)) if args.greedy_order == "lsb_to_msb" else list(reversed(range(8)))
    steps = []

    for bit in bit_order:
        best = None
        for threshold in range(args.sweep_min_threshold, max_threshold + 1, args.sweep_step):
            candidate = list(thresholds)
            candidate[bit] = threshold
            total, skipped, ratio = tcs_stats_from_histograms(cumulative, rows_total_by_bit, candidate)
            activity = make_activity_fields(qk_sparse_density, ratio, dense=False)
            reduction_bigbird = activity["computation_reduction_vs_bigbird"]
            target_error = abs(reduction_bigbird - args.target_reduction_vs_bigbird)
            record = (target_error, threshold, total, skipped, ratio, activity)
            if best is None or record[0] < best[0]:
                best = record

        target_error, threshold, total, skipped, ratio, activity = best
        thresholds[bit] = threshold
        steps.append(
            {
                "bit": bit,
                "chosen_threshold": threshold,
                "thresholds_lsb_to_msb": format_thresholds(thresholds),
                "tcs_row_skip_ratio": ratio,
                "reduction_vs_bigbird_str": activity["reduction_vs_bigbird_str"],
                "target_error": target_error,
            }
        )

    total, skipped, ratio = tcs_stats_from_histograms(cumulative, rows_total_by_bit, thresholds)
    activity = make_activity_fields(qk_sparse_density, ratio, dense=False)
    return thresholds, total, skipped, ratio, activity, steps


def print_table(rows):
    headers = [
        "track",
        "N",
        "tile_n",
        "num_blocks",
        "layers",
        "heads",
        "head_dim",
        "mode",
        "method",
        "thresholds_lsb_to_msb",
        "qk_total_tiles",
        "qk_computed_tiles",
        "qk_skipped_tiles",
        "qk_sparse_density",
        "qk_tile_skip_ratio",
        "bit_ops_total",
        "tcs_rows_total",
        "tcs_rows_skipped",
        "tcs_row_skip_ratio",
        "dense_activity_ratio",
        "bigbird_activity_ratio",
        "computation_reduction_vs_dense",
        "reduction_vs_dense_str",
        "computation_reduction_vs_bigbird",
        "reduction_vs_bigbird_str",
        "operation_saving_vs_dense",
        "operation_saving_vs_bigbird",
    ]
    print("\nMAIN HW ACTIVITY TABLE")
    print(" | ".join(headers))
    print(" | ".join("-" * len(h) for h in headers))
    for row in rows:
        print(" | ".join(fmt_cell(row.get(header)) for header in headers))


def print_search_table(title, rows):
    if not rows:
        return
    headers = [
        "search",
        "N",
        "thresholds_lsb_to_msb",
        "tcs_rows_total",
        "tcs_rows_skipped",
        "tcs_row_skip_ratio",
        "computation_reduction_vs_dense",
        "reduction_vs_dense_str",
        "computation_reduction_vs_bigbird",
        "reduction_vs_bigbird_str",
        "operation_saving_vs_bigbird",
        "target_reduction_vs_bigbird",
        "target_error",
    ]
    print("\n" + title)
    print(" | ".join(headers))
    print(" | ".join("-" * len(h) for h in headers))
    for row in rows:
        print(" | ".join(fmt_cell(row.get(header)) for header in headers))


def main():
    args = parse_args()
    max_threshold = args.sweep_max_threshold if args.sweep_max_threshold is not None else args.head_dim
    print("track:", TRACK_NAME)
    print("Note: this is a paper-like QK activity proxy, not exact paper reproduction.")
    print("Note: values are not measured CUDA speedup, memory saving, energy, or cycle-accurate hardware data.")
    print("TCS convention: thresholds are LSB-to-MSB, active row rule is bit_sum > threshold.")
    print("target_reduction_vs_bigbird:", args.target_reduction_vs_bigbird)

    rows = []
    sweep_rows = []
    greedy_rows = []
    greedy_step_rows = []

    for N in args.N_list:
        num_blocks = ceil_div(N, args.tile_n)
        block_mask, total_per_layer_head, computed_per_layer_head, skipped_per_layer_head = sparse_block_counts(num_blocks, args)
        assert_expected_counts(N, args, total_per_layer_head, computed_per_layer_head)

        multiplier = args.layers * args.heads
        dense_total = total_per_layer_head * multiplier
        sparse_computed = computed_per_layer_head * multiplier
        sparse_skipped = skipped_per_layer_head * multiplier
        qk_sparse_density = sparse_computed / dense_total if dense_total else 0.0

        cumulative, rows_total_by_bit = build_tcs_histograms(N, block_mask, args, max_threshold)

        rows.append(
            make_row(
                args=args,
                N=N,
                mode="dense_bitserial",
                method="dense_baseline",
                qk_total_tiles=dense_total,
                qk_computed_tiles=dense_total,
                qk_skipped_tiles=0,
                bit_ops_total=dense_total * 8,
            )
        )
        rows.append(
            make_row(
                args=args,
                N=N,
                mode="sparse_bitserial",
                method="bigbird_proxy",
                qk_total_tiles=dense_total,
                qk_computed_tiles=sparse_computed,
                qk_skipped_tiles=sparse_skipped,
                bit_ops_total=sparse_computed * 8,
            )
        )

        for threshold_name in args.threshold_sets:
            thresholds = THRESHOLD_SETS[threshold_name]
            tcs_total, tcs_skipped, tcs_ratio = tcs_stats_from_histograms(cumulative, rows_total_by_bit, thresholds)
            rows.append(
                make_row(
                    args=args,
                    N=N,
                    mode="sparse_bitserial_tcs",
                    method=f"prop_tcs_{threshold_name}",
                    qk_total_tiles=dense_total,
                    qk_computed_tiles=sparse_computed,
                    qk_skipped_tiles=sparse_skipped,
                    bit_ops_total=sparse_computed * 8,
                    tcs_rows_total=tcs_total,
                    tcs_rows_skipped=tcs_skipped,
                    tcs_row_skip_ratio=tcs_ratio,
                    thresholds=thresholds,
                )
            )

        if args.run_sweep:
            sweep_rows.extend(sweep_rows_for_N(args, N, qk_sparse_density, cumulative, rows_total_by_bit))

        if args.run_greedy_search:
            thresholds, total, skipped, ratio, activity, steps = greedy_search_for_N(
                args, N, qk_sparse_density, cumulative, rows_total_by_bit
            )
            rows.append(
                make_row(
                    args=args,
                    N=N,
                    mode="sparse_bitserial_tcs",
                    method=f"prop_tcs_greedy_target_{args.target_reduction_vs_bigbird:.3f}",
                    qk_total_tiles=dense_total,
                    qk_computed_tiles=sparse_computed,
                    qk_skipped_tiles=sparse_skipped,
                    bit_ops_total=sparse_computed * 8,
                    tcs_rows_total=total,
                    tcs_rows_skipped=skipped,
                    tcs_row_skip_ratio=ratio,
                    thresholds=thresholds,
                )
            )
            greedy_rows.append(
                {
                    "search": "greedy_final",
                    "N": N,
                    "thresholds_lsb_to_msb": format_thresholds(thresholds),
                    "tcs_rows_total": total,
                    "tcs_rows_skipped": skipped,
                    "tcs_row_skip_ratio": ratio,
                    "computation_reduction_vs_dense": activity["computation_reduction_vs_dense"],
                    "reduction_vs_dense_str": activity["reduction_vs_dense_str"],
                    "computation_reduction_vs_bigbird": activity["computation_reduction_vs_bigbird"],
                    "reduction_vs_bigbird_str": activity["reduction_vs_bigbird_str"],
                    "operation_saving_vs_bigbird": activity["operation_saving_vs_bigbird"],
                    "target_reduction_vs_bigbird": args.target_reduction_vs_bigbird,
                    "target_error": abs(activity["computation_reduction_vs_bigbird"] - args.target_reduction_vs_bigbird),
                }
            )
            for step_index, step in enumerate(steps):
                greedy_step_rows.append({"search": f"greedy_step_{step_index}", "N": N, **step})

    print_table(rows)
    print_search_table("UNIFORM THRESHOLD SWEEP TOP-K", sweep_rows)
    print_search_table("GREEDY SEARCH FINAL", greedy_rows)

    if greedy_step_rows:
        headers = [
            "search",
            "N",
            "bit",
            "chosen_threshold",
            "thresholds_lsb_to_msb",
            "tcs_row_skip_ratio",
            "reduction_vs_bigbird_str",
            "target_error",
        ]
        print("\nGREEDY SEARCH STEPS")
        print(" | ".join(headers))
        print(" | ".join("-" * len(h) for h in headers))
        for row in greedy_step_rows:
            print(" | ".join(fmt_cell(row.get(header)) for header in headers))


if __name__ == "__main__":
    main()

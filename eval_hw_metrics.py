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


def synthetic_tcs_stats(N, block_mask, thresholds, args):
    """Estimate TCS row skipping with synthetic INT8 Q bit activity.

    This follows the current simulator convention: thresholds are LSB-to-MSB and
    a row is active when bit_sum > threshold. The result is an activity proxy,
    not measured hardware speed, energy, or cycle behavior.
    """
    generator = torch.Generator().manual_seed(args.seed + N + sum(thresholds))
    num_blocks = ceil_div(N, args.tile_n)
    tcs_rows_total = 0
    tcs_rows_skipped = 0

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

                for bit, threshold in enumerate(thresholds):
                    q_bit = torch.bitwise_and(torch.bitwise_right_shift(q_uint8, bit), 1)
                    bit_sum = q_bit.sum(dim=-1)
                    skipped_rows = int((bit_sum <= threshold).sum().item())
                    tcs_rows_total += rows * computed_k_tiles
                    tcs_rows_skipped += skipped_rows * computed_k_tiles

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


def make_row(args, N, mode, method, qk_total_tiles, qk_computed_tiles, qk_skipped_tiles, bit_ops_total, tcs_rows_total=0, tcs_rows_skipped=0, tcs_row_skip_ratio=0.0):
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
    print(" | ".join(headers))
    print(" | ".join("-" * len(h) for h in headers))
    for row in rows:
        print(" | ".join(fmt_cell(row.get(header)) for header in headers))


def main():
    args = parse_args()
    print("track:", TRACK_NAME)
    print("Note: this is a paper-like QK activity proxy, not exact paper reproduction.")
    print("Note: values are not measured CUDA speedup, memory saving, energy, or cycle-accurate hardware data.")

    rows = []
    for N in args.N_list:
        num_blocks = ceil_div(N, args.tile_n)
        block_mask, total_per_layer_head, computed_per_layer_head, skipped_per_layer_head = sparse_block_counts(num_blocks, args)
        assert_expected_counts(N, args, total_per_layer_head, computed_per_layer_head)

        multiplier = args.layers * args.heads
        dense_total = total_per_layer_head * multiplier
        sparse_computed = computed_per_layer_head * multiplier
        sparse_skipped = skipped_per_layer_head * multiplier

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
            tcs_total, tcs_skipped, tcs_ratio = synthetic_tcs_stats(N, block_mask, thresholds, args)
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
                )
            )

    print_table(rows)


if __name__ == "__main__":
    main()

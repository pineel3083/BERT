#!/usr/bin/env python3
"""Parametric sweep wrapper for attention_cim_sim.py."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Iterable, List, Sequence

from attention_cim_sim import SimConfig, run_all


def parse_int_list(values: Sequence[str]) -> List[int]:
    parsed: List[int] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                parsed.append(int(item))
    return parsed


def iter_configs(args: argparse.Namespace) -> Iterable[SimConfig]:
    for n in args.N_list:
        for v_bw in args.v_replication_bandwidth_list:
            for write_a in args.write_a_cycles_list:
                for write_v in args.write_v_cycles_list:
                    for write_x in args.write_x_cycles_list:
                        for av_compute in args.av_compute_cycles_list:
                            for vgen in args.vgen_cycles_list:
                                yield SimConfig(
                                    n=n,
                                    d=args.D,
                                    a_row_tile=args.a_row_tile,
                                    reduction_tile=args.reduction_tile,
                                    v_output_tile=args.v_output_tile,
                                    total_engines=args.total_engines,
                                    qkg_engines=args.qkg_engines,
                                    vg_engines=args.vg_engines,
                                    write_a_cycles=write_a,
                                    write_v_cycles=write_v,
                                    write_x_cycles=write_x,
                                    av_compute_cycles=av_compute,
                                    vgen_cycles=vgen,
                                    replicate_v=args.replicate_v,
                                    v_replication_bandwidth=v_bw,
                                    output_dir=args.output_dir,
                                )


def result_rows(args: argparse.Namespace) -> Iterable[Dict[str, object]]:
    for cfg in iter_configs(args):
        if cfg.v_replication_bandwidth < 1:
            raise ValueError("v replication bandwidth must be >= 1")
        for result in run_all(cfg):
            row: Dict[str, object] = {
                "N": cfg.n,
                "D": cfg.d,
                "R_row_blocks": cfg.row_blocks,
                "C_column_blocks": cfg.column_blocks,
                "O_output_blocks": cfg.output_blocks,
                "a_row_tile": cfg.a_row_tile,
                "reduction_tile": cfg.reduction_tile,
                "v_output_tile": cfg.v_output_tile,
                "total_engines": cfg.total_engines,
                "qkg_engines": cfg.qkg_engines,
                "vg_engines": cfg.vg_engines,
                "write_a_cycles": cfg.write_a_cycles,
                "write_v_cycles": cfg.write_v_cycles,
                "write_x_cycles": cfg.write_x_cycles,
                "av_compute_cycles": cfg.av_compute_cycles,
                "vgen_cycles": cfg.vgen_cycles,
                "replicate_v": cfg.replicate_v,
                "v_replication_bandwidth": cfg.v_replication_bandwidth,
                "case": result.case,
            }
            row.update(result.metrics)
            yield row


def write_csv(rows: Sequence[Dict[str, object]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_compact(rows: Sequence[Dict[str, object]]) -> None:
    headers = [
        "N",
        "v_bw",
        "case",
        "cycles",
        "qkg_norm",
        "sys_norm",
        "busy_norm",
        "write_frac",
    ]
    table = []
    for row in rows:
        table.append(
            [
                str(row["N"]),
                str(row["v_replication_bandwidth"]),
                str(row["case"]),
                f"{float(row['total_cycles']):.0f}",
                f"{float(row['qkg_av_util_norm_to_hiva_paper']):.3f}",
                f"{float(row['system_av_util_norm_to_hiva_paper']):.3f}",
                f"{float(row['system_busy_util_norm_to_hiva_paper']):.3f}",
                f"{float(row['write_overhead_fraction']):.3f}",
            ]
        )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in table))
        for col in range(len(headers))
    ]
    print(" | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in table:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N-list", nargs="+", default=["1024", "2048", "4096"])
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--a-row-tile", type=int, default=192)
    parser.add_argument("--reduction-tile", type=int, default=64)
    parser.add_argument("--v-output-tile", type=int, default=64)
    parser.add_argument("--total-engines", type=int, default=6)
    parser.add_argument("--qkg-engines", type=int, default=4)
    parser.add_argument("--vg-engines", type=int, default=2)
    parser.add_argument("--write-a-cycles-list", nargs="+", default=["1"])
    parser.add_argument("--write-v-cycles-list", nargs="+", default=["1"])
    parser.add_argument("--write-x-cycles-list", nargs="+", default=["1"])
    parser.add_argument("--av-compute-cycles-list", nargs="+", default=["1"])
    parser.add_argument("--vgen-cycles-list", nargs="+", default=["1"])
    parser.add_argument("--v-replication-bandwidth-list", nargs="+", default=["1", "2", "4"])
    parser.add_argument("--replicate-v", action="store_true", default=True)
    parser.add_argument("--no-replicate-v", action="store_false", dest="replicate_v")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    args.N_list = parse_int_list(args.N_list)
    args.write_a_cycles_list = parse_int_list(args.write_a_cycles_list)
    args.write_v_cycles_list = parse_int_list(args.write_v_cycles_list)
    args.write_x_cycles_list = parse_int_list(args.write_x_cycles_list)
    args.av_compute_cycles_list = parse_int_list(args.av_compute_cycles_list)
    args.vgen_cycles_list = parse_int_list(args.vgen_cycles_list)
    args.v_replication_bandwidth_list = parse_int_list(args.v_replication_bandwidth_list)
    return args


def main() -> None:
    args = parse_args()
    rows = list(result_rows(args))
    output_csv = args.output_csv or os.path.join(args.output_dir, "attention_cim_param_sweep.csv")
    write_csv(rows, output_csv)

    print("track: attention_cim_av_dataflow_parametric_sweep")
    print("note: sweep results are ablation metrics, not exact TP-DCIM paper reproduction")
    print_compact(rows)
    print(f"saved sweep metrics: {output_csv}")


if __name__ == "__main__":
    main()

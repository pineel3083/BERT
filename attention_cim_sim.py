#!/usr/bin/env python3
"""Cycle-level AV MatMul dataflow simulator for CIM Transformer accelerators.

This is an ablation simulator, not an exact reproduction of the TP-DCIM paper.
It compares paper-like HIVA/V-stationary, resource-matched HIVA/V-stationary,
and proposed VIHA/A-stationary ping-pong schedules under a shared engine budget.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ACTIONS = (
    "IDLE",
    "WRITE_A",
    "WRITE_V",
    "WRITE_X",
    "VGEN_COMPUTE",
    "AV_COMPUTE",
)


@dataclass(frozen=True)
class SimConfig:
    n: int
    d: int = 64
    a_row_tile: int = 192
    reduction_tile: int = 64
    v_output_tile: int = 64
    total_engines: int = 6
    qkg_engines: int = 4
    vg_engines: int = 2
    write_a_cycles: int = 1
    write_v_cycles: int = 1
    write_x_cycles: int = 1
    av_compute_cycles: int = 1
    vgen_cycles: int = 1
    replicate_v: bool = True
    output_dir: str = "outputs"
    x_reuse_count: Optional[float] = None

    @property
    def row_blocks(self) -> int:
        return ceil_div(self.n, self.a_row_tile)

    @property
    def column_blocks(self) -> int:
        return ceil_div(self.n, self.reduction_tile)

    @property
    def output_blocks(self) -> int:
        return ceil_div(self.d, self.v_output_tile)

    @property
    def v_blocks(self) -> int:
        return self.column_blocks * self.output_blocks

    @property
    def av_tile_ops(self) -> int:
        return self.row_blocks * self.column_blocks * self.output_blocks


@dataclass(frozen=True)
class Engine:
    name: str
    role: str


@dataclass
class SimResult:
    case: str
    config: SimConfig
    timeline: "Timeline"
    metrics: Dict[str, float]


@dataclass(frozen=True)
class Batch:
    v_block: int
    rows: Tuple[int, ...]


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("tile sizes and cycle counts must be positive where used")
    return (a + b - 1) // b


def parse_bool(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {text}")


class Timeline:
    """Per-cycle engine action log."""

    def __init__(self, engines: Sequence[Engine]):
        self.engines = list(engines)
        self._actions: List[List[str]] = [[] for _ in engines]
        self._details: List[List[str]] = [[] for _ in engines]

    def schedule(self, engine_idx: int, start: int, duration: int, action: str, detail: str = "") -> None:
        if duration < 0:
            raise ValueError("duration must be non-negative")
        if duration == 0:
            return
        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action}")
        actions = self._actions[engine_idx]
        details = self._details[engine_idx]
        if start < len(actions):
            raise ValueError(
                f"engine {self.engines[engine_idx].name} overlaps at cycle {start}: "
                f"existing length {len(actions)}"
            )
        while len(actions) < start:
            actions.append("IDLE")
            details.append("")
        for _ in range(duration):
            actions.append(action)
            details.append(detail)

    def time_of(self, engine_idx: int) -> int:
        return len(self._actions[engine_idx])

    @property
    def total_cycles(self) -> int:
        return max((len(actions) for actions in self._actions), default=0)

    def finalize(self) -> None:
        total = self.total_cycles
        for actions, details in zip(self._actions, self._details):
            while len(actions) < total:
                actions.append("IDLE")
                details.append("")

    def breakdown(self) -> List[Dict[str, int]]:
        self.finalize()
        rows: List[Dict[str, int]] = []
        for engine, actions in zip(self.engines, self._actions):
            row = {"engine": engine.name, "role": engine.role}
            for action in ACTIONS:
                row[action] = actions.count(action)
            rows.append(row)
        return rows

    def action_total(self, action: str, role: Optional[str] = None) -> int:
        self.finalize()
        total = 0
        for engine, actions in zip(self.engines, self._actions):
            if role is not None and engine.role != role:
                continue
            total += actions.count(action)
        return total

    def non_idle_total(self) -> int:
        self.finalize()
        return sum(1 for actions in self._actions for action in actions if action != "IDLE")

    def idle_total(self) -> int:
        self.finalize()
        return sum(actions.count("IDLE") for actions in self._actions)

    def iter_csv_rows(self, case: str) -> Iterable[Dict[str, object]]:
        self.finalize()
        for cycle in range(self.total_cycles):
            for engine_idx, engine in enumerate(self.engines):
                yield {
                    "case": case,
                    "cycle": cycle,
                    "engine": engine.name,
                    "role": engine.role,
                    "action": self._actions[engine_idx][cycle],
                    "detail": self._details[engine_idx][cycle],
                }


def make_engines(cfg: SimConfig) -> List[Engine]:
    engines = [Engine(f"QKG{idx}", "QKG") for idx in range(cfg.qkg_engines)]
    engines.extend(Engine(f"VG{idx}", "VG") for idx in range(cfg.vg_engines))
    if len(engines) != cfg.total_engines:
        raise ValueError("total_engines must equal qkg_engines + vg_engines")
    return engines


def qkg_indices(cfg: SimConfig) -> List[int]:
    return list(range(cfg.qkg_engines))


def vg_indices(cfg: SimConfig) -> List[int]:
    return list(range(cfg.qkg_engines, cfg.qkg_engines + cfg.vg_engines))


def v_block_label(cfg: SimConfig, v_block: int) -> str:
    col = v_block // cfg.output_blocks
    out = v_block % cfg.output_blocks
    return f"V[c{col},o{out}]"


def schedule_v_generation_eager(timeline: Timeline, cfg: SimConfig, case: str) -> List[int]:
    """Generate all V blocks as soon as VG engines become available."""

    vg = vg_indices(cfg)
    ready = [0 for _ in range(cfg.v_blocks)]
    for v_block in range(cfg.v_blocks):
        engine_idx = min(vg, key=timeline.time_of)
        start = timeline.time_of(engine_idx)
        detail = v_block_label(cfg, v_block)
        timeline.schedule(engine_idx, start, cfg.write_x_cycles, "WRITE_X", f"{detail} X load")
        timeline.schedule(
            engine_idx,
            timeline.time_of(engine_idx),
            cfg.vgen_cycles,
            "VGEN_COMPUTE",
            f"{detail} gen ({case})",
        )
        ready[v_block] = timeline.time_of(engine_idx)
    return ready


def schedule_hiva_paper_like(cfg: SimConfig) -> SimResult:
    """Weak V-stationary baseline: one VG engine and one AV engine are active."""

    timeline = Timeline(make_engines(cfg))
    av_engine = qkg_indices(cfg)[0]
    vg_engine = vg_indices(cfg)[0]
    ready: Dict[int, int] = {}

    # Initial V block has to be generated before the first AV sweep can start.
    timeline.schedule(vg_engine, 0, cfg.write_x_cycles, "WRITE_X", f"{v_block_label(cfg, 0)} X load")
    timeline.schedule(vg_engine, timeline.time_of(vg_engine), cfg.vgen_cycles, "VGEN_COMPUTE", "initial V gen")
    ready[0] = timeline.time_of(vg_engine)

    for v_block in range(cfg.v_blocks):
        start = max(timeline.time_of(av_engine), ready[v_block])
        label = v_block_label(cfg, v_block)
        timeline.schedule(av_engine, start, cfg.write_v_cycles, "WRITE_V", f"{label} write")
        compute_start = timeline.time_of(av_engine)

        # The WV/VG engine may generate exactly the next V block during this sweep,
        # then it idles until the AV engine moves to the next column block.
        next_block = v_block + 1
        if next_block < cfg.v_blocks and next_block not in ready:
            vg_start = max(timeline.time_of(vg_engine), compute_start)
            next_label = v_block_label(cfg, next_block)
            timeline.schedule(vg_engine, vg_start, cfg.write_x_cycles, "WRITE_X", f"{next_label} X load")
            timeline.schedule(
                vg_engine,
                timeline.time_of(vg_engine),
                cfg.vgen_cycles,
                "VGEN_COMPUTE",
                f"{next_label} gen",
            )
            ready[next_block] = timeline.time_of(vg_engine)

        for row in range(cfg.row_blocks):
            detail = f"{label} sweep A[row{row}]"
            timeline.schedule(
                av_engine,
                timeline.time_of(av_engine),
                cfg.av_compute_cycles,
                "AV_COMPUTE",
                detail,
            )

    return build_result("hiva_paper_like", cfg, timeline)


def schedule_hiva_resource_matched(cfg: SimConfig) -> SimResult:
    timeline = Timeline(make_engines(cfg))
    ready = schedule_v_generation_eager(timeline, cfg, "hiva_resource_matched")
    qkg = qkg_indices(cfg)

    if cfg.replicate_v:
        qkg_time = 0
        for v_block in range(cfg.v_blocks):
            label = v_block_label(cfg, v_block)
            start = max(qkg_time, ready[v_block])
            for engine_idx in qkg:
                timeline.schedule(engine_idx, start, cfg.write_v_cycles, "WRITE_V", f"{label} replicated write")
            compute_start = start + cfg.write_v_cycles
            for row_base in range(0, cfg.row_blocks, len(qkg)):
                for lane, engine_idx in enumerate(qkg):
                    row = row_base + lane
                    if row < cfg.row_blocks:
                        timeline.schedule(
                            engine_idx,
                            compute_start,
                            cfg.av_compute_cycles,
                            "AV_COMPUTE",
                            f"{label} A[row{row}]",
                        )
                compute_start += cfg.av_compute_cycles
            qkg_time = compute_start
        case = "hiva_resource_matched_replicate_v"
    else:
        for v_block in range(cfg.v_blocks):
            label = v_block_label(cfg, v_block)
            engine_idx = min(qkg, key=lambda idx: max(timeline.time_of(idx), ready[v_block]))
            start = max(timeline.time_of(engine_idx), ready[v_block])
            timeline.schedule(engine_idx, start, cfg.write_v_cycles, "WRITE_V", f"{label} single write")
            for row in range(cfg.row_blocks):
                timeline.schedule(
                    engine_idx,
                    timeline.time_of(engine_idx),
                    cfg.av_compute_cycles,
                    "AV_COMPUTE",
                    f"{label} serial A[row{row}]",
                )
        case = "hiva_resource_matched_no_replicate"

    return build_result(case, cfg, timeline)


def make_viha_batches(cfg: SimConfig) -> List[Batch]:
    batches: List[Batch] = []
    for v_block in range(cfg.v_blocks):
        for row_base in range(0, cfg.row_blocks, 2):
            rows = tuple(row for row in (row_base, row_base + 1) if row < cfg.row_blocks)
            batches.append(Batch(v_block=v_block, rows=rows))
    return batches


def schedule_viha_a_stationary_pingpong(cfg: SimConfig) -> SimResult:
    if cfg.qkg_engines != 4:
        raise ValueError("viha_a_stationary_pingpong currently expects exactly 4 QKG engines")

    timeline = Timeline(make_engines(cfg))
    ready = schedule_v_generation_eager(timeline, cfg, "viha_a_stationary_pingpong")
    batches = make_viha_batches(cfg)
    if not batches:
        return build_result("viha_a_stationary_pingpong", cfg, timeline)

    groups = {
        0: (0, 1),
        1: (2, 3),
    }

    def load_batch(group_id: int, batch: Batch, start: int) -> int:
        label = v_block_label(cfg, batch.v_block)
        engines = groups[group_id]
        for lane, engine_idx in enumerate(engines):
            if lane < len(batch.rows):
                row = batch.rows[lane]
                timeline.schedule(
                    engine_idx,
                    start,
                    cfg.write_a_cycles,
                    "WRITE_A",
                    f"A[row{row},{label}] load",
                )
        return start + cfg.write_a_cycles

    def compute_batch(group_id: int, batch: Batch, start: int) -> int:
        label = v_block_label(cfg, batch.v_block)
        engines = groups[group_id]
        compute_start = max(start, ready[batch.v_block])
        for lane, engine_idx in enumerate(engines):
            if lane < len(batch.rows):
                row = batch.rows[lane]
                timeline.schedule(
                    engine_idx,
                    compute_start,
                    cfg.av_compute_cycles,
                    "AV_COMPUTE",
                    f"A[row{row},{label}] x {label}",
                )
        return compute_start + cfg.av_compute_cycles

    # Initial fill: group0 loads the first A batch before ping-pong phases begin.
    loaded_group = 0
    loaded_batch = batches[0]
    phase_start = load_batch(loaded_group, loaded_batch, 0)
    next_batch_idx = 1

    while loaded_batch is not None:
        compute_group = loaded_group
        load_group = 1 - compute_group

        compute_end = compute_batch(compute_group, loaded_batch, phase_start)
        next_loaded: Optional[Batch] = None
        load_end = phase_start
        if next_batch_idx < len(batches):
            next_loaded = batches[next_batch_idx]
            load_end = load_batch(load_group, next_loaded, phase_start)
            next_batch_idx += 1

        phase_start = max(compute_end, load_end)
        loaded_group = load_group
        loaded_batch = next_loaded

    return build_result("viha_a_stationary_pingpong", cfg, timeline)


def build_result(case: str, cfg: SimConfig, timeline: Timeline) -> SimResult:
    timeline.finalize()
    total_cycles = timeline.total_cycles
    total_engine_cycles = total_cycles * cfg.total_engines
    qkg_engine_cycles = total_cycles * cfg.qkg_engines
    useful_av = timeline.action_total("AV_COMPUTE", role="QKG")
    vgen = timeline.action_total("VGEN_COMPUTE", role="VG")
    write_a = timeline.action_total("WRITE_A")
    write_v = timeline.action_total("WRITE_V")
    write_x = timeline.action_total("WRITE_X")
    write_total = write_a + write_v + write_x
    non_idle = timeline.non_idle_total()
    idle = timeline.idle_total()
    x_reuse = cfg.x_reuse_count if cfg.x_reuse_count is not None else float(cfg.output_blocks)

    metrics: Dict[str, float] = {
        "total_cycles": float(total_cycles),
        "R_row_blocks": float(cfg.row_blocks),
        "C_column_blocks": float(cfg.column_blocks),
        "O_output_blocks": float(cfg.output_blocks),
        "total_av_tile_ops": float(cfg.av_tile_ops),
        "useful_av_compute_engine_cycles": float(useful_av),
        "v_generation_engine_cycles": float(vgen),
        "write_a_engine_cycles": float(write_a),
        "write_v_engine_cycles": float(write_v),
        "write_x_engine_cycles": float(write_x),
        "write_engine_cycles_total": float(write_total),
        "idle_engine_cycles": float(idle),
        "non_idle_engine_cycles": float(non_idle),
        "qkg_av_compute_utilization": safe_div(useful_av, qkg_engine_cycles),
        "system_av_compute_utilization": safe_div(useful_av, total_engine_cycles),
        "system_busy_utilization": safe_div(non_idle, total_engine_cycles),
        "write_overhead_fraction": safe_div(write_total, non_idle),
        "v_data_reuse_count": float(cfg.row_blocks),
        "a_data_reuse_count": float(cfg.output_blocks),
        "x_data_reuse_count": float(x_reuse),
    }
    return SimResult(case=case, config=cfg, timeline=timeline, metrics=metrics)


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def add_normalized_metrics(results: Sequence[SimResult]) -> None:
    baseline = next((result for result in results if result.case == "hiva_paper_like"), results[0])
    for result in results:
        result.metrics["qkg_av_util_norm_to_hiva_paper"] = safe_div(
            result.metrics["qkg_av_compute_utilization"],
            baseline.metrics["qkg_av_compute_utilization"],
        )
        result.metrics["system_av_util_norm_to_hiva_paper"] = safe_div(
            result.metrics["system_av_compute_utilization"],
            baseline.metrics["system_av_compute_utilization"],
        )
        result.metrics["system_busy_util_norm_to_hiva_paper"] = safe_div(
            result.metrics["system_busy_utilization"],
            baseline.metrics["system_busy_utilization"],
        )
        result.metrics["cycle_speedup_vs_hiva_paper"] = safe_div(
            baseline.metrics["total_cycles"],
            result.metrics["total_cycles"],
        )


def run_all(cfg: SimConfig) -> List[SimResult]:
    results = [
        schedule_hiva_paper_like(cfg),
        schedule_hiva_resource_matched(cfg),
        schedule_viha_a_stationary_pingpong(cfg),
    ]
    add_normalized_metrics(results)
    return results


def format_float(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}"


def print_summary(results: Sequence[SimResult]) -> None:
    columns = [
        ("case", "case"),
        ("cycles", "total_cycles"),
        ("av_ops", "total_av_tile_ops"),
        ("qkg_av_util", "qkg_av_compute_utilization"),
        ("qkg_norm", "qkg_av_util_norm_to_hiva_paper"),
        ("sys_av_util", "system_av_compute_utilization"),
        ("busy_util", "system_busy_utilization"),
        ("write_frac", "write_overhead_fraction"),
        ("speedup", "cycle_speedup_vs_hiva_paper"),
    ]
    rows = []
    for result in results:
        row = []
        for _, key in columns:
            if key == "case":
                row.append(result.case)
            else:
                row.append(format_float(result.metrics[key]))
        rows.append(row)

    widths = []
    for idx, (header, _) in enumerate(columns):
        widths.append(max(len(header), *(len(row[idx]) for row in rows)))

    header_line = " | ".join(header.ljust(widths[idx]) for idx, (header, _) in enumerate(columns))
    sep_line = "-+-".join("-" * width for width in widths)
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))))


def write_metrics_csv(results: Sequence[SimResult], path: str) -> None:
    keys = [
        "case",
        "N",
        "D",
        "a_row_tile",
        "reduction_tile",
        "v_output_tile",
        "qkg_engines",
        "vg_engines",
        "write_a_cycles",
        "write_v_cycles",
        "write_x_cycles",
        "av_compute_cycles",
        "vgen_cycles",
    ]
    metric_keys = list(results[0].metrics.keys())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys + metric_keys)
        writer.writeheader()
        for result in results:
            row: Dict[str, object] = {
                "case": result.case,
                "N": result.config.n,
                "D": result.config.d,
                "a_row_tile": result.config.a_row_tile,
                "reduction_tile": result.config.reduction_tile,
                "v_output_tile": result.config.v_output_tile,
                "qkg_engines": result.config.qkg_engines,
                "vg_engines": result.config.vg_engines,
                "write_a_cycles": result.config.write_a_cycles,
                "write_v_cycles": result.config.write_v_cycles,
                "write_x_cycles": result.config.write_x_cycles,
                "av_compute_cycles": result.config.av_compute_cycles,
                "vgen_cycles": result.config.vgen_cycles,
            }
            row.update(result.metrics)
            writer.writerow(row)


def write_timeline_csv(results: Sequence[SimResult], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        fieldnames = ["case", "cycle", "engine", "role", "action", "detail"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerows(result.timeline.iter_csv_rows(result.case))


def write_engine_breakdown_csv(results: Sequence[SimResult], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        fieldnames = ["case", "engine", "role", *ACTIONS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for row in result.timeline.breakdown():
                row_with_case: Dict[str, object] = {"case": result.case}
                row_with_case.update(row)
                writer.writerow(row_with_case)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=2048, help="sequence length")
    parser.add_argument("--D", type=int, default=64, help="attention head/output dimension")
    parser.add_argument("--a-row-tile", type=int, default=192, help="A row tile size")
    parser.add_argument("--reduction-tile", type=int, default=64, help="A/V reduction tile size")
    parser.add_argument("--v-output-tile", type=int, default=64, help="V output tile size")
    parser.add_argument("--total-engines", type=int, default=6)
    parser.add_argument("--qkg-engines", type=int, default=4)
    parser.add_argument("--vg-engines", type=int, default=2)
    parser.add_argument("--write-a-cycles", type=positive_int, default=1)
    parser.add_argument("--write-v-cycles", type=positive_int, default=1)
    parser.add_argument("--write-x-cycles", type=positive_int, default=1)
    parser.add_argument("--av-compute-cycles", type=positive_int, default=1)
    parser.add_argument("--vgen-cycles", type=positive_int, default=1)
    parser.add_argument("--replicate-v", type=parse_bool, default=True)
    parser.add_argument("--x-reuse-count", type=float, default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--timeline-csv", default=None)
    parser.add_argument("--breakdown-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimConfig(
        n=args.N,
        d=args.D,
        a_row_tile=args.a_row_tile,
        reduction_tile=args.reduction_tile,
        v_output_tile=args.v_output_tile,
        total_engines=args.total_engines,
        qkg_engines=args.qkg_engines,
        vg_engines=args.vg_engines,
        write_a_cycles=args.write_a_cycles,
        write_v_cycles=args.write_v_cycles,
        write_x_cycles=args.write_x_cycles,
        av_compute_cycles=args.av_compute_cycles,
        vgen_cycles=args.vgen_cycles,
        replicate_v=args.replicate_v,
        output_dir=args.output_dir,
        x_reuse_count=args.x_reuse_count,
    )

    results = run_all(cfg)

    print("track: attention_cim_av_dataflow_cycle_sim")
    print("note: ablation simulator, not exact TP-DCIM paper reproduction")
    print(
        "config: "
        f"N={cfg.n} D={cfg.d} R={cfg.row_blocks} C={cfg.column_blocks} "
        f"O={cfg.output_blocks} replicate_v={cfg.replicate_v}"
    )
    print_summary(results)

    metrics_path = args.metrics_csv or os.path.join(cfg.output_dir, f"attention_cim_sim_N{cfg.n}_metrics.csv")
    timeline_path = args.timeline_csv or os.path.join(cfg.output_dir, f"attention_cim_sim_N{cfg.n}_timeline.csv")
    breakdown_path = args.breakdown_csv or os.path.join(
        cfg.output_dir,
        f"attention_cim_sim_N{cfg.n}_engine_breakdown.csv",
    )
    write_metrics_csv(results, metrics_path)
    write_timeline_csv(results, timeline_path)
    write_engine_breakdown_csv(results, breakdown_path)
    print(f"saved metrics: {metrics_path}")
    print(f"saved timeline: {timeline_path}")
    print(f"saved engine breakdown: {breakdown_path}")


if __name__ == "__main__":
    main()

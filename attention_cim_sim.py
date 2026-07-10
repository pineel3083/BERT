#!/usr/bin/env python3
"""Cycle-level AV MatMul dataflow simulator for CIM Transformer accelerators.

This is an ablation simulator, not an exact reproduction of the TP-DCIM paper.
The model is intentionally explicit about V-generation, finite V buffering,
array writes, input feeding, and optimistic upper-bound assumptions.
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
    "WRITE_A_ARRAY",
    "WRITE_V_ARRAY",
    "LOAD_A_INPUT",
    "LOAD_V_INPUT",
    "LOAD_X_INPUT",
    "VGEN_COMPUTE",
    "AV_COMPUTE",
)

VIHA_MODES = ("pingpong", "prefetch_a_upper_bound", "serial")


@dataclass(frozen=True)
class SimConfig:
    n: int
    d: int = 64
    model_dim: int = 768
    a_row_tile: int = 192
    reduction_tile: int = 64
    v_output_tile: int = 64
    vg_reduction_tile: int = 128
    vgen_parts_per_v: Optional[int] = None
    bytes_per_element: int = 1
    v_buffer_bytes: int = 8192
    total_engines: int = 6
    qkg_engines: int = 4
    vg_engines: int = 2
    write_a_cycles: int = 1
    write_v_cycles: int = 1
    write_x_cycles_per_part: int = 1
    vgen_cycles_per_part: int = 1
    av_compute_cycles: int = 1
    hiva_a_input_load_cycles: int = 1
    hiva_overlap_a_input_with_compute: bool = False
    viha_v_input_load_cycles: int = 0
    viha_overlap_v_input_with_compute: bool = True
    replicate_v: bool = True
    v_replication_bandwidth: int = 1
    viha_mode: str = "pingpong"
    include_parallel_j_upper_bound: bool = False
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

    @property
    def resolved_vgen_parts_per_v(self) -> int:
        if self.vgen_parts_per_v is not None:
            return self.vgen_parts_per_v
        return ceil_div(self.model_dim, self.vg_reduction_tile)

    @property
    def v_tile_bytes(self) -> int:
        return self.reduction_tile * self.v_output_tile * self.bytes_per_element

    @property
    def a_tile_bytes(self) -> int:
        return self.a_row_tile * self.reduction_tile * self.bytes_per_element

    @property
    def x_part_bytes(self) -> int:
        return self.v_output_tile * self.vg_reduction_tile * self.bytes_per_element

    @property
    def v_buffer_capacity_blocks(self) -> int:
        return max(1, self.v_buffer_bytes // max(1, self.v_tile_bytes))


@dataclass(frozen=True)
class Engine:
    name: str
    role: str


@dataclass
class VBufferStats:
    max_occupancy_blocks: int = 0
    full_stall_cycles: int = 0
    qkg_wait_for_v_cycles: int = 0
    generated_blocks: int = 0
    generated_parts: int = 0


@dataclass
class SimResult:
    case: str
    config: SimConfig
    timeline: "Timeline"
    metrics: Dict[str, float]


@dataclass(frozen=True)
class Batch:
    batch_idx: int
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
    """Per-cycle engine action log.

    The simulator uses one visible action per engine per cycle. Hidden overlap
    assumptions are therefore represented by reducing scheduled visible latency,
    not by placing two actions in one engine slot.
    """

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

    def role_non_idle_total(self, role: str) -> int:
        self.finalize()
        return sum(
            1
            for engine, actions in zip(self.engines, self._actions)
            if engine.role == role
            for action in actions
            if action != "IDLE"
        )

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


class VBufferProducer:
    """Finite-buffer V producer with tiled partial V generation.

    Generated-but-not-consumed V blocks reserve entries in an 8KB-style buffer.
    This intentionally avoids the previous unlimited eager pre-generation model.
    """

    def __init__(self, timeline: Timeline, cfg: SimConfig, vg_engine_ids: Sequence[int]):
        self.timeline = timeline
        self.cfg = cfg
        self.vg_engine_ids = list(vg_engine_ids)
        self.next_to_generate = 0
        self.ready_time: Dict[int, int] = {}
        self.scheduled_unconsumed: List[int] = []
        self.stats = VBufferStats()

    def prime(self) -> None:
        self.fill(0)

    def fill(self, not_before: int) -> None:
        while (
            self.next_to_generate < self.cfg.v_blocks
            and len(self.scheduled_unconsumed) < self.cfg.v_buffer_capacity_blocks
        ):
            self._schedule_v_block(self.next_to_generate, not_before)

    def request(self, v_block: int, request_time: int) -> int:
        while v_block not in self.ready_time:
            if len(self.scheduled_unconsumed) >= self.cfg.v_buffer_capacity_blocks:
                break
            self._schedule_v_block(self.next_to_generate, request_time)
        if v_block not in self.ready_time:
            raise RuntimeError(f"V block {v_block} is not available; consumption order is inconsistent")
        ready = self.ready_time[v_block]
        if ready > request_time:
            self.stats.qkg_wait_for_v_cycles += ready - request_time
        return max(request_time, ready)

    def consume(self, v_block: int, consume_time: int) -> None:
        if v_block in self.scheduled_unconsumed:
            self.scheduled_unconsumed.remove(v_block)
        if self.next_to_generate < self.cfg.v_blocks:
            earliest_vg = min(self.timeline.time_of(engine_idx) for engine_idx in self.vg_engine_ids)
            if earliest_vg < consume_time:
                self.stats.full_stall_cycles += consume_time - earliest_vg
        self.fill(consume_time)

    def _schedule_v_block(self, v_block: int, not_before: int) -> None:
        parts = self.cfg.resolved_vgen_parts_per_v
        end_times: List[int] = []
        label = v_block_label(self.cfg, v_block)
        for part in range(parts):
            engine_idx = min(self.vg_engine_ids, key=self.timeline.time_of)
            start = max(self.timeline.time_of(engine_idx), not_before)
            detail = f"{label} p{part}/{parts}"
            self.timeline.schedule(
                engine_idx,
                start,
                self.cfg.write_x_cycles_per_part,
                "LOAD_X_INPUT",
                f"{detail} X load",
            )
            self.timeline.schedule(
                engine_idx,
                self.timeline.time_of(engine_idx),
                self.cfg.vgen_cycles_per_part,
                "VGEN_COMPUTE",
                f"{detail} gen",
            )
            end_times.append(self.timeline.time_of(engine_idx))
        self.ready_time[v_block] = max(end_times) if end_times else not_before
        self.scheduled_unconsumed.append(v_block)
        self.next_to_generate += 1
        self.stats.generated_blocks += 1
        self.stats.generated_parts += parts
        self.stats.max_occupancy_blocks = max(self.stats.max_occupancy_blocks, len(self.scheduled_unconsumed))


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


def schedule_v_array_write(
    timeline: Timeline,
    cfg: SimConfig,
    qkg: Sequence[int],
    start: int,
    v_block: int,
    replicate: bool,
) -> int:
    label = v_block_label(cfg, v_block)
    targets = list(qkg) if replicate else [qkg[0]]
    bandwidth = max(1, cfg.v_replication_bandwidth)
    cursor = start
    for base in range(0, len(targets), bandwidth):
        for engine_idx in targets[base : base + bandwidth]:
            timeline.schedule(engine_idx, cursor, cfg.write_v_cycles, "WRITE_V_ARRAY", f"{label} array write")
        cursor += cfg.write_v_cycles
    return cursor


def schedule_hiva_rows(
    timeline: Timeline,
    cfg: SimConfig,
    assignments: Sequence[Tuple[int, int]],
    start: int,
    label: str,
) -> int:
    end = start
    for engine_idx, row in assignments:
        cursor = start
        if cfg.hiva_a_input_load_cycles:
            timeline.schedule(
                engine_idx,
                cursor,
                cfg.hiva_a_input_load_cycles,
                "LOAD_A_INPUT",
                f"A[row{row},{label}] input",
            )
            cursor += cfg.hiva_a_input_load_cycles
        timeline.schedule(engine_idx, cursor, cfg.av_compute_cycles, "AV_COMPUTE", f"A[row{row}] x {label}")
        end = max(end, timeline.time_of(engine_idx))
    return end


def schedule_viha_compute_rows(
    timeline: Timeline,
    cfg: SimConfig,
    assignments: Sequence[Tuple[int, int]],
    start: int,
    label: str,
) -> int:
    end = start
    for engine_idx, row in assignments:
        cursor = start
        if cfg.viha_v_input_load_cycles and not cfg.viha_overlap_v_input_with_compute:
            timeline.schedule(
                engine_idx,
                cursor,
                cfg.viha_v_input_load_cycles,
                "LOAD_V_INPUT",
                f"{label} input for A[row{row}]",
            )
            cursor += cfg.viha_v_input_load_cycles
        elif cfg.viha_v_input_load_cycles:
            # Visible timing assumes the V input feed is hidden inside AV compute.
            timeline.schedule(
                engine_idx,
                cursor,
                cfg.viha_v_input_load_cycles,
                "LOAD_V_INPUT",
                f"{label} hidden input feed",
            )
            cursor += cfg.viha_v_input_load_cycles
        timeline.schedule(engine_idx, cursor, cfg.av_compute_cycles, "AV_COMPUTE", f"A[row{row}] x {label}")
        end = max(end, timeline.time_of(engine_idx))
    return end


def schedule_hiva_paper_like(cfg: SimConfig) -> SimResult:
    """Weak V-stationary baseline: one AV engine and one WV/VG engine are active."""

    timeline = Timeline(make_engines(cfg))
    av_engine = qkg_indices(cfg)[0]
    producer = VBufferProducer(timeline, cfg, [vg_indices(cfg)[0]])
    producer.prime()
    av_time = 0

    for v_block in range(cfg.v_blocks):
        label = v_block_label(cfg, v_block)
        av_time = producer.request(v_block, av_time)
        av_time = schedule_v_array_write(timeline, cfg, [av_engine], av_time, v_block, replicate=False)
        for row in range(cfg.row_blocks):
            av_time = schedule_hiva_rows(timeline, cfg, [(av_engine, row)], av_time, label)
        producer.consume(v_block, av_time)

    return build_result(
        "hiva_paper_like",
        cfg,
        timeline,
        producer.stats,
        hiva_j_serial=False,
        hiva_parallel_j_upper_bound=False,
        v_replication_factor=1,
    )


def schedule_hiva_resource_matched_j_serial(cfg: SimConfig) -> SimResult:
    """Fair resource-matched HIVA: one V_j at a time, A rows parallelized across QKG."""

    timeline = Timeline(make_engines(cfg))
    qkg = qkg_indices(cfg)
    producer = VBufferProducer(timeline, cfg, vg_indices(cfg))
    producer.prime()
    qkg_time = 0

    for v_block in range(cfg.v_blocks):
        label = v_block_label(cfg, v_block)
        qkg_time = producer.request(v_block, qkg_time)
        qkg_time = schedule_v_array_write(timeline, cfg, qkg, qkg_time, v_block, cfg.replicate_v)

        if cfg.replicate_v:
            cursor = qkg_time
            for row_base in range(0, cfg.row_blocks, len(qkg)):
                assignments = [
                    (engine_idx, row_base + lane)
                    for lane, engine_idx in enumerate(qkg)
                    if row_base + lane < cfg.row_blocks
                ]
                cursor = schedule_hiva_rows(timeline, cfg, assignments, cursor, label)
            qkg_time = cursor
            replication_factor = len(qkg)
        else:
            engine_idx = qkg[0]
            cursor = max(qkg_time, timeline.time_of(engine_idx))
            for row in range(cfg.row_blocks):
                cursor = schedule_hiva_rows(timeline, cfg, [(engine_idx, row)], cursor, label)
            qkg_time = cursor
            replication_factor = 1

        producer.consume(v_block, qkg_time)

    return build_result(
        "hiva_resource_matched_j_serial",
        cfg,
        timeline,
        producer.stats,
        hiva_j_serial=True,
        hiva_parallel_j_upper_bound=False,
        v_replication_factor=replication_factor,
    )


def schedule_hiva_parallel_j_upper_bound(cfg: SimConfig) -> SimResult:
    """Optimistic upper bound: different V_j blocks can occupy different QKG engines."""

    timeline = Timeline(make_engines(cfg))
    qkg = qkg_indices(cfg)
    producer = VBufferProducer(timeline, cfg, vg_indices(cfg))
    producer.prime()

    for v_block in range(cfg.v_blocks):
        engine_idx = min(qkg, key=timeline.time_of)
        label = v_block_label(cfg, v_block)
        start = producer.request(v_block, timeline.time_of(engine_idx))
        cursor = schedule_v_array_write(timeline, cfg, [engine_idx], start, v_block, replicate=False)
        for row in range(cfg.row_blocks):
            cursor = schedule_hiva_rows(timeline, cfg, [(engine_idx, row)], cursor, label)
        producer.consume(v_block, cursor)

    return build_result(
        "hiva_resource_matched_parallel_j_upper_bound",
        cfg,
        timeline,
        producer.stats,
        hiva_j_serial=False,
        hiva_parallel_j_upper_bound=True,
        v_replication_factor=1,
    )


def make_viha_batches(cfg: SimConfig) -> List[Batch]:
    batches: List[Batch] = []
    batch_idx = 0
    for v_block in range(cfg.v_blocks):
        for row_base in range(0, cfg.row_blocks, 2):
            rows = tuple(row for row in (row_base, row_base + 1) if row < cfg.row_blocks)
            batches.append(Batch(batch_idx=batch_idx, v_block=v_block, rows=rows))
            batch_idx += 1
    return batches


def schedule_viha_a_stationary_pingpong(cfg: SimConfig) -> SimResult:
    if cfg.qkg_engines != 4:
        raise ValueError("viha ping-pong currently expects exactly 4 QKG engines")

    timeline = Timeline(make_engines(cfg))
    producer = VBufferProducer(timeline, cfg, vg_indices(cfg))
    producer.prime()
    batches = make_viha_batches(cfg)
    if not batches:
        return build_result("viha_pingpong_2compute_2load", cfg, timeline, producer.stats)

    groups = {0: (0, 1), 1: (2, 3)}
    last_batch_for_v = {batch.v_block: batch.batch_idx for batch in batches}

    def load_batch(group_id: int, batch: Batch, start: int) -> int:
        label = v_block_label(cfg, batch.v_block)
        end = start
        for lane, engine_idx in enumerate(groups[group_id]):
            if lane < len(batch.rows):
                row = batch.rows[lane]
                timeline.schedule(
                    engine_idx,
                    start,
                    cfg.write_a_cycles,
                    "WRITE_A_ARRAY",
                    f"A[row{row},{label}] array write",
                )
                end = max(end, timeline.time_of(engine_idx))
        return end

    def compute_batch(group_id: int, batch: Batch, start: int) -> int:
        label = v_block_label(cfg, batch.v_block)
        compute_start = producer.request(batch.v_block, start)
        assignments = [
            (engine_idx, batch.rows[lane])
            for lane, engine_idx in enumerate(groups[group_id])
            if lane < len(batch.rows)
        ]
        return schedule_viha_compute_rows(timeline, cfg, assignments, compute_start, label)

    loaded_group = 0
    loaded_batch = batches[0]
    phase_start = load_batch(loaded_group, loaded_batch, 0)
    next_batch_idx = 1

    while loaded_batch is not None:
        compute_group = loaded_group
        load_group = 1 - compute_group
        compute_end = compute_batch(compute_group, loaded_batch, phase_start)

        if loaded_batch.batch_idx == last_batch_for_v[loaded_batch.v_block]:
            producer.consume(loaded_batch.v_block, compute_end)

        next_loaded: Optional[Batch] = None
        load_end = phase_start
        if next_batch_idx < len(batches):
            next_loaded = batches[next_batch_idx]
            load_end = load_batch(load_group, next_loaded, phase_start)
            next_batch_idx += 1

        phase_start = max(compute_end, load_end)
        loaded_group = load_group
        loaded_batch = next_loaded

    return build_result("viha_pingpong_2compute_2load", cfg, timeline, producer.stats)


def schedule_viha_serial_load_compute(cfg: SimConfig) -> SimResult:
    timeline = Timeline(make_engines(cfg))
    producer = VBufferProducer(timeline, cfg, vg_indices(cfg))
    producer.prime()
    batches = make_viha_batches(cfg)
    groups = {0: (0, 1), 1: (2, 3)}
    last_batch_for_v = {batch.v_block: batch.batch_idx for batch in batches}
    phase_start = 0

    for batch in batches:
        group_id = batch.batch_idx % 2
        label = v_block_label(cfg, batch.v_block)
        load_end = phase_start
        for lane, engine_idx in enumerate(groups[group_id]):
            if lane < len(batch.rows):
                row = batch.rows[lane]
                timeline.schedule(
                    engine_idx,
                    phase_start,
                    cfg.write_a_cycles,
                    "WRITE_A_ARRAY",
                    f"A[row{row},{label}] serial array write",
                )
                load_end = max(load_end, timeline.time_of(engine_idx))
        compute_start = producer.request(batch.v_block, load_end)
        assignments = [
            (engine_idx, batch.rows[lane])
            for lane, engine_idx in enumerate(groups[group_id])
            if lane < len(batch.rows)
        ]
        phase_start = schedule_viha_compute_rows(timeline, cfg, assignments, compute_start, label)
        if batch.batch_idx == last_batch_for_v[batch.v_block]:
            producer.consume(batch.v_block, phase_start)

    return build_result("viha_serial_load_compute", cfg, timeline, producer.stats)


def schedule_viha_prefetch_a_upper_bound(cfg: SimConfig) -> SimResult:
    timeline = Timeline(make_engines(cfg))
    qkg = qkg_indices(cfg)
    producer = VBufferProducer(timeline, cfg, vg_indices(cfg))
    producer.prime()
    qkg_time = 0

    for v_block in range(cfg.v_blocks):
        label = v_block_label(cfg, v_block)
        qkg_time = producer.request(v_block, qkg_time)
        cursor = qkg_time
        for row_base in range(0, cfg.row_blocks, len(qkg)):
            assignments = [
                (engine_idx, row_base + lane)
                for lane, engine_idx in enumerate(qkg)
                if row_base + lane < cfg.row_blocks
            ]
            cursor = schedule_viha_compute_rows(timeline, cfg, assignments, cursor, label)
        qkg_time = cursor
        producer.consume(v_block, qkg_time)

    return build_result("viha_prefetch_a_4compute_upper_bound", cfg, timeline, producer.stats)


def schedule_viha(cfg: SimConfig) -> SimResult:
    if cfg.viha_mode == "pingpong":
        return schedule_viha_a_stationary_pingpong(cfg)
    if cfg.viha_mode == "prefetch_a_upper_bound":
        return schedule_viha_prefetch_a_upper_bound(cfg)
    if cfg.viha_mode == "serial":
        return schedule_viha_serial_load_compute(cfg)
    raise ValueError(f"unknown viha mode: {cfg.viha_mode}")


def event_count(total_cycles: int, cycles_per_event: int) -> int:
    if cycles_per_event <= 0:
        return 0
    return total_cycles // cycles_per_event


def build_result(
    case: str,
    cfg: SimConfig,
    timeline: Timeline,
    v_stats: Optional[VBufferStats] = None,
    hiva_j_serial: bool = False,
    hiva_parallel_j_upper_bound: bool = False,
    v_replication_factor: int = 0,
) -> SimResult:
    timeline.finalize()
    v_stats = v_stats or VBufferStats()
    total_cycles = timeline.total_cycles
    total_engine_cycles = total_cycles * cfg.total_engines
    qkg_engine_cycles = total_cycles * cfg.qkg_engines
    vg_engine_cycles = total_cycles * cfg.vg_engines

    useful_av = timeline.action_total("AV_COMPUTE", role="QKG")
    a_array_write = timeline.action_total("WRITE_A_ARRAY", role="QKG")
    v_array_write = timeline.action_total("WRITE_V_ARRAY", role="QKG")
    a_input_load = timeline.action_total("LOAD_A_INPUT", role="QKG")
    v_input_load = timeline.action_total("LOAD_V_INPUT", role="QKG")
    x_input_load = timeline.action_total("LOAD_X_INPUT", role="VG")
    vgen = timeline.action_total("VGEN_COMPUTE", role="VG")
    write_or_input_total = a_array_write + v_array_write + a_input_load + v_input_load + x_input_load
    non_idle = timeline.non_idle_total()
    idle = timeline.idle_total()
    qkg_busy = timeline.role_non_idle_total("QKG")
    vg_busy = timeline.role_non_idle_total("VG")
    x_reuse = cfg.x_reuse_count if cfg.x_reuse_count is not None else float(cfg.output_blocks)

    a_array_events = event_count(a_array_write, cfg.write_a_cycles)
    v_array_events = event_count(v_array_write, cfg.write_v_cycles)
    a_input_events = event_count(a_input_load, cfg.hiva_a_input_load_cycles)
    v_input_events = event_count(v_input_load, cfg.viha_v_input_load_cycles)
    x_input_events = event_count(x_input_load, cfg.write_x_cycles_per_part)

    metrics: Dict[str, float] = {
        "total_cycles": float(total_cycles),
        "R_row_blocks": float(cfg.row_blocks),
        "C_column_blocks": float(cfg.column_blocks),
        "O_output_blocks": float(cfg.output_blocks),
        "total_av_tile_ops": float(cfg.av_tile_ops),
        "useful_av_compute_engine_cycles": float(useful_av),
        "total_v_generation_parts": float(v_stats.generated_parts),
        "total_v_generation_engine_cycles": float(vgen),
        "total_a_array_write_engine_cycles": float(a_array_write),
        "total_v_array_write_engine_cycles": float(v_array_write),
        "total_a_input_load_engine_cycles": float(a_input_load),
        "total_v_input_load_engine_cycles": float(v_input_load),
        "total_x_input_load_engine_cycles": float(x_input_load),
        "write_or_input_engine_cycles_total": float(write_or_input_total),
        "idle_engine_cycles": float(idle),
        "non_idle_engine_cycles": float(non_idle),
        "qkg_av_compute_utilization": safe_div(useful_av, qkg_engine_cycles),
        "system_av_compute_utilization": safe_div(useful_av, total_engine_cycles),
        "system_busy_utilization": safe_div(non_idle, total_engine_cycles),
        "qkg_busy_utilization": safe_div(qkg_busy, qkg_engine_cycles),
        "vg_busy_utilization": safe_div(vg_busy, vg_engine_cycles),
        "qkg_av_compute_fraction": safe_div(useful_av, qkg_busy),
        "qkg_array_write_fraction": safe_div(a_array_write + v_array_write, qkg_busy),
        "qkg_input_load_fraction": safe_div(a_input_load + v_input_load, qkg_busy),
        "write_overhead_fraction": safe_div(write_or_input_total, non_idle),
        "v_data_reuse_count": float(cfg.row_blocks),
        "a_data_reuse_count": float(cfg.output_blocks),
        "x_data_reuse_count": float(x_reuse),
        "model_dim": float(cfg.model_dim),
        "vg_reduction_tile": float(cfg.vg_reduction_tile),
        "vgen_parts_per_v": float(cfg.resolved_vgen_parts_per_v),
        "write_x_cycles_per_part": float(cfg.write_x_cycles_per_part),
        "vgen_cycles_per_part": float(cfg.vgen_cycles_per_part),
        "v_tile_bytes": float(cfg.v_tile_bytes),
        "v_buffer_bytes": float(cfg.v_buffer_bytes),
        "v_buffer_capacity_blocks": float(cfg.v_buffer_capacity_blocks),
        "max_v_buffer_occupancy_blocks": float(v_stats.max_occupancy_blocks),
        "v_buffer_full_stall_cycles": float(v_stats.full_stall_cycles),
        "qkg_wait_for_v_cycles": float(v_stats.qkg_wait_for_v_cycles),
        "total_a_array_write_bytes": float(a_array_events * cfg.a_tile_bytes),
        "total_v_array_write_bytes": float(v_array_events * cfg.v_tile_bytes),
        "total_a_input_bytes": float(a_input_events * cfg.a_tile_bytes),
        "total_v_input_bytes": float(v_input_events * cfg.v_tile_bytes),
        "total_x_input_bytes": float(x_input_events * cfg.x_part_bytes),
        "total_write_or_input_bytes": float(
            a_array_events * cfg.a_tile_bytes
            + v_array_events * cfg.v_tile_bytes
            + a_input_events * cfg.a_tile_bytes
            + v_input_events * cfg.v_tile_bytes
            + x_input_events * cfg.x_part_bytes
        ),
        "v_replication_factor": float(v_replication_factor),
        "v_replication_bandwidth": float(cfg.v_replication_bandwidth),
        "hiva_j_serial": float(1 if hiva_j_serial else 0),
        "hiva_parallel_j_upper_bound": float(1 if hiva_parallel_j_upper_bound else 0),
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
    if cfg.viha_mode not in VIHA_MODES:
        raise ValueError(f"--viha-mode must be one of {', '.join(VIHA_MODES)}")
    results = [
        schedule_hiva_paper_like(cfg),
        schedule_hiva_resource_matched_j_serial(cfg),
    ]
    if cfg.include_parallel_j_upper_bound:
        results.append(schedule_hiva_parallel_j_upper_bound(cfg))
    results.append(schedule_viha(cfg))
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
        ("qkg_busy", "qkg_busy_utilization"),
        ("vg_busy", "vg_busy_utilization"),
        ("vbuf_max", "max_v_buffer_occupancy_blocks"),
        ("qkg_wait_v", "qkg_wait_for_v_cycles"),
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

    print(" | ".join(header.ljust(widths[idx]) for idx, (header, _) in enumerate(columns)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))))


def config_csv_keys() -> List[str]:
    return [
        "case",
        "N",
        "D",
        "model_dim",
        "a_row_tile",
        "reduction_tile",
        "v_output_tile",
        "vg_reduction_tile",
        "vgen_parts_per_v",
        "bytes_per_element",
        "v_buffer_bytes",
        "qkg_engines",
        "vg_engines",
        "write_a_cycles",
        "write_v_cycles",
        "write_x_cycles_per_part",
        "vgen_cycles_per_part",
        "av_compute_cycles",
        "hiva_a_input_load_cycles",
        "viha_v_input_load_cycles",
        "replicate_v",
        "v_replication_bandwidth",
        "viha_mode",
        "include_parallel_j_upper_bound",
    ]


def result_config_row(result: SimResult) -> Dict[str, object]:
    cfg = result.config
    return {
        "case": result.case,
        "N": cfg.n,
        "D": cfg.d,
        "model_dim": cfg.model_dim,
        "a_row_tile": cfg.a_row_tile,
        "reduction_tile": cfg.reduction_tile,
        "v_output_tile": cfg.v_output_tile,
        "vg_reduction_tile": cfg.vg_reduction_tile,
        "vgen_parts_per_v": cfg.resolved_vgen_parts_per_v,
        "bytes_per_element": cfg.bytes_per_element,
        "v_buffer_bytes": cfg.v_buffer_bytes,
        "qkg_engines": cfg.qkg_engines,
        "vg_engines": cfg.vg_engines,
        "write_a_cycles": cfg.write_a_cycles,
        "write_v_cycles": cfg.write_v_cycles,
        "write_x_cycles_per_part": cfg.write_x_cycles_per_part,
        "vgen_cycles_per_part": cfg.vgen_cycles_per_part,
        "av_compute_cycles": cfg.av_compute_cycles,
        "hiva_a_input_load_cycles": cfg.hiva_a_input_load_cycles,
        "viha_v_input_load_cycles": cfg.viha_v_input_load_cycles,
        "replicate_v": cfg.replicate_v,
        "v_replication_bandwidth": cfg.v_replication_bandwidth,
        "viha_mode": cfg.viha_mode,
        "include_parallel_j_upper_bound": cfg.include_parallel_j_upper_bound,
    }


def write_metrics_csv(results: Sequence[SimResult], path: str) -> None:
    keys = config_csv_keys()
    metric_keys = list(results[0].metrics.keys())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys + metric_keys)
        writer.writeheader()
        for result in results:
            row = result_config_row(result)
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


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def optional_positive_int(value: str) -> Optional[int]:
    if value.strip().lower() in ("auto", "none", "null"):
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive or auto")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=2048, help="sequence length")
    parser.add_argument("--D", type=int, default=64, help="attention head/output dimension")
    parser.add_argument("--model-dim", type=int, default=768)
    parser.add_argument("--a-row-tile", type=int, default=192)
    parser.add_argument("--reduction-tile", type=int, default=64)
    parser.add_argument("--v-output-tile", type=int, default=64)
    parser.add_argument("--vg-reduction-tile", type=int, default=128)
    parser.add_argument("--vgen-parts-per-v", type=optional_positive_int, default=None)
    parser.add_argument("--bytes-per-element", type=int, default=1)
    parser.add_argument("--v-buffer-bytes", type=int, default=8192)
    parser.add_argument("--total-engines", type=int, default=6)
    parser.add_argument("--qkg-engines", type=int, default=4)
    parser.add_argument("--vg-engines", type=int, default=2)
    parser.add_argument("--write-a-cycles", type=non_negative_int, default=1)
    parser.add_argument("--write-v-cycles", type=non_negative_int, default=1)
    parser.add_argument("--write-x-cycles-per-part", type=non_negative_int, default=1)
    parser.add_argument("--vgen-cycles-per-part", type=non_negative_int, default=1)
    parser.add_argument("--write-x-cycles", type=non_negative_int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--vgen-cycles", type=non_negative_int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--av-compute-cycles", type=non_negative_int, default=1)
    parser.add_argument("--hiva-a-input-load-cycles", type=non_negative_int, default=1)
    parser.add_argument("--hiva-overlap-a-input-with-compute", type=parse_bool, default=False)
    parser.add_argument("--viha-v-input-load-cycles", type=non_negative_int, default=0)
    parser.add_argument("--viha-overlap-v-input-with-compute", type=parse_bool, default=True)
    parser.add_argument("--replicate-v", type=parse_bool, default=True)
    parser.add_argument("--v-replication-bandwidth", type=int, default=1)
    parser.add_argument("--viha-mode", choices=VIHA_MODES, default="pingpong")
    parser.add_argument("--include-parallel-j-upper-bound", action="store_true")
    parser.add_argument("--x-reuse-count", type=float, default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--timeline-csv", default=None)
    parser.add_argument("--breakdown-csv", default=None)
    args = parser.parse_args()
    if args.write_x_cycles is not None:
        args.write_x_cycles_per_part = args.write_x_cycles
    if args.vgen_cycles is not None:
        args.vgen_cycles_per_part = args.vgen_cycles
    return args


def main() -> None:
    args = parse_args()
    if args.v_replication_bandwidth < 1:
        raise ValueError("--v-replication-bandwidth must be >= 1")
    cfg = SimConfig(
        n=args.N,
        d=args.D,
        model_dim=args.model_dim,
        a_row_tile=args.a_row_tile,
        reduction_tile=args.reduction_tile,
        v_output_tile=args.v_output_tile,
        vg_reduction_tile=args.vg_reduction_tile,
        vgen_parts_per_v=args.vgen_parts_per_v,
        bytes_per_element=args.bytes_per_element,
        v_buffer_bytes=args.v_buffer_bytes,
        total_engines=args.total_engines,
        qkg_engines=args.qkg_engines,
        vg_engines=args.vg_engines,
        write_a_cycles=args.write_a_cycles,
        write_v_cycles=args.write_v_cycles,
        write_x_cycles_per_part=args.write_x_cycles_per_part,
        vgen_cycles_per_part=args.vgen_cycles_per_part,
        av_compute_cycles=args.av_compute_cycles,
        hiva_a_input_load_cycles=args.hiva_a_input_load_cycles,
        hiva_overlap_a_input_with_compute=args.hiva_overlap_a_input_with_compute,
        viha_v_input_load_cycles=args.viha_v_input_load_cycles,
        viha_overlap_v_input_with_compute=args.viha_overlap_v_input_with_compute,
        replicate_v=args.replicate_v,
        v_replication_bandwidth=args.v_replication_bandwidth,
        viha_mode=args.viha_mode,
        include_parallel_j_upper_bound=args.include_parallel_j_upper_bound,
        output_dir=args.output_dir,
        x_reuse_count=args.x_reuse_count,
    )

    results = run_all(cfg)

    print("track: attention_cim_av_dataflow_cycle_sim")
    print("note: ablation simulator, not exact TP-DCIM paper reproduction")
    print(
        "config: "
        f"N={cfg.n} D={cfg.d} R={cfg.row_blocks} C={cfg.column_blocks} O={cfg.output_blocks} "
        f"vgen_parts={cfg.resolved_vgen_parts_per_v} vbuf_blocks={cfg.v_buffer_capacity_blocks} "
        f"viha_mode={cfg.viha_mode} replicate_v={cfg.replicate_v}"
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

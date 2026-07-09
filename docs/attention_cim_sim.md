# Attention CIM AV Dataflow Simulator

`attention_cim_sim.py` is a small standard-library-only cycle-level simulator for
AV MatMul dataflow ablations in CIM-based Transformer accelerators.

It is not an exact reproduction of the TP-DCIM paper. The goal is to separate
several effects that are coupled in the paper figures:

1. HIVA/V-stationary underutilization,
2. resource-matched V-stationary scheduling,
3. VIHA/A-stationary ping-pong scheduling,
4. V-generation hiding,
5. write and replication overhead.

## Background Model

The simulator targets AV MatMul:

```text
Z = A x V
A: N x N
V: N x D
```

Default blocking:

```text
A row tile       = 192
reduction tile   = 64
V output tile    = 64
R = ceil(N / A row tile)
C = ceil(N / reduction tile)
O = ceil(D / V output tile)
```

The logical AV tile count is:

```text
R * C * O
```

The default resource budget follows the paper-level description:

```text
total CIM engines = 6
QKG / AV engines = 4
VG / V-generation engines = 2
```

Each engine performs one action per cycle:

```text
IDLE
WRITE_A
WRITE_V
WRITE_X
VGEN_COMPUTE
AV_COMPUTE
```

Default latencies are one cycle for each write/compute action, but they are all
CLI-configurable.

## Implemented Schedules

### `hiva_paper_like`

This is a weak V-stationary baseline modeled after Fig. 1(c):

- one VG/WV engine generates a V block,
- one QKG/AV engine writes that V block into CIM,
- the QKG/AV engine sweeps all A row-blocks for that V block,
- the VG engine may generate the next V block during the sweep,
- after that, VG can be idle while the AV engine continues sweeping.

This intentionally captures the paper-like underutilization behavior.

### `hiva_resource_matched_replicate_v`

This uses the same 6-engine budget more aggressively:

- 2 VG engines generate V blocks eagerly,
- 4 QKG engines run V-stationary AV,
- each V block is replicated into all QKG engines,
- row-block AV work for that V block is split across QKG engines.

The replication cost is counted as `WRITE_V` cycles on every QKG copy.

Set `--replicate-v false` to instead run:

```text
hiva_resource_matched_no_replicate
```

In that mode, only one QKG engine stores each V block and sweeps its A row-blocks
serially. Different V blocks can still be assigned to different QKG engines.

### `viha_a_stationary_pingpong`

This models the proposed A-stationary dataflow:

- 2 VG engines generate V blocks,
- 4 QKG engines store A blocks and receive V as input,
- QKG engines are split into two groups:
  - group0: QKG0, QKG1
  - group1: QKG2, QKG3
- one group computes while the other loads the next A blocks,
- groups swap in a ping-pong schedule,
- if a needed V block is not ready, the compute group stalls.

## Metrics

The simulator reports and saves:

- `total_cycles`
- per-engine action breakdown
- total AV tile operations
- useful AV compute engine-cycles
- V-generation engine-cycles
- write engine-cycles by type
- idle engine-cycles
- `qkg_av_compute_utilization`
- `system_av_compute_utilization`
- `system_busy_utilization`
- `write_overhead_fraction`
- V, A, and X approximate reuse counts
- normalized utilization/speedup values relative to `hiva_paper_like`

The paper's Fig. 11 normalization denominator is not fully specified, so this
simulator deliberately reports multiple utilization definitions.

## Usage

Run default N=2048:

```bash
python attention_cim_sim.py
```

Run the paper-size examples:

```bash
python attention_cim_sim.py --N 1024
python attention_cim_sim.py --N 2048
python attention_cim_sim.py --N 4096
```

Override latency assumptions:

```bash
python attention_cim_sim.py \
  --N 4096 \
  --write-a-cycles 1 \
  --write-v-cycles 1 \
  --write-x-cycles 1 \
  --av-compute-cycles 1 \
  --vgen-cycles 1
```

Run resource-matched HIVA without V replication:

```bash
python attention_cim_sim.py --N 2048 --replicate-v false
```

By default, outputs are saved to:

```text
outputs/attention_cim_sim_N{N}_metrics.csv
outputs/attention_cim_sim_N{N}_timeline.csv
```

You can override paths:

```bash
python attention_cim_sim.py \
  --N 2048 \
  --metrics-csv results/hw_av_metrics_N2048.csv \
  --timeline-csv results/hw_av_timeline_N2048.csv
```

## Caveats

- This is a deterministic ablation model, not a cycle-accurate hardware model.
- It does not model NoC contention, SRAM banking, analog MAC timing, ADC/DAC
  latency, or detailed CIM peripheral cost.
- It assumes fixed per-tile write and compute latencies.
- Multi-output-tile behavior is represented through `O = ceil(D / v_output_tile)`;
  the BERT-base head-dimension case usually has `D=64`, so `O=1`.
- The reported normalized utilization should be interpreted as a trend/debugging
  aid rather than as a claim of exact Fig. 11 reproduction.

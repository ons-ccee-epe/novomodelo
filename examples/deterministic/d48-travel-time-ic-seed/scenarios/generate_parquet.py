"""
Generate Parquet scenario files for the d48-travel-time-ic-seed case (the
non-zero windowed initial-condition defluence-seed case).

Run with: uv tool run --with pyarrow -- python3 generate_parquet.py

Produces:
  - inflow_seasonal_stats.parquet
  - load_seasonal_stats.parquet

Determinism: std_m3s = 0.0 and std_mw = 0.0 on every row, so each realized
inflow/load equals its mean exactly (no per-opening noise) and the simulated
trajectory is bit-reproducible.

Topology
========
Cascade U (id 0) -> J (id 1): U carries travel_time_hours = 336.0 on its arc to
J. The study runs 3 weekly (168 h) stages, so the pre-study release window
[start_0 - 336h, start_0) = [2023-12-18, 2024-01-01) at 100 m3/s splits evenly
across the first two study stages (k_0 = k_1 = 1/2) via the IC-anchor overlap.
Both hydros have zero natural inflow AND zero reservoir capacity: the whole
exercise is driven by the in-transit water already seeded into J's stage-0
buckets from U's pre-study release, not by any in-study release or inflow
signal. Demand is 200 MWh per stage, carried by one thermal at 10 $/MWh.
"""

import os
import pyarrow as pa
import pyarrow.parquet as pq

script_dir = os.path.dirname(os.path.abspath(__file__))

N_STAGES = 3

# ── inflow_seasonal_stats.parquet ─────────────────────────────────────────────
#
# Schema: hydro_id INT32, stage_id INT32, mean_m3s FLOAT64, std_m3s FLOAT64
#
# Zero inflow for both hydros at every stage: the case is driven entirely by the
# pre-study defluence seed and the travel-time bucket, not by any inflow signal.
inflow_hydro_ids = []
inflow_stage_ids = []
inflow_mean_m3s = []
inflow_std_m3s = []
for hydro_id in (0, 1):
    for stage_id in range(N_STAGES):
        inflow_hydro_ids.append(hydro_id)
        inflow_stage_ids.append(stage_id)
        inflow_mean_m3s.append(0.0)
        inflow_std_m3s.append(0.0)

inflow_schema = pa.schema(
    [
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("mean_m3s", pa.float64(), nullable=False),
        pa.field("std_m3s", pa.float64(), nullable=False),
    ]
)

inflow_table = pa.table(
    {
        "hydro_id": pa.array(inflow_hydro_ids, type=pa.int32()),
        "stage_id": pa.array(inflow_stage_ids, type=pa.int32()),
        "mean_m3s": pa.array(inflow_mean_m3s, type=pa.float64()),
        "std_m3s": pa.array(inflow_std_m3s, type=pa.float64()),
    },
    schema=inflow_schema,
)

inflow_path = os.path.join(script_dir, "inflow_seasonal_stats.parquet")
pq.write_table(inflow_table, inflow_path, compression="zstd")
print(f"wrote {len(inflow_table)} rows -> {inflow_path}")

# ── load_seasonal_stats.parquet ───────────────────────────────────────────────
#
# Schema: bus_id INT32, stage_id INT32, mean_mw FLOAT64, std_mw FLOAT64
#
# One bus (id 0); 200 MWh per 168 h stage -> mean_mw = 200/168; std 0.0
# (deterministic). One FLAT block per stage, so mean_mw is the whole-stage load.
LOAD_MEAN_MW = 200.0 / 168.0

load_bus_ids = [0] * N_STAGES
load_stage_ids = list(range(N_STAGES))
load_mean_mw = [LOAD_MEAN_MW] * N_STAGES
load_std_mw = [0.0] * N_STAGES

load_schema = pa.schema(
    [
        pa.field("bus_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("mean_mw", pa.float64(), nullable=False),
        pa.field("std_mw", pa.float64(), nullable=False),
    ]
)

load_table = pa.table(
    {
        "bus_id": pa.array(load_bus_ids, type=pa.int32()),
        "stage_id": pa.array(load_stage_ids, type=pa.int32()),
        "mean_mw": pa.array(load_mean_mw, type=pa.float64()),
        "std_mw": pa.array(load_std_mw, type=pa.float64()),
    },
    schema=load_schema,
)

load_path = os.path.join(script_dir, "load_seasonal_stats.parquet")
pq.write_table(load_table, load_path, compression="zstd")
print(f"wrote {len(load_table)} rows -> {load_path}")

print("done.")

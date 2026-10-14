"""
Generate Parquet scenario files for the
d50-travel-time-plain-tributary-confluence case (plain tributary into a
bucketed chronological confluence).

Run with: uv tool run --with pyarrow -- python3 generate_parquet.py

Produces:
  - inflow_seasonal_stats.parquet
  - load_seasonal_stats.parquet

Determinism: std_m3s = 0.0 and std_mw = 0.0 on every row, so each realized
inflow/load equals its mean exactly (no per-opening noise) and the simulated
trajectory is bit-reproducible.

Topology
========
Confluence J (id 2) is fed by TWO upstreams: U (id 1) carries
travel_time_hours = 200.0 on its arc to J (a declared travel-time arc, so J
holds a maturing transit bucket), and V (id 0) is a PLAIN tributary into J with
NO travel_time_hours (a same-stage arc, no maturing bucket). U alone sorts after
V in the id-ordered cascade upstream list, so the arrival-density resolver visits
the plain tributary first.

Stages 0 and 1 are single 168 h weekly parallel senders; stage 2 is a monthly
720 h chronological arrival stage split into blocks [20, 100, 600]. The arrival
stage is at a non-zero index and is chronological, so J's maturing bucket
delivers across its blocks by the arrival-frame delivery density derived from
U's arc alone — the plain tributary V must NOT influence that split. U starts
with 100 hm3 of storage and zero natural inflow: draining it over stages 0 and 1
pushes water into transit that matures at stage 2, where the per-block split
reveals the arrival-frame delivery density. V carries zero water (zero storage,
zero inflow, zero turbine/generation capacity), so J's arrival at stage 2 is the
maturing bucket alone and the split reads U's arrival density. Demand at bus 0
keeps hydro generation valuable so U drains within the horizon.
"""

import os
import pyarrow as pa
import pyarrow.parquet as pq

script_dir = os.path.dirname(os.path.abspath(__file__))

N_STAGES = 3
HYDRO_IDS = (0, 1, 2)

# inflow_seasonal_stats.parquet
#
# Schema: hydro_id INT32, stage_id INT32, mean_m3s FLOAT64, std_m3s FLOAT64
#
# Zero inflow for every hydro at every stage: the case is driven entirely by U's
# initial storage draining through the travel-time bucket, not by any inflow.
inflow_hydro_ids = []
inflow_stage_ids = []
inflow_mean_m3s = []
inflow_std_m3s = []
for hydro_id in HYDRO_IDS:
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

# load_seasonal_stats.parquet
#
# Schema: bus_id INT32, stage_id INT32, mean_mw FLOAT64, std_mw FLOAT64
#
# One bus (id 0); flat 5 MW at every stage (and, absent load_factors.json, every
# block within the chronological stage); std 0.0 (deterministic). 5 MW keeps
# thermal marginal and every hydro MWh valued, which drives U to drain within the
# horizon.
load_bus_ids = [0] * N_STAGES
load_stage_ids = list(range(N_STAGES))
load_mean_mw = [5.0] * N_STAGES
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

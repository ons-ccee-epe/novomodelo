"""
Generate Parquet scenario files for the d42-nonfilling-hydro-commissioning case.

Run with: uv tool run --with pyarrow -- python3 generate_parquet.py

Produces:
  - inflow_seasonal_stats.parquet
  - load_seasonal_stats.parquet

Determinism: std_m3s = 0.0 and std_mw = 0.0 on every row, so each realized
inflow/load equals its mean exactly (no per-opening noise) and the simulated
trajectory is bit-reproducible.

Topology / binding regime
=========================
Three NON-filling hydros (no filling block anywhere):

  - H_new  (id 0): entry_stage_id = 2, downstream = H_down (id 1). Dormant
    (PreFilling reformulation) at stages 0-1, Operating at 2-3.
  - H_down (id 1): no commissioning window, downstream = null (real reservoir
    outlet). Operating at every stage; receives H_new's routed inflow while H_new
    is dormant.
  - H_tail (id 2): entry_stage_id = 2, downstream = null (cascade tail). Dormant
    at 0-1 (its routed inflow exits at the sink), Operating at 2-3.

While H_new is dormant its dam is absent: its turbine/spillage/diversion columns
are pinned [0, 0] and its incremental inflow short-circuits onto H_down's
water-balance row at -zeta (the river flows past the un-built site). H_tail's
inflow has no downstream while dormant, so it is discarded at the sink. H_new and
H_tail carry a large local inflow (40 m3/s) so the routed/discarded volume is far
from zero and the storage trajectory (hence the parity hash) discriminates the
short-circuit reformulation from a naive column-freeze that would trap the water.
"""

import os
import pyarrow as pa
import pyarrow.parquet as pq

script_dir = os.path.dirname(os.path.abspath(__file__))

N_STAGES = 4

# ── inflow_seasonal_stats.parquet ─────────────────────────────────────────────
#
# Schema: hydro_id INT32, stage_id INT32, mean_m3s FLOAT64, std_m3s FLOAT64
#
# Per-(hydro, stage) mean schedule; std is 0.0 everywhere (deterministic).
#
#         stage:    0     1     2     3
# H_new  (0):      40    40    40    40   (dormant 0-1: routed to H_down; large)
# H_down (1):      20    20    20    20   (downstream reservoir, own incremental)
# H_tail (2):      40    40    40    40   (dormant 0-1: discarded at sink; large)
inflow_means = {
    0: [40.0, 40.0, 40.0, 40.0],
    1: [20.0, 20.0, 20.0, 20.0],
    2: [40.0, 40.0, 40.0, 40.0],
}

inflow_hydro_ids = []
inflow_stage_ids = []
inflow_mean_m3s = []
inflow_std_m3s = []
for hydro_id in sorted(inflow_means):
    for stage_id in range(N_STAGES):
        inflow_hydro_ids.append(hydro_id)
        inflow_stage_ids.append(stage_id)
        inflow_mean_m3s.append(inflow_means[hydro_id][stage_id])
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
# One bus (id 0); constant 150 MW mean; std 0.0 (deterministic). Per-block shape
# comes from load_factors.json (one FLAT block per stage).
load_bus_ids = [0] * N_STAGES
load_stage_ids = list(range(N_STAGES))
load_mean_mw = [150.0] * N_STAGES
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

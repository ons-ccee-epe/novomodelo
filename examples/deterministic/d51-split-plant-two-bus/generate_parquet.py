"""
Generate Parquet fixtures for the d51-split-plant-two-bus test case.

Run with: python3 generate_parquet.py

Produces:
  - system/fpha_hyperplanes.parquet
  - system/hydro_energy_productivity.parquet
  - scenarios/inflow_seasonal_stats.parquet
  - scenarios/load_seasonal_stats.parquet
  - constraints/hydro_bounds.parquet
  - constraints/thermal_bounds.parquet

Case: 1 FPHA hydro (H0) with two unit groups on two different buses (B0, B1),
one connecting line, one thermal at B0 with a per-block bound override, 2
stages (stage 0 has 2 blocks: PEAK/OFFPEAK; stage 1 has 1 block).

The FPHA hyperplanes (2 planes, stage_id=null => apply to every stage) are the
same coefficients as d06-fpha-variable-head. H0's two unit groups keep their
own declared per-stage envelope (no per-group override): group H0-B0 stays at
30.0 MW / 42.0 m3/s, group H0-B1 at 20.0 MW / 28.0 m3/s -- their declared sum
(50.0 MW / 70.0 m3/s) sits strictly below the plant's own declared envelope
(60.0 MW / 80.0 m3/s), satisfying the no-raising rule on the group axis with
room to spare. hydro_bounds.parquet instead LOWERS the plant's own resolved
envelope at stage 0 to 28.0 MW / 32.0 m3/s -- between the two groups' declared
values -- so the per-cell `min(group box, plant envelope)` resolution
(spec section 1.3) binds on the PLANT term for H0-B0's cell (28.0 < 30.0) and
on the GROUP term for H0-B1's cell (20.0 < 28.0) at that stage, exercising both
terms of the closing `min` while keeping every group-axis and plant-axis
override within the no-raising rule (lowering only).
"""

import os

import pyarrow as pa
import pyarrow.parquet as pq

script_dir = os.path.dirname(os.path.abspath(__file__))


def write_table(table: pa.Table, relative_path: str) -> None:
    path = os.path.join(script_dir, relative_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    print(f"wrote {len(table)} rows -> {path}")


# ── system/fpha_hyperplanes.parquet ────────────────────────────────────────
#
# Schema: hydro_id (INT32), stage_id (INT32, nullable), plane_id (INT32),
#         gamma_0/gamma_v/gamma_q/gamma_s (FLOAT64), kappa (FLOAT64)
#
# Two precomputed planes, stage_id=null (apply to every stage) -- identical
# coefficients to d06-fpha-variable-head's variable-head planes.

hyperplanes_schema = pa.schema(
    [
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=True),
        pa.field("plane_id", pa.int32(), nullable=False),
        pa.field("gamma_0", pa.float64(), nullable=False),
        pa.field("gamma_v", pa.float64(), nullable=False),
        pa.field("gamma_q", pa.float64(), nullable=False),
        pa.field("gamma_s", pa.float64(), nullable=False),
        pa.field("kappa", pa.float64(), nullable=True),
    ]
)

hyperplanes_table = pa.table(
    {
        "hydro_id": pa.array([0, 0], type=pa.int32()),
        "stage_id": pa.array([None, None], type=pa.int32()),
        "plane_id": pa.array([0, 1], type=pa.int32()),
        "gamma_0": pa.array([0.0, 0.0], type=pa.float64()),
        "gamma_v": pa.array([0.002, 0.001], type=pa.float64()),
        "gamma_q": pa.array([0.80, 0.95], type=pa.float64()),
        "gamma_s": pa.array([0.0, 0.0], type=pa.float64()),
        "kappa": pa.array([1.0, 1.0], type=pa.float64()),
    },
    schema=hyperplanes_schema,
)
write_table(hyperplanes_table, "system/fpha_hyperplanes.parquet")

# ── system/hydro_energy_productivity.parquet ───────────────────────────────
#
# Schema: hydro_id (INT32), stage_id (INT32, nullable),
#         equivalent_productivity_mw_per_m3s (FLOAT64, nullable),
#         reference_outflow_m3s (FLOAT64, nullable),
#         specific_productivity_mw_per_m3s_per_m (FLOAT64, nullable)
#
# Inert for FPHA hydros (Layer 6 productivity resolution skips them), carried
# only for parity with every other FPHA example deck's file set.

productivity_schema = pa.schema(
    [
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=True),
        pa.field("equivalent_productivity_mw_per_m3s", pa.float64(), nullable=True),
        pa.field("reference_outflow_m3s", pa.float64(), nullable=True),
        pa.field("specific_productivity_mw_per_m3s_per_m", pa.float64(), nullable=True),
    ]
)

productivity_table = pa.table(
    {
        "hydro_id": pa.array([0], type=pa.int32()),
        "stage_id": pa.array([None], type=pa.int32()),
        "equivalent_productivity_mw_per_m3s": pa.array([1.0], type=pa.float64()),
        "reference_outflow_m3s": pa.array([None], type=pa.float64()),
        "specific_productivity_mw_per_m3s_per_m": pa.array([None], type=pa.float64()),
    },
    schema=productivity_schema,
)
write_table(productivity_table, "system/hydro_energy_productivity.parquet")

# ── scenarios/inflow_seasonal_stats.parquet ────────────────────────────────
#
# Schema: hydro_id (INT32), stage_id (INT32), mean_m3s (FLOAT64), std_m3s (FLOAT64)
#
# Deterministic (std=0): inflow set close to the expected total plant
# turbined draw at each stage so storage stays comfortably within bounds.

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
        "hydro_id": pa.array([0, 0], type=pa.int32()),
        "stage_id": pa.array([0, 1], type=pa.int32()),
        "mean_m3s": pa.array([70.0, 45.0], type=pa.float64()),
        "std_m3s": pa.array([0.0, 0.0], type=pa.float64()),
    },
    schema=inflow_schema,
)
write_table(inflow_table, "scenarios/inflow_seasonal_stats.parquet")

# ── scenarios/load_seasonal_stats.parquet ──────────────────────────────────
#
# Schema: bus_id (INT32), stage_id (INT32), mean_mw (FLOAT64), std_mw (FLOAT64)
#
# Deterministic load on BOTH buses (both cells must dispatch): B0 45.0 MW,
# B1 10.0 MW, constant across both stages.

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
        "bus_id": pa.array([0, 0, 1, 1], type=pa.int32()),
        "stage_id": pa.array([0, 1, 0, 1], type=pa.int32()),
        "mean_mw": pa.array([45.0, 45.0, 10.0, 10.0], type=pa.float64()),
        "std_mw": pa.array([0.0, 0.0, 0.0, 0.0], type=pa.float64()),
    },
    schema=load_schema,
)
write_table(load_table, "scenarios/load_seasonal_stats.parquet")

# ── constraints/hydro_bounds.parquet ───────────────────────────────────────
#
# Schema: hydro_id (INT32), stage_id (INT32), max_generation_mw (FLOAT64),
#         max_turbined_m3s (FLOAT64)
#
# One stage-wide, plant-axis LOWERING row (no block_id column -> applies to
# every block of stage 0): the resolved plant envelope drops from its declared
# 60.0 MW / 80.0 m3/s to 28.0 MW / 32.0 m3/s, strictly between H0-B0's
# declared 30.0 MW / 42.0 m3/s and H0-B1's declared 20.0 MW / 28.0 m3/s. Stage
# 1 carries no override, so both cells fall back to their own declared value
# there (the plant's declared 60.0/80.0 never binds). Lowering only -- legal
# under the plant-axis no-raising rule (43) exactly as it would be illegal to
# raise.

hydro_bounds_schema = pa.schema(
    [
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("max_generation_mw", pa.float64(), nullable=True),
        pa.field("max_turbined_m3s", pa.float64(), nullable=True),
    ]
)

hydro_bounds_table = pa.table(
    {
        "hydro_id": pa.array([0], type=pa.int32()),
        "stage_id": pa.array([0], type=pa.int32()),
        "max_generation_mw": pa.array([28.0], type=pa.float64()),
        "max_turbined_m3s": pa.array([32.0], type=pa.float64()),
    },
    schema=hydro_bounds_schema,
)
write_table(hydro_bounds_table, "constraints/hydro_bounds.parquet")

# ── constraints/thermal_bounds.parquet ─────────────────────────────────────
#
# Schema: thermal_id (INT32), stage_id (INT32), max_generation_mw (FLOAT64),
#         block_id (INT32, nullable)
#
# T0's per-block override at stage 0 (the multi-block stage): the PEAK block
# (block 0, 200h) caps generation at 8.0 MW, strictly below the OFFPEAK block
# (block 1, 530h) cap of 25.0 MW.

thermal_bounds_schema = pa.schema(
    [
        pa.field("thermal_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("max_generation_mw", pa.float64(), nullable=True),
        pa.field("block_id", pa.int32(), nullable=True),
    ]
)

thermal_bounds_table = pa.table(
    {
        "thermal_id": pa.array([0, 0], type=pa.int32()),
        "stage_id": pa.array([0, 0], type=pa.int32()),
        "max_generation_mw": pa.array([8.0, 25.0], type=pa.float64()),
        "block_id": pa.array([0, 1], type=pa.int32()),
    },
    schema=thermal_bounds_schema,
)
write_table(thermal_bounds_table, "constraints/thermal_bounds.parquet")

print("done.")

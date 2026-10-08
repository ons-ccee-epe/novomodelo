"""
Generate Parquet fixtures for the d52-per-block-thermal-fold test case.

Run with: python3 generate_parquet.py

Produces:
  - scenarios/inflow_seasonal_stats.parquet
  - scenarios/load_seasonal_stats.parquet
  - constraints/thermal_bounds.parquet

Case: 1 bus (B0), two thermals — T-CHEAP (id 0, cost 0.0 $/MWh, base cap
1000.0 MW) and T-SUB (id 1, cost 300.0 $/MWh, cap 1000.0 MW) — no hydro. Stage
0 has two blocks (PEAK 100h, OFFPEAK 300h); stage 1 has one block (SINGLE
400h). T-CHEAP carries a per-block override at stage 0: PEAK caps it at
100.0 MW, strictly below the OFFPEAK cap of 500.0 MW. Deterministic bus load
is 150.0 MW in every block of every stage, so the PEAK-block cap forces T-SUB
to cover the 50.0 MW shortfall there (T-CHEAP is the cheaper source, so the LP
maximizes it before ever dispatching T-SUB) while OFFPEAK and stage 1 stay
uncapped-relative-to-load and dispatch entirely from T-CHEAP.

This is the committed "per-block" configuration used directly by the
peak-block-cap-binds test. The "hours-weighted fold" configuration used by the
fold-delta test is generated at test time (never committed) by overwriting
this file's block rows with a single stage-wide row whose value is the
hours-weighted average of the two block caps.
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


# ── scenarios/inflow_seasonal_stats.parquet ────────────────────────────────
#
# Schema: hydro_id (INT32), stage_id (INT32), mean_m3s (FLOAT64), std_m3s (FLOAT64)
#
# Empty: the case has no hydro plants.

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
        "hydro_id": pa.array([], type=pa.int32()),
        "stage_id": pa.array([], type=pa.int32()),
        "mean_m3s": pa.array([], type=pa.float64()),
        "std_m3s": pa.array([], type=pa.float64()),
    },
    schema=inflow_schema,
)
write_table(inflow_table, "scenarios/inflow_seasonal_stats.parquet")

# ── scenarios/load_seasonal_stats.parquet ──────────────────────────────────
#
# Schema: bus_id (INT32), stage_id (INT32), mean_mw (FLOAT64), std_mw (FLOAT64)
#
# Deterministic (std=0) 150.0 MW load on B0, constant across both stages.

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
        "bus_id": pa.array([0, 0], type=pa.int32()),
        "stage_id": pa.array([0, 1], type=pa.int32()),
        "mean_mw": pa.array([150.0, 150.0], type=pa.float64()),
        "std_mw": pa.array([0.0, 0.0], type=pa.float64()),
    },
    schema=load_schema,
)
write_table(load_table, "scenarios/load_seasonal_stats.parquet")

# ── constraints/thermal_bounds.parquet ─────────────────────────────────────
#
# Schema: thermal_id (INT32), stage_id (INT32), max_generation_mw (FLOAT64),
#         block_id (INT32, nullable)
#
# T-CHEAP's per-block override at stage 0 (the multi-block stage): the PEAK
# block (block 0, 100h) caps generation at 100.0 MW, strictly below the
# OFFPEAK block (block 1, 300h) cap of 500.0 MW. No row for stage 1 (T-CHEAP
# keeps its base 1000.0 MW cap there) and no row for T-SUB (uncapped
# relative to any possible shortfall).

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
        "max_generation_mw": pa.array([100.0, 500.0], type=pa.float64()),
        "block_id": pa.array([0, 1], type=pa.int32()),
    },
    schema=thermal_bounds_schema,
)
write_table(thermal_bounds_table, "constraints/thermal_bounds.parquet")

print("done.")

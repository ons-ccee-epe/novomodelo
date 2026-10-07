"""
Generate Parquet scenario files for d56-external-authoritative test case.

Run with: uv tool run --with pyarrow -- python3 generate_parquet.py

Produces (committed alongside this script):
  - external_load_scenarios.parquet
  - external_inflow_scenarios.parquet

Also produces, at a fixed cross-crate path outside this deck, the reject
fixture the novomodelo-io integration suite composes into a TempDir to exercise
the AR(p > 0) + sigma=0 rejection:
  - crates/novomodelo-io/tests/fixtures/d56_reject_ar_coefficients.parquet
"""

import os
import pyarrow as pa
import pyarrow.parquet as pq

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))

# ── external_load_scenarios.parquet ───────────────────────────────────────────
#
# Schema: stage_id (INT32), scenario_id (INT32), bus_id (INT32), value_mw (FLOAT64)
#
# One scenario (scenario_id=0) per stage, bus_id=1, a constant 123.0 MW at
# every stage -- sigma=0 by construction. This deck carries no
# load_seasonal_stats.parquet at all, so the realized load can only come
# from this external value, never a seasonal mean.

load_stage_ids = [0, 1]
load_scenario_ids = [0, 0]
load_bus_ids = [1, 1]
load_values_mw = [123.0, 123.0]

load_schema = pa.schema(
    [
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("scenario_id", pa.int32(), nullable=False),
        pa.field("bus_id", pa.int32(), nullable=False),
        pa.field("value_mw", pa.float64(), nullable=False),
    ]
)

load_table = pa.table(
    {
        "stage_id": pa.array(load_stage_ids, type=pa.int32()),
        "scenario_id": pa.array(load_scenario_ids, type=pa.int32()),
        "bus_id": pa.array(load_bus_ids, type=pa.int32()),
        "value_mw": pa.array(load_values_mw, type=pa.float64()),
    },
    schema=load_schema,
)

load_path = os.path.join(script_dir, "external_load_scenarios.parquet")
pq.write_table(load_table, load_path, compression="zstd")
print(f"wrote {len(load_table)} rows -> {load_path}")

# ── external_inflow_scenarios.parquet ─────────────────────────────────────────
#
# Schema: stage_id (INT32), scenario_id (INT32), hydro_id (INT32), value_m3s (FLOAT64)
#
# One scenario (scenario_id=0) per stage, hydro_id=1, a constant 80.0 m3/s at
# every stage -- sigma=0. This deck declares no inflow_ar_coefficients.parquet,
# so hydro 1 is AR(0): its deterministic base is exactly this external value.

inflow_stage_ids = [0, 1]
inflow_scenario_ids = [0, 0]
inflow_hydro_ids = [1, 1]
inflow_values_m3s = [80.0, 80.0]

inflow_schema = pa.schema(
    [
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("scenario_id", pa.int32(), nullable=False),
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("value_m3s", pa.float64(), nullable=False),
    ]
)

inflow_table = pa.table(
    {
        "stage_id": pa.array(inflow_stage_ids, type=pa.int32()),
        "scenario_id": pa.array(inflow_scenario_ids, type=pa.int32()),
        "hydro_id": pa.array(inflow_hydro_ids, type=pa.int32()),
        "value_m3s": pa.array(inflow_values_m3s, type=pa.float64()),
    },
    schema=inflow_schema,
)

inflow_path = os.path.join(script_dir, "external_inflow_scenarios.parquet")
pq.write_table(inflow_table, inflow_path, compression="zstd")
print(f"wrote {len(inflow_table)} rows -> {inflow_path}")

# ── crates/novomodelo-io/tests/fixtures/d56_reject_ar_coefficients.parquet ─────────
#
# Schema: hydro_id (INT32), stage_id (INT32), lag (INT32), coefficient (FLOAT64)
#
# One AR(1) row for hydro_id=1 at stage 0 (season_id=0, this deck's only
# season), coefficient=0.5 -- well inside the |psi| < 1 stationarity gate.
# Composed into a TempDir as scenarios/inflow_ar_coefficients.parquet
# alongside this deck's own unchanged files, it makes hydro 1 AR(1) while
# its external inflow column stays constant (sigma=0), which the
# deterministic AR(p > 0) inflow rule rejects.

reject_hydro_ids = [1]
reject_stage_ids = [0]
reject_lags = [1]
reject_coefficients = [0.5]

reject_schema = pa.schema(
    [
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("lag", pa.int32(), nullable=False),
        pa.field("coefficient", pa.float64(), nullable=False),
    ]
)

reject_table = pa.table(
    {
        "hydro_id": pa.array(reject_hydro_ids, type=pa.int32()),
        "stage_id": pa.array(reject_stage_ids, type=pa.int32()),
        "lag": pa.array(reject_lags, type=pa.int32()),
        "coefficient": pa.array(reject_coefficients, type=pa.float64()),
    },
    schema=reject_schema,
)

reject_dir = os.path.join(repo_root, "crates", "novomodelo-io", "tests", "fixtures")
os.makedirs(reject_dir, exist_ok=True)
reject_path = os.path.join(reject_dir, "d56_reject_ar_coefficients.parquet")
pq.write_table(reject_table, reject_path, compression="zstd")
print(f"wrote {len(reject_table)} rows -> {reject_path}")

print("done.")

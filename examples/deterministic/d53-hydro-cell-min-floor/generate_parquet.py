"""
Generate Parquet fixtures for the d53-hydro-cell-min-floor test case.

Run with: python3 generate_parquet.py

Produces:
  - system/fpha_hyperplanes.parquet
  - system/hydro_energy_productivity.parquet
  - scenarios/inflow_seasonal_stats.parquet
  - scenarios/load_seasonal_stats.parquet

Case: 1 FPHA hydro (H0) with two unit groups on two different buses (B0, B1),
one connecting line, one thermal at B0, 2 stages (stage 0 has 2 blocks:
PEAK/OFFPEAK; stage 1 has 1 block) -- the same split-plant topology as
d51-split-plant-two-bus, but WITHOUT that case's max-side group-bounds/
thermal-bounds overrides (this case's point is the min-side per-cell floor,
not the max-side envelope collapse d51 already exercises).

H0-B1 (unit group 1) declares a nonzero `min_turbined_m3s = 27.0` directly in
system/hydros.json (no override needed -- the floor is the declared value).
Inflow is deliberately low (mean_m3s = 3.0 at stage 0, 2.0 at stage 1) so
that, summed with the initial storage (120.0 hm3, `initial_conditions.json`),
the reservoir cannot sustain B1's 27.0 m3/s floor for both stages even with
B0's cell turbining nothing at all: total water available is 120.0 + (3.0 +
2.0) * 730 * 0.0036 = 133.14 hm3, while B1's floor alone demands 27.0 * 730 *
0.0036 * 2 = 141.9 hm3 over the two stages -- a ~8.8 hm3 shortfall that must
surface as `turbine_below_slack` on B1's cell, never a structural cap (B1's
own `max_turbined_m3s` stays 28.0 at every stage, strictly above the floor).

The FPHA hyperplanes (2 planes, stage_id=null => apply to every stage) are the
same coefficients as d06-fpha-variable-head / d51-split-plant-two-bus.
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
# Deterministic (std=0): inflow set deliberately LOW so that, combined with
# the initial storage, the reservoir cannot sustain B1's 27.0 m3/s floor
# across both stages -- see the module docstring's water-budget derivation.

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
        "mean_m3s": pa.array([3.0, 2.0], type=pa.float64()),
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

print("done.")

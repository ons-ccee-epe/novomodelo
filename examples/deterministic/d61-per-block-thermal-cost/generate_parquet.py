"""
Generate Parquet fixtures for the d61-per-block-thermal-cost test case.

Run with: uv run --with pyarrow python generate_parquet.py

Produces:
  - scenarios/inflow_seasonal_stats.parquet
  - scenarios/load_seasonal_stats.parquet
  - constraints/thermal_bounds.parquet

Case: 1 bus (B0) with a deterministic 100.0 MW load, two thermals -- T-A (id 0,
registry cost 400.0 $/MWh, cap 60.0 MW) and T-B (id 1, cost 300.0 $/MWh, cap
60.0 MW) -- and one reservoir hydro H0 (constant productivity 0.36 MW per m3/s,
12.0 hm3 initial storage, 5.0 m3/s deterministic inflow). Two stages, each with
the same two blocks (PEAK 100h, OFFPEAK 300h). T-A prices its generation per
block through constraints/thermal_bounds.parquet: PEAK 500.0, OFFPEAK 100.0,
and a stage-wide row of 200.0 that the block rows override. T-B has no row, so
its registry cost 300.0 applies in every block. No bound is set in any row, so
every cap is the registry 60.0 MW.

Design conditions and their numbers (L = 100 MW load, h_P = 100 h, h_O = 300 h,
cap = 60 MW, rho = 0.36, M3S_TO_HM3 = 0.0036, zeta = (h_P + h_O) * M3S_TO_HM3
= 1.44):

  1. Capacity-limited: cap < L (60 < 100), so the load is not met by one
     thermal and the deficit cost (1000) exceeds every thermal cost
     (1000 > 500 > 300 > 200 > 100).
  2. Per-block pricing is visible in the dispatch: in PEAK, T-B (300) is
     cheaper than T-A (500), so T-B sits at its cap of 60 MW and T-A is the
     interior unit at 40 MW; the PEAK marginal cost is 500. Folding the block
     costs to one stage cost reverses the merit order (see 3).
  3. Hours-weighted mean fold: the weighted mean of T-A's block costs is
     (500 * 100 + 100 * 300) / 400 = 200, which is below T-B's 300. With that
     single stage cost, T-A sits at its cap of 60 MW and T-B is the interior
     unit at 40 MW in both blocks; the marginal cost is 300 in both blocks.
  4. Scarce water and no kinks: total water
     12 + 2 * 5 * 1.44 = 26.4 hm3, i.e. 26.4 * rho / M3S_TO_HM3 = 2640 MWh,
     which is below (L - cap) * h_P = 4000 MWh, so T-A stays interior in PEAK
     even if all the water lands in one stage and the water is priced at the
     PEAK cost 500. The turbine never binds (largest PEAK flow
     26.4 / (h_P * M3S_TO_HM3) = 73.3 m3/s < 200 m3/s; <= 26.4 MW < 72 MW),
     storage stays interior (never above 12 + 7.2 = 19.2 < 100 hm3, no spill),
     and stage 1 always receives 5 * 1.44 = 7.2 hm3 of its own inflow, so the
     trial storage never sits at a zero-storage kink.

The thermal-only configurations (raised caps, drained reservoir) and the
stage-cost fold configuration are generated at test time in temporary copies,
never committed.
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
# Deterministic (std=0) 5.0 m3/s inflow into H0, constant across both stages.

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
        "mean_m3s": pa.array([5.0, 5.0], type=pa.float64()),
        "std_m3s": pa.array([0.0, 0.0], type=pa.float64()),
    },
    schema=inflow_schema,
)
write_table(inflow_table, "scenarios/inflow_seasonal_stats.parquet")

# ── scenarios/load_seasonal_stats.parquet ──────────────────────────────────
#
# Schema: bus_id (INT32), stage_id (INT32), mean_mw (FLOAT64), std_mw (FLOAT64)
#
# Deterministic (std=0) 100.0 MW load on B0, constant across both stages.

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
        "mean_mw": pa.array([100.0, 100.0], type=pa.float64()),
        "std_mw": pa.array([0.0, 0.0], type=pa.float64()),
    },
    schema=load_schema,
)
write_table(load_table, "scenarios/load_seasonal_stats.parquet")

# ── constraints/thermal_bounds.parquet ─────────────────────────────────────
#
# Schema: thermal_id (INT32), stage_id (INT32), block_id (INT32, nullable),
#         min_generation_mw (FLOAT64, nullable), max_generation_mw (FLOAT64,
#         nullable), cost_per_mwh (FLOAT64, nullable)
#
# T-A's cost rows in both stages: a stage-wide row (block_id null) of 200.0 and
# block rows PEAK (block 0) 500.0 and OFFPEAK (block 1) 100.0; the block rows
# take precedence over the stage row. No row sets a bound, so min_generation_mw
# and max_generation_mw are all null (the explicit schema keeps them float64
# rather than the null type an all-None array would infer). T-B has no row.

thermal_bounds_schema = pa.schema(
    [
        pa.field("thermal_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("block_id", pa.int32(), nullable=True),
        pa.field("min_generation_mw", pa.float64(), nullable=True),
        pa.field("max_generation_mw", pa.float64(), nullable=True),
        pa.field("cost_per_mwh", pa.float64(), nullable=True),
    ]
)

thermal_bounds_table = pa.table(
    {
        "thermal_id": pa.array([0, 0, 0, 0, 0, 0], type=pa.int32()),
        "stage_id": pa.array([0, 0, 0, 1, 1, 1], type=pa.int32()),
        "block_id": pa.array([None, 0, 1, None, 0, 1], type=pa.int32()),
        "min_generation_mw": pa.array([None] * 6, type=pa.float64()),
        "max_generation_mw": pa.array([None] * 6, type=pa.float64()),
        "cost_per_mwh": pa.array(
            [200.0, 500.0, 100.0, 200.0, 500.0, 100.0], type=pa.float64()
        ),
    },
    schema=thermal_bounds_schema,
)
write_table(thermal_bounds_table, "constraints/thermal_bounds.parquet")

print("done.")

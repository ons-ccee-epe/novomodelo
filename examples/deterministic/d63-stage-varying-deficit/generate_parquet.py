"""
Generate Parquet fixtures for the d63-stage-varying-deficit test case.

Run with: uv run --with pyarrow python generate_parquet.py

Produces:
  - scenarios/inflow_seasonal_stats.parquet
  - scenarios/load_seasonal_stats.parquet
  - constraints/penalty_overrides_bus_deficit.parquet

Case: d09-multi-deficit (one bus B0 with the entity deficit curve
[10 MW @ 500, uncapped @ 5000] $/MWh, one 30 MW thermal T0 at 10 $/MWh, a
deterministic 50 MW load, no hydro, global deficit curve [uncapped @ 1000]) with
stage 1 split into two blocks, PEAK (id 0, 240 h) and OFFPEAK (id 1, 490 h).
Stage 0 keeps one SINGLE block of 730 h. constraints/penalty_overrides_bus_deficit
.parquet overrides the depth of B0's first deficit segment in stage 1 only:

  - a stage-wide key (stage 1, block_id null): first segment 0.0 MW, last
    segment uncapped;
  - a block key (stage 1, block 0 = PEAK): first segment 18.0 MW, last segment
    uncapped.

Stage 0 has no key, so it uses the entity curve (first segment 10.0 MW). In stage
1, PEAK uses the block key and OFFPEAK, which has no block key, uses the stage
key. Every key lists both segment indices of the bus's curve; the uncapped last
segment is a null depth_mw and the first segment is never null.

Design conditions and their numbers (L = 50 MW load, thermal cap 30 MW,
D = L - 30 = 20 MW of deficit in every block, d = depth of the first deficit
segment in MW, h = block hours):

  1. Capacity-limited: the thermal sits at its cap in every block (10 < 500), so
     the deficit is D = 20 MW and the first segment of depth d at 500 $/MWh is
     followed by (20 - d) MW at 5000 $/MWh. The per-hour cost is
         f(d) = 300 + 500 * d + 5000 * (20 - d),   0 <= d <= 20,
     where 300 = 30 MW * 10 $/MWh is the thermal cost and 500 * d + 5000 * (20 - d)
     is the deficit share.
  2. Deterministic and stateless: std 0, one opening, no hydro, so each stage
     costs h * f(d) whatever the other stage does, and the lower and upper bounds
     are equal at the total below.
  3. Distinct depths: d = 10 (entity curve), 18 (block key) and 0 (stage key) give
     three different per-hour costs.
  4. PEAK 240 h and OFFPEAK 490 h differ, so a block key applied to the wrong
     block changes the total.
  5. The global curve [uncapped @ 1000] has one segment and B0's entity curve has
     two. A resolver that took the global curve as the base would see one segment,
     and the two-segment keys would be refused as a segment-count mismatch, so
     that mistake never produces a plausible total. Priced on the global curve
     alone, the case would cost 2 * 730 * (300 + 1000 * 20) = 29,638,000.

Oracle:

  cell              d   f(d) $/h   hours   cost         deficit share
  stage 0 SINGLE   10     55,300     730    40,369,000   55,000 * 730 = 40,150,000
  stage 1 PEAK     18     19,300     240     4,632,000   19,000 * 240 =  4,560,000
  stage 1 OFFPEAK   0    100,300     490    49,147,000  100,000 * 490 = 49,000,000

  total cost: 94,148,000
  stage deficit cost: stage 0 = 40,150,000, stage 1 = 53,560,000

  Twin (constraints/penalty_overrides_bus_deficit.parquet removed): every cell
  has d = 10, so the total is 2 * 730 * 55,300 = 80,738,000 and the deficit cost
  is 40,150,000 in each stage.

Each precedence mistake gives a different total or a different per-stage vector:

  override ignored                      80,738,000
  block rows ignored                   113,588,000
  stage row ignored                     72,098,000
  block row applied to every block      54,458,000 (stage 0 at 40,369,000)
  block row on the wrong block          73,898,000
  stage row leaks into stage 0         126,998,000
  stage index shifted                   54,458,000 (stage 0 at 14,089,000)

Only the two 54,458,000 totals coincide, and their per-stage vectors differ.
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
# No hydro in this case, so the table has no rows.

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
# Deterministic (std=0) 50.0 MW load on B0, constant across both stages.

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
        "mean_mw": pa.array([50.0, 50.0], type=pa.float64()),
        "std_mw": pa.array([0.0, 0.0], type=pa.float64()),
    },
    schema=load_schema,
)
write_table(load_table, "scenarios/load_seasonal_stats.parquet")

# ── constraints/penalty_overrides_bus_deficit.parquet ──────────────────────
#
# Schema: bus_id (INT32), stage_id (INT32), block_id (INT32, nullable),
#         segment_index (INT32), depth_mw (FLOAT64, nullable)
#
# Stage 1 only. The stage-wide key (block_id null) sets the first segment to
# 0.0 MW; the block key (block 0, PEAK) sets it to 18.0 MW and takes precedence
# over the stage key in PEAK. Segment 1 is the uncapped last segment of the
# bus's curve, so its depth_mw is null in both keys.

override_schema = pa.schema(
    [
        pa.field("bus_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("block_id", pa.int32(), nullable=True),
        pa.field("segment_index", pa.int32(), nullable=False),
        pa.field("depth_mw", pa.float64(), nullable=True),
    ]
)

override_table = pa.table(
    {
        "bus_id": pa.array([0, 0, 0, 0], type=pa.int32()),
        "stage_id": pa.array([1, 1, 1, 1], type=pa.int32()),
        "block_id": pa.array([None, None, 0, 0], type=pa.int32()),
        "segment_index": pa.array([0, 1, 0, 1], type=pa.int32()),
        "depth_mw": pa.array([0.0, None, 18.0, None], type=pa.float64()),
    },
    schema=override_schema,
)
write_table(override_table, "constraints/penalty_overrides_bus_deficit.parquet")
